#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest_snapshot.py — knowledge-vault 快照一次性引导入库脚本（部署线）。

依据：docs/ingest-mapping-rules-v1.md（映射规则 v1）+ 2026-10-05 用户裁决：
- Q1 改判：99 废纸篓整目录排除，不迁入（manifest 记 skipped_excluded，不分配 file_id）。
- Q2 根部业务文件（md+canvas 成对）→ 00 收集箱/000 暂未处理/；
  Q3 根部凭据笔记 → 04 信息/042 个人的 key 等信息/；库级定义文件 AGENTS.md → 目标区根部。
- Q4 顶层「03 Subagent 任务与回执」原样保留；Q5 重名冲突拒绝+报告（state=conflict）；
  Q6 status 全 library；Q7 context_tag=相对路径前 3 层目录名 JSON 数组；
  Q8 v1 仅 md(.md/.markdown)+txt 切块（其余只登记）；Q9 疑似凭据文件=登记+排除向量库；
  Q10 04 信息全部登记+排除向量库（其中 041/042 按规则文档八节「内容永不解析」：
  不改写 frontmatter、不切块、不进 FTS）；Q11 非 md summary 不排期；
  Q12 0212 公司项目 md 照常切块入向量；Q13 无 >50MB 例外（超限只登记）。
- 敏感信息明文入库：文件层面不脱敏、不跳过；Q9/Q10 的「排除」仅指不进 zvec 向量库。

实现要点：
- manifest 先行（JSONL，state/ingest_manifest.jsonl，唯一进度真源）：
  字段 source_rel_path / target_rel_path / size_bytes / sha256_source / file_id /
  batch / state / error / attempts / chunks / content_hash / updated_at。
  state 状态机：pending → copied → registered → indexed；旁路 failed / conflict /
  skipped_excluded。断点续跑按 state 跳过已完成。
- file_id 按目标相对路径 UTF-8 字典序预分配（UTF-8 字节序==码点序，sorted 即可），
  从 1 起单调递增；规则文档九节 9.4：首次入库为一次性引导，允许直接写 index.db
  （登记用 INSERT OR IGNORE + 落库后回读校验），chunks/FTS 走 Store.add_chunks，
  向量走 ZvecStore.upsert_chunks（与运行期管线同主键、同归一化语义）。
- frontmatter 只加不改：md 补缺失的标准 7 字段（file_id/title/context_tag/
  created_at/modified_at/summary/status；已有 created 不动、另补 created_at 优先取其值），
  无 frontmatter 生成头部；改写走「备份 + 临时文件 + 原子替换」，改写后恢复原 mtime。
  041/042 与非 md 一律不改写本体。
- 批次（顺序按规则文档，先小后大）：B0=00 收集箱抽 50 文件端到端引导验证；
  B1=根部文件+assets+00 收集箱其余；B2=03 笔记+顶层 03 Subagent；B3=01 资料；
  B4=02 项目（0212 除外）；B5=04 信息；B6=0212 公司项目；B7=99 废纸篓（全 skip）。
- 每批「拷贝(保 mtime) → sha256 校验(失败重试 1 次) → 登记 →（md+txt 且未排除）
  切块+FTS →（未被向量排除）embedding → zvec」。
- 全程源快照只读；sha256 全量对账（源 vs 落地）。

用法：
  python ingest_snapshot.py generate [--regenerate]     # 生成 manifest（含全量 sha256）
  python ingest_snapshot.py run --only B0               # B0 端到端 + 检索冒烟后停止
  python ingest_snapshot.py run                         # 冒烟缺失则先补，再跑余下批次 + 收尾
  python ingest_snapshot.py run --batches B2,B3         # 只跑指定批（不收尾）
  python ingest_snapshot.py smoke-final [--queries "a;b;c"]
  python ingest_snapshot.py status | report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from knowledge_vault.chunker import (
    MARKDOWN_EXTENSIONS,
    TEXT_EXTENSIONS,
    chunk_markdown,
    chunk_plain,
    split_frontmatter,
)
from knowledge_vault.embedder import BGEM3Embedder
from knowledge_vault.registry import infer_file_type, utc_iso
from knowledge_vault.store import Store
from knowledge_vault.vectorstore import ZvecStore

# ------------------------------------------------------------------ 路径常量

#: 入库源根（默认 = 用户 Obsidian 仓库原位置，单真源）。--source 可覆盖
#: （如指向历史快照做副本区落位）；一律用 ~ 形态，不写入个人绝对路径。
SNAPSHOT_ROOT = Path("~/Documents/obsidian_file")

#: 目标区根：--in-place（原地登记）时 = SNAPSHOT_ROOT（零拷贝、零写入 vault）；
#: 否则为副本区（历史快照落位模式）。
VAULT_ROOT = Path("~/Documents/knowledge-vault_file").expanduser()
STATE_DIR = Path("~/llwwds_application/knowledge-vault/state").expanduser()
TESTDATA_DIR = Path("~/llwwds_application/knowledge-vault/testdata").expanduser()
HF_HOME = Path("~/llwwds_application/knowledge-vault/hf_cache").expanduser()

MANIFEST_PATH = STATE_DIR / "ingest_manifest.jsonl"
LOG_PATH = STATE_DIR / "ingest_run.log"
BACKUP_DIR = STATE_DIR / "ingest_fm_backup"
SMOKE_MARKER = STATE_DIR / "ingest_b0_smoke.json"
REPORT_PATH = STATE_DIR / "ingest_report.json"
GEN_STATS_PATH = STATE_DIR / "ingest_generate_stats.json"

def _errors_path() -> Path:
    return TESTDATA_DIR / f"ingest_errors_{time.strftime('%Y%m%d')}.json"

# ------------------------------------------------------------------ 运行模式（main 按 CLI 参数覆盖）

#: 原地登记：source == vault_root，零拷贝、绝不写 vault 文件（frontmatter 改写强制关闭）
IN_PLACE = False
#: frontmatter 只加不改的总开关（原地模式强制 False）
FM_WRITE = True
#: 用户级路径排除（相对源根的 posix 目录路径；--exclude-dir 与 KV_EXCLUDE_DIRS 合并）
USER_EXCLUDE_RELPATHS: set[str] = set()


def rel_excluded(rel_dir: str, dirname: str) -> bool:
    """判断 (rel_dir, dirname) 组成的目录相对路径是否命中用户路径排除。"""
    rel = f"{rel_dir}/{dirname}" if rel_dir and rel_dir != "." else dirname
    return rel in USER_EXCLUDE_RELPATHS

# ------------------------------------------------------------------ 规则常量

TRASH_TOP = "99 废纸篓"  # Q1 改判：整目录排除
EXCLUDE_DIR_NAMES = {".git", ".obsidian", ".trash", ".claude", "__pycache__"}
EXCLUDE_FILE_NAMES = {".DS_Store", ".gitkeep", ".gitignore", ".gitattributes"}

#: 根部文件特例重定向（Q2/Q3；库级定义文件落目标区根部）
ROOT_REDIRECTS = {
    "AGENTS.md": "AGENTS.md",
    "internship-impact-tree.md": "00 收集箱/000 暂未处理/internship-impact-tree.md",
    "internship-impact-tree.canvas": "00 收集箱/000 暂未处理/internship-impact-tree.canvas",
    "酒馆深客松 API key.md": "04 信息/042 个人的 key 等信息/酒馆深客松 API key.md",
    "酒馆黑客松产品 DeepSeek API key.md": "04 信息/042 个人的 key 等信息/酒馆黑客松产品 DeepSeek API key.md",
}

BATCH_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
B0_SIZE = 50          # B0 引导验证抽样数（00 收集箱）
LARGE_THRESHOLD = 50 * 1024 * 1024   # Q13：>50MB 只登记，无例外
MAX_ATTEMPTS = 3      # 单文件失败重试上限（含首次）
FLUSH_EVERY = 25      # manifest 落盘节奏（每 N 个文件 / 60 秒）

#: 敏感基线清单（快照分析产出，126 项，仅设备本地）
SENSITIVE_BASELINE_PATH = TESTDATA_DIR / "snapshot_analysis" / "sensitive_files.txt"
#: 运行时文件名模式兜底（规则文档八节；不含裸 key/auth 以避免教程误伤，基线已覆盖）
SENSITIVE_NAME_PATTERNS = (
    "api_key", "apikey", "api key", "api-key", "token", "secret",
    "password", "passwd", "凭据", "账号", "密码", "密钥",
)
SENSITIVE_EXTS = (".env", ".pem", ".p12", ".pfx", ".kdbx")

FM_FIELD_ORDER = ("file_id", "title", "context_tag", "created_at", "modified_at",
                  "summary", "status")
FM_DATE_KEYS = ("created", "created_at", "date", "datetime")
FM_MODIFIED_KEYS = ("modified_at", "updated")

#: 全量收尾检索冒烟默认查询（跨目录；可用 --queries 覆盖）
DEFAULT_SMOKE_QUERIES = ["风控 agent 工作任务", "实习成果树 证据", "知识库 文件夹定义"]

log = logging.getLogger("ingest")


# ------------------------------------------------------------------ 工具函数

def sha256_file(path: Path, block: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_tag_for(target_rel: str) -> str:
    """前 3 层目录名的 JSON 数组（Q7）；根部文件为 []。"""
    parts = target_rel.split("/")
    dirs = parts[:-1][:3]
    return json.dumps(dirs, ensure_ascii=False)


def top_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def batch_for(target_rel: str) -> str:
    if target_rel.startswith(TRASH_TOP + "/"):
        return "B7"
    t = target_rel
    if t.startswith("02 项目/021 专业项目/0212 公司项目/"):
        return "B6"
    if t.startswith("02 项目/"):
        return "B4"
    if t.startswith("04 信息/"):
        return "B5"
    if t.startswith(("03 笔记/", "03 Subagent 任务与回执/")):
        return "B2"
    if t.startswith("01 资料/"):
        return "B3"
    # 根部文件（重定向落位）与 assets
    return "B1"


def is_under_04(target_rel: str) -> bool:
    return target_rel.startswith("04 信息/")


def is_041_042(target_rel: str) -> bool:
    parts = target_rel.split("/")
    return (
        len(parts) >= 2
        and parts[0] == "04 信息"
        and (parts[1].startswith("041") or parts[1].startswith("042"))
    )


def load_sensitive_baseline() -> set[str]:
    paths: set[str] = set()
    if not SENSITIVE_BASELINE_PATH.exists():
        log.warning("敏感基线清单缺失: %s", SENSITIVE_BASELINE_PATH)
        return paths
    for line in SENSITIVE_BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            paths.add(parts[2].strip())
        elif len(parts) == 1:
            paths.add(parts[0].strip())
    return paths


SENSITIVE_BASELINE = load_sensitive_baseline()
_sens_memo: dict[str, bool] = {}


def is_sensitive(target_rel: str) -> bool:
    """疑似敏感：基线清单（按快照源路径）+ 文件名模式兜底（Q9）。"""
    if target_rel in _sens_memo:
        return _sens_memo[target_rel]
    base = target_rel.rsplit("/", 1)[-1].lower()
    hit = target_rel in SENSITIVE_BASELINE or any(
        p in base for p in SENSITIVE_NAME_PATTERNS
    ) or base.endswith(SENSITIVE_EXTS)
    _sens_memo[target_rel] = hit
    return hit


def index_eligible(row: dict) -> bool:
    """切块+FTS 资格：md/txt 且 ≤50MB 且不在 041/042（内容永不解析）。"""
    target = row["target_rel_path"]
    ext = Path(target).suffix.lower()
    return (
        ext in TEXT_EXTENSIONS
        and row["size_bytes"] <= LARGE_THRESHOLD
        and not is_041_042(target)
    )


def vector_eligible(row: dict) -> bool:
    """向量资格：可切块 且 非 04 信息全目录（Q10-A） 且 非疑似敏感（Q9-B）。"""
    target = row["target_rel_path"]
    return index_eligible(row) and not is_under_04(target) and not is_sensitive(target)


# ------------------------------------------------------------ frontmatter 处理

_FM_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:(.*)$")


def _fm_top_keys(fm_text: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for line in fm_text.splitlines():
        m = _FM_KEY.match(line)
        if m:
            keys[m.group(1)] = m.group(2).strip()
    return keys


_FM_DATE_RE = re.compile(
    r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"
    r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?日?$"
)


def parse_fm_date(value: str | None) -> str | None:
    """frontmatter 日期值 → 归一化 ISO-8601（纯日期保留 date-only）。"""
    if not value:
        return None
    v = value.strip().strip('"').strip("'").strip()
    if not v:
        return None
    dt = None
    has_time = ":" in v  # 含时间部分的形态必带冒号（HH:MM）；纯日期不带
    try:
        dt = datetime.fromisoformat(v[:-1] + "+00:00" if v.endswith("Z") else v)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y年%m月%d日", "%Y.%m.%d"):
            try:
                dt = datetime.strptime(v, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        m = _FM_DATE_RE.match(v)  # 非补零形态（2026/1/2、2026年1月2日 等）
        if m is None:
            return None
        y, mo, d, hh, mi, ss = m.groups()
        try:
            dt = datetime(
                int(y), int(mo), int(d),
                int(hh or 0), int(mi or 0), int(ss or 0),
            )
        except ValueError:
            return None
        has_time = hh is not None
    if dt is None:
        return None
    if not has_time:
        return dt.date().isoformat()
    return dt.isoformat(timespec="seconds")


def _yaml_str(v: str) -> str:
    return json.dumps(v, ensure_ascii=False)


def build_rewritten(
    orig_text: str,
    *,
    file_id: int,
    title: str,
    context_tag_json: str,
    created_at: str,
    modified_at: str,
    status: str = "library",
) -> str | None:
    """只加不改：补缺失标准字段；无 frontmatter 生成 7 字段头部。无需改写返回 None。"""
    fm, body = split_frontmatter(orig_text)
    if fm is None:
        lines = [
            "---\n",
            f"file_id: {file_id}\n",
            f"title: {_yaml_str(title)}\n",
            f"context_tag: {context_tag_json}\n",
            f"created_at: {created_at}\n",
            f"modified_at: {modified_at}\n",
            "summary:\n",
            f"status: {status}\n",
            "---\n",
        ]
        return "".join(lines) + body
    keys = _fm_top_keys(fm)
    append: list[str] = []
    if "file_id" not in keys:
        append.append(f"file_id: {file_id}\n")
    if "title" not in keys:
        append.append(f"title: {_yaml_str(title)}\n")
    if "context_tag" not in keys:
        append.append(f"context_tag: {context_tag_json}\n")
    if "created_at" not in keys:
        append.append(f"created_at: {created_at}\n")
    if "modified_at" not in keys:
        append.append(f"modified_at: {modified_at}\n")
    if "summary" not in keys:
        append.append("summary:\n")
    if "status" not in keys:
        append.append(f"status: {status}\n")
    if not append:
        return None
    flines = fm.splitlines(keepends=True)
    closing = flines[-1]
    new_fm = "".join(flines[:-1]) + "".join(append) + closing
    return new_fm + body


def rewrite_frontmatter(
    dst: Path, row: dict, *, title: str, context_tag_json: str
) -> tuple[str, str, str, bool, str | None] | None:
    """对目标 md 做只加不改改写。返回 (content_hash12, created_at, mtime_iso,
    rewritten, fm_title)。fm_title 为 frontmatter 中非空 title（登记 title 优先取它，
    规则文档五节 5.5）；无法安全改写（非 UTF-8）返回 None。"""
    st = dst.stat()
    mtime_iso = utc_iso(st.st_mtime)
    raw = dst.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    fm, _ = split_frontmatter(text)
    keys = _fm_top_keys(fm) if fm else {}
    fm_title = None
    existing_title = (keys.get("title") or "").strip().strip('"').strip("'").strip()
    if existing_title:
        fm_title = existing_title
    created_at = mtime_iso
    for k in FM_DATE_KEYS:
        parsed = parse_fm_date(keys.get(k))
        if parsed:
            created_at = parsed
            break
    modified_at = mtime_iso
    for k in FM_MODIFIED_KEYS:
        parsed = parse_fm_date(keys.get(k))
        if parsed:
            modified_at = parsed
            break
    new_text = build_rewritten(
        text,
        file_id=row["file_id"],
        title=title,
        context_tag_json=context_tag_json,
        created_at=created_at,
        modified_at=modified_at,
    )
    if new_text is None:  # 7 字段齐备，无需改写
        return sha256_file(dst)[:12], created_at, mtime_iso, False, fm_title
    new_bytes = new_text.encode("utf-8")
    content_hash = hashlib.sha256(new_bytes).hexdigest()[:12]
    # 备份原文件副本（设备本地，规则文档六节 6.5）
    bpath = BACKUP_DIR / row["target_rel_path"]
    bpath.parent.mkdir(parents=True, exist_ok=True)
    bpath.write_bytes(raw)
    # 临时文件 + 原子替换，改写后恢复原 mtime
    tmp = dst.with_name(dst.name + ".ingest-tmp")
    try:
        tmp.write_bytes(new_bytes)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    os.utime(dst, (st.st_atime, st.st_mtime))
    return content_hash, created_at, mtime_iso, True, fm_title


# ------------------------------------------------------------------- manifest

MANIFEST_FIELDS = (
    "source_rel_path", "target_rel_path", "size_bytes", "sha256_source",
    "file_id", "batch", "state", "error", "attempts", "chunks",
    "content_hash", "updated_at",
)


def _row_sort_key(row: dict):
    fid = row.get("file_id")
    return (fid is None, fid if fid is not None else 0, row["source_rel_path"])


def save_manifest(manifest: dict[str, dict]) -> None:
    manifest_path_tmp = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    with open(manifest_path_tmp, "w", encoding="utf-8") as fh:
        for row in sorted(manifest.values(), key=_row_sort_key):
            fh.write(json.dumps(
                {k: row.get(k) for k in MANIFEST_FIELDS},
                ensure_ascii=False,
            ) + "\n")
    os.replace(manifest_path_tmp, MANIFEST_PATH)


def load_manifest() -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest 不存在，请先执行 generate: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            manifest[row["source_rel_path"]] = row
    return manifest


# -------------------------------------------------------------------- generate

def walk_snapshot() -> tuple[list[dict], list[dict], dict]:
    """遍历快照（零写入）。返回 (included, trash, walk_stats)。

    included: source_rel_path / target_rel_path / size_bytes；
    排除项（隐藏、.gitkeep 等）直接跳过；废纸篓单独收集。
    """
    included: list[dict] = []
    trash: list[dict] = []
    stats = {"files_seen": 0, "excluded_hidden": 0, "excluded_meta": 0,
             "unexpected_root": 0}
    for dirpath, dirnames, filenames in os.walk(SNAPSHOT_ROOT):
        rel_dir = Path(dirpath).relative_to(SNAPSHOT_ROOT).as_posix()
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in EXCLUDE_DIR_NAMES and not d.startswith(".")
            and not rel_excluded(rel_dir, d)
        )
        for name in sorted(filenames):
            if name.startswith("."):
                stats["excluded_hidden"] += 1
                continue
            if name in EXCLUDE_FILE_NAMES:
                stats["excluded_meta"] += 1
                continue
            full = Path(dirpath) / name
            if full.is_symlink():
                stats["excluded_hidden"] += 1
                continue
            stats["files_seen"] += 1
            rel = full.relative_to(SNAPSHOT_ROOT).as_posix()
            size = full.stat().st_size
            if rel.startswith(TRASH_TOP + "/"):
                trash.append({"source_rel_path": rel, "target_rel_path": rel,
                              "size_bytes": size})
                continue
            if "/" not in rel:  # 根部文件：副本模式走特例重定向表；原地模式登记原位
                if IN_PLACE:
                    target = rel
                else:
                    target = ROOT_REDIRECTS.get(name)
                    if target is None:
                        stats["unexpected_root"] += 1
                        included.append({"source_rel_path": rel, "target_rel_path": rel,
                                         "size_bytes": size, "_unexpected_root": True})
                        continue
                    target = target
            else:
                target = rel
            included.append({"source_rel_path": rel, "target_rel_path": target,
                             "size_bytes": size})
    return included, trash, stats


def collect_empty_dirs(included: list[dict]) -> list[str]:
    """空目录（无任何迁入文件的目录，含仅剩排除占位文件的目录）→ mkdir 保留。"""
    non_empty: set[str] = set()
    for e in included:
        target = e["target_rel_path"]
        parts = target.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            non_empty.add("/".join(parts[:i]))
    dirs: list[str] = []
    for dirpath, dirnames, filenames in os.walk(SNAPSHOT_ROOT):
        walk_rel_dir = Path(dirpath).relative_to(SNAPSHOT_ROOT).as_posix()
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in EXCLUDE_DIR_NAMES and not d.startswith(".")
            and not (dirpath == str(SNAPSHOT_ROOT) and d == TRASH_TOP)
            and not rel_excluded(walk_rel_dir, d)
        )
        rel_dir = walk_rel_dir
        if rel_dir == ".":
            continue
        if rel_dir not in non_empty:
            dirs.append(rel_dir)
    return sorted(set(dirs))


def select_b0(rows: list[dict]) -> set[str]:
    """00 收集箱内确定性抽 50：48 个 md/txt（端到端全链）+ 2 个其他（登记路径）。"""
    md = [r for r in rows if Path(r["target_rel_path"]).suffix.lower() in TEXT_EXTENSIONS]
    other = [r for r in rows
             if Path(r["target_rel_path"]).suffix.lower() not in TEXT_EXTENSIONS]

    def evenly(lst: list[dict], k: int) -> list[dict]:
        if len(lst) <= k:
            return list(lst)
        if k == 1:
            return [lst[len(lst) // 2]]
        idx = sorted({round(i * (len(lst) - 1) / (k - 1)) for i in range(k)})
        return [lst[i] for i in idx]

    picked = evenly(md, 48) + evenly(other, min(2, len(other)))
    return {r["source_rel_path"] for r in picked}


def cmd_generate(regenerate: bool) -> None:
    t0 = time.time()
    if MANIFEST_PATH.exists():
        if not regenerate:
            log.info("manifest 已存在，跳过生成（--regenerate 强制重建）: %s", MANIFEST_PATH)
            return
        if regenerate:
            # 防误删进度：已有执行进度时拒绝重建
            with open(MANIFEST_PATH, encoding="utf-8") as fh:
                started = sum(
                    1 for line in fh
                    if '"state": "indexed"' in line or '"state": "registered"' in line
                    or '"state": "copied"' in line
                )
            if started:
                raise SystemExit(
                    f"manifest 已有 {started} 行执行进度，拒绝 --regenerate（如确需重建请人工删除文件）"
                )
            log.info("--regenerate：重建 manifest")
    VAULT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    included, trash, walk_stats = walk_snapshot()
    log.info("遍历完成：迁入候选 %d，废纸篓 %d，walk=%s，耗时 %.1fs",
             len(included), len(trash), walk_stats, time.time() - t0)

    manifest: dict[str, dict] = {}

    # --- 冲突检查（Q5-A 拒绝+报告）：同目标多源 + 大小写折叠
    by_target: dict[str, list[str]] = {}
    for e in included:
        by_target.setdefault(e["target_rel_path"], []).append(e["source_rel_path"])
    conflict_sources: set[str] = set()
    for target, srcs in sorted(by_target.items()):
        if len(srcs) > 1:
            log.error("目标路径冲突（Q5 拒绝）: %s <- %s", target, srcs)
            conflict_sources.update(srcs)
    seen_casefold: dict[str, str] = {}
    for target in sorted(by_target):
        key = target.casefold()
        if key in seen_casefold and seen_casefold[key] != target:
            log.error("大小写折叠冲突（Q5 拒绝）: %s vs %s", seen_casefold[key], target)
            conflict_sources.update(by_target[target])
            conflict_sources.update(by_target[seen_casefold[key]])
        else:
            seen_casefold.setdefault(key, target)

    # --- 全量 sha256（源文件，校验基线）
    hashed = 0
    hash_t0 = time.time()
    for e in included:
        row = {
            "source_rel_path": e["source_rel_path"],
            "target_rel_path": e["target_rel_path"],
            "size_bytes": e["size_bytes"],
            "sha256_source": None,
            "file_id": None,
            "batch": None,
            "state": "pending",
            "error": None,
            "attempts": 0,
            "chunks": None,
            "content_hash": None,
            "updated_at": None,
        }
        if e.get("_unexpected_root"):
            row["state"] = "conflict"
            row["error"] = "未登记的根部散落文件（人工裁决后重跑）"
        elif e["source_rel_path"] in conflict_sources:
            row["state"] = "conflict"
            row["error"] = "特例重定向/大小写折叠路径冲突（Q5-A 拒绝，人工裁决）"
        else:
            row["sha256_source"] = sha256_file(SNAPSHOT_ROOT / e["source_rel_path"])
            hashed += 1
            if hashed % 5000 == 0:
                log.info("sha256 进度 %d/%d（%.1fs）", hashed, len(included),
                         time.time() - hash_t0)
        row["batch"] = batch_for(row["target_rel_path"])
        manifest[row["source_rel_path"]] = row
    log.info("sha256 完成 %d 个文件，耗时 %.1fs", hashed, time.time() - hash_t0)

    # --- B0 抽样（00 收集箱 50 个）
    b0_pool = [r for r in manifest.values()
               if r["target_rel_path"].startswith("00 收集箱/")
               and r["state"] == "pending"]
    for src in select_b0(b0_pool):
        manifest[src]["batch"] = "B0"

    # --- file_id 预分配：目标相对路径 UTF-8 字典序（==码点序），从 1 起
    pending_rows = sorted(
        (r for r in manifest.values() if r["state"] == "pending"),
        key=lambda r: r["target_rel_path"].encode("utf-8"),
    )
    for i, row in enumerate(pending_rows, start=1):
        row["file_id"] = i
    log.info("file_id 预分配完成：1..%d（%d 行；conflict %d 行不分配）",
             len(pending_rows), len(pending_rows),
             sum(1 for r in manifest.values() if r["state"] == "conflict"))

    # --- 废纸篓：整目录排除（Q1 改判），manifest 记 skipped_excluded
    for e in trash:
        manifest[e["source_rel_path"]] = {
            "source_rel_path": e["source_rel_path"],
            "target_rel_path": e["target_rel_path"],
            "size_bytes": e["size_bytes"],
            "sha256_source": None,
            "file_id": None,
            "batch": "B7",
            "state": "skipped_excluded",
            "error": "99 废纸篓整目录排除（2026-10-05 用户裁决 Q1）",
            "attempts": 0,
            "chunks": None,
            "content_hash": None,
            "updated_at": None,
        }

    save_manifest(manifest)
    log.info("manifest 落盘: %s（%d 行）", MANIFEST_PATH, len(manifest))

    # --- 空目录 mkdir 保留
    empty_dirs = collect_empty_dirs(included)
    for d in empty_dirs:
        (VAULT_ROOT / d).mkdir(parents=True, exist_ok=True)
    log.info("空目录 mkdir 保留 %d 个", len(empty_dirs))

    # 批次分布统计
    batch_counter = Counter(r["batch"] for r in manifest.values())
    state_counter = Counter(r["state"] for r in manifest.values())
    gen_stats = {
        "generated_at": utc_iso(),
        "elapsed_sec": round(time.time() - t0, 1),
        "walk": walk_stats,
        "included": len(included),
        "trash": len(trash),
        "by_batch": dict(sorted(batch_counter.items())),
        "by_state": dict(sorted(state_counter.items())),
        "empty_dirs_mkdir": len(empty_dirs),
        "sensitive_baseline_size": len(SENSITIVE_BASELINE),
    }
    GEN_STATS_PATH.write_text(
        json.dumps(gen_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("生成统计: %s", json.dumps(gen_stats, ensure_ascii=False))


# ------------------------------------------------------------------------ run

class Ctx:
    """运行上下文：Store / ZvecStore / Embedder 懒加载 + 计时 + 停止旗标。

    embedding 攒批：向量写入按「跨文件队列」攒到 ``embed_batch_texts`` 条再一次性
    encode（FlagEmbedding 调用内自带按长度排序），减少每文件固定开销；批内文件
    保持 state=registered 直到向量落库成功才置 indexed（崩溃后按 registered 重做，
    配合 index_row 的先清后写保证幂等）。
    """

    def __init__(self, embed_devices: str = "cpu", embed_threads: int = 8,
                 embed_batch: int = 8, embed_batch_texts: int = 64) -> None:
        self.manifest: dict[str, dict] = {}
        self.store: Store | None = None
        self.zvec: ZvecStore | None = None
        self.embedder: BGEM3Embedder | None = None
        self.embed_devices = embed_devices
        self.embed_threads = embed_threads
        self.embed_batch = embed_batch
        self.embed_batch_texts = embed_batch_texts
        self.embed_queue: list[tuple[dict, list[int], list[str]]] = []
        self.stop = False
        self.prune_target = False
        self.prune_stats: dict | None = None
        self.since_flush = 0
        self.last_flush = time.time()
        self.t_copy = 0.0
        self.t_register = 0.0
        self.t_index = 0.0
        self.t_embed = 0.0
        self.fm_rewritten = 0
        self.fm_skipped_decode = 0

    def ensure_store(self) -> Store:
        if self.store is None:
            self.store = Store()
        return self.store

    def ensure_zvec(self) -> ZvecStore:
        if self.zvec is None:
            d = STATE_DIR / "zvec_chunks"
            if (d / "meta.json").is_file():
                self.zvec = ZvecStore.open(d)
                log.info("打开 zvec collection: %s（count=%d）", d, self.zvec.count())
            else:
                self.zvec = ZvecStore.create(d)
                log.info("新建 zvec collection: %s", d)
        return self.zvec

    def ensure_embedder(self) -> BGEM3Embedder:
        if self.embedder is None:
            log.info("加载 bge-m3 embedder（devices=%s threads=%d batch=%d）...",
                     self.embed_devices, self.embed_threads, self.embed_batch)
            self.embedder = BGEM3Embedder(
                batch_size=self.embed_batch,
                threads=self.embed_threads,
                devices=self.embed_devices,
            )
        return self.embedder

    def flush(self, force: bool = False) -> None:
        if force or self.since_flush >= FLUSH_EVERY or time.time() - self.last_flush > 60:
            save_manifest(self.manifest)
            self.since_flush = 0
            self.last_flush = time.time()

    def close(self) -> None:
        if self.zvec is not None:
            self.zvec.close()
            self.zvec = None
        if self.store is not None:
            self.store.close()
            self.store = None


def needs_work(row: dict) -> bool:
    if row["state"] in ("indexed", "skipped_excluded", "skipped", "conflict"):
        return False
    if row["state"] == "registered":
        return index_eligible(row)
    if row["state"] == "failed":
        return row.get("attempts", 0) < MAX_ATTEMPTS and row["sha256_source"]
    return row["state"] in ("pending", "copied")


def ensure_copied(ctx: Ctx, row: dict) -> bool:
    """拷贝（保 mtime）→ sha256 校验（失败重试 1 次，规则文档九节 9.3）。

    原地模式（IN_PLACE）：源即目标，零拷贝，仅对现文件复验 manifest sha256。
    """
    t0 = time.time()
    src = SNAPSHOT_ROOT / row["source_rel_path"]
    try:
        if IN_PLACE:
            if sha256_file(src) == row["sha256_source"]:
                return True
            row["state"] = "failed"
            row["error"] = "sha256 与 manifest 不一致（原地模式不拷贝，请人工核对源文件变更）"
            return False
        dst = VAULT_ROOT / row["target_rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256_file(dst) == row["sha256_source"]:
            return True
        for attempt in range(2):  # 首次 + 重试 1 次
            shutil.copy2(src, dst)
            if sha256_file(dst) == row["sha256_source"]:
                return True
            log.warning("sha256 不一致（第 %d 次）: %s", attempt + 1, row["target_rel_path"])
        row["state"] = "failed"
        row["error"] = "sha256 校验不一致（拷贝重试 1 次后仍失败）"
        return False
    finally:
        ctx.t_copy += time.time() - t0


def register_row(ctx: Ctx, row: dict) -> None:
    """frontmatter 只加不改（md，041/042 除外；原地模式强制不改写）→ 写 documents。"""
    t0 = time.time()
    dst = (SNAPSHOT_ROOT if IN_PLACE else VAULT_ROOT) / row["target_rel_path"]
    st = dst.stat()
    mtime_iso = utc_iso(st.st_mtime)
    size = st.st_size
    content_hash = row["sha256_source"][:12]
    created_at = mtime_iso
    title = dst.stem
    tag_json = context_tag_for(row["target_rel_path"])

    ext = dst.suffix.lower()
    is_md = ext in MARKDOWN_EXTENSIONS
    if is_md and not is_041_042(row["target_rel_path"]) and FM_WRITE and not IN_PLACE:
        result = rewrite_frontmatter(
            dst, row, title=title, context_tag_json=tag_json
        )
        if result is None:
            ctx.fm_skipped_decode += 1
            log.warning("非 UTF-8 md，跳过 frontmatter 改写: %s", row["target_rel_path"])
        else:
            content_hash, created_at, mtime_iso, rewritten, fm_title = result
            if fm_title:  # 规则文档五节 5.5：fm title 非空优先
                title = fm_title
            if rewritten:
                ctx.fm_rewritten += 1
                st = dst.stat()
                size = st.st_size

    conn = ctx.ensure_store().conn
    now = utc_iso()
    conn.execute(
        "INSERT OR IGNORE INTO documents ("
        "file_id, file_path, file_type, size_bytes, is_large, title, context_tag, "
        "summary, status, content_hash, created_at, mtime, registered_at, updated_at, "
        "deleted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            row["file_id"], str(dst), infer_file_type(dst.name), int(size),
            int(size > LARGE_THRESHOLD), title, tag_json, None, "library",
            content_hash, created_at, mtime_iso, now, now, None,
        ),
    )
    db_row = conn.execute(
        "SELECT file_path, content_hash FROM documents WHERE file_id = ?",
        (row["file_id"],),
    ).fetchone()
    if db_row is None:
        dup = conn.execute(
            "SELECT file_id FROM documents WHERE file_path = ? AND deleted_at IS NULL",
            (str(dst),),
        ).fetchone()
        raise RuntimeError(f"登记落库失败 file_id={row['file_id']}，同路径活跃行={dict(dup) if dup else None}")
    if db_row["file_path"] != str(dst):
        raise RuntimeError(
            f"file_id={row['file_id']} 路径不一致: {db_row['file_path']!r} != {str(dst)!r}"
        )
    if db_row["content_hash"] != content_hash:
        conn.execute(
            "UPDATE documents SET content_hash=?, size_bytes=?, mtime=?, title=?, "
            "context_tag=?, updated_at=? WHERE file_id=?",
            (content_hash, int(size), mtime_iso, title, tag_json, now, row["file_id"]),
        )
    row["content_hash"] = content_hash
    ctx.t_register += time.time() - t0


def index_row(ctx: Ctx, row: dict) -> int:
    """切块 + FTS +（未被排除时）攒批 embedding + zvec。返回 chunk 数。

    幂等：先清旧（FTS chunks + 向量）再重写（与运行期管线同语义）。向量写入
    走跨文件攒批队列，本文件保持 registered 直到向量落库（见 flush_embed_queue）。
    """
    t0 = time.time()
    dst = VAULT_ROOT / row["target_rel_path"]
    ext = dst.suffix.lower()
    text = dst.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_markdown(text) if ext in MARKDOWN_EXTENSIONS else chunk_plain(text)
    n = len(chunks)
    store = ctx.ensure_store()
    store.delete_chunks_for_file(row["file_id"])  # 幂等清旧（空则 no-op）
    if vector_eligible(row):
        ctx.ensure_zvec().delete_file(row["file_id"])  # 幂等清旧（无向量则 no-op）
    if n == 0:
        ctx.t_index += time.time() - t0
        return 0
    texts = [c.text for c in chunks]
    store.add_chunks(row["file_id"], texts)  # FTS（chunks 表 + chunks_fts）
    if not vector_eligible(row):
        # 无向量资格（Q9/Q10 排除或登记类）：FTS 即终态
        ctx.t_index += time.time() - t0
        return n
    ctx.embed_queue.append((row, [c.seq for c in chunks], texts))
    ctx.t_index += time.time() - t0
    return n


def flush_embed_queue(ctx: Ctx, stats: Counter, *, force: bool = False) -> None:
    """攒批 encode + 按文件回写向量；队列文本数达阈值或批次结束时调用。"""
    queued = ctx.embed_queue
    if not queued:
        return
    total = sum(len(texts) for _, _, texts in queued)
    if not force and total < ctx.embed_batch_texts:
        return
    texts_all: list[str] = []
    spans: list[tuple[dict, list[int], int, int]] = []
    for row, seqs, texts in queued:
        spans.append((row, seqs, len(texts_all), len(texts_all) + len(texts)))
        texts_all.extend(texts)
    try:
        te = time.time()
        vecs = ctx.ensure_embedder().encode(texts_all)
        ctx.t_embed += time.time() - te
        z = ctx.ensure_zvec()
        for row, seqs, s, e in spans:
            z.upsert_chunks(row["file_id"], seqs, vecs[s:e], ["chunk"] * len(seqs))
            row["state"] = "indexed"
            stats["indexed"] += 1
            stats["vector"] += 1
        log.info("向量攒批落库: %d 文件 / %d chunks（累计 embed %.0fs）",
                 len(spans), total, ctx.t_embed)
    except Exception as exc:  # noqa: BLE001 — 攒批失败整批标 failed，断点重做
        for row, _, _, _ in spans:
            row["state"] = "failed"
            row["error"] = f"embed_batch: {type(exc).__name__}: {exc}"
            stats["failed"] += 1
        log.exception("向量攒批失败（%d 文件 / %d chunks）", len(spans), total)
    finally:
        ctx.embed_queue = []


def process_row(ctx: Ctx, row: dict, stats: Counter) -> None:
    try:
        row["attempts"] = row.get("attempts", 0) + 1
        row["updated_at"] = utc_iso()
        if row["state"] in ("pending", "failed"):
            if not ensure_copied(ctx, row):
                stats["failed"] += 1
                return
            row["state"] = "copied"
        # state == copied → 登记
        register_row(ctx, row)
        row["state"] = "registered"
        # state == registered → 索引（有资格才做）
        if index_eligible(row):
            row["chunks"] = index_row(ctx, row)
            if not vector_eligible(row) or not row["chunks"]:
                # Q9/Q10 排除或登记类：FTS 即终态；0 chunk 文件无可嵌入内容
                row["state"] = "indexed"
                stats["indexed"] += 1
                if not vector_eligible(row):
                    stats["fts_only"] += 1
            # 向量资格文件：state 由 flush_embed_queue 推进为 indexed
        else:
            stats["registered_only"] += 1
    except Exception as exc:  # noqa: BLE001 — 单文件失败不阻塞批次
        row["state"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        stats["failed"] += 1
        log.exception("处理失败: %s", row["target_rel_path"])
    finally:
        flush_embed_queue(ctx, stats)
        ctx.since_flush += 1
        ctx.flush()


def run_batch(ctx: Ctx, batch: str, rows: list[dict]) -> Counter:
    stats: Counter = Counter()
    todo = [r for r in rows if needs_work(r)]
    log.info("[%s] 开始：批内 %d 行，需处理 %d", batch, len(rows), len(todo))
    t0 = time.time()
    done_since_log = 0
    for row in todo:
        if ctx.stop:
            log.warning("[%s] 收到停止信号，中断批次（进度已按 manifest 保留）", batch)
            break
        before = row["state"]
        process_row(ctx, row, stats)
        if row["state"] != before or row["state"] == "failed":
            done_since_log += 1
        if done_since_log >= 500:
            done_since_log = 0
            log.info("[%s] 进度: %s", batch, dict(stats))
    # 批内失败重试（attempts 未达上限）
    if ctx.stop:
        flush_embed_queue(ctx, stats, force=True)  # 停止前把已入队向量落库
    retryable = [r for r in todo
                 if r["state"] == "failed" and r.get("attempts", 0) < MAX_ATTEMPTS]
    if retryable and not ctx.stop:
        log.info("[%s] 重试 %d 个失败文件", batch, len(retryable))
        for row in retryable:
            if ctx.stop:
                break
            process_row(ctx, row, stats)
    flush_embed_queue(ctx, stats, force=True)  # 批末强制落向量
    ctx.flush(force=True)
    log.info(
        "[%s] 批次完成: %s | 耗时 %.1fs（copy %.1fs / register %.1fs / index %.1fs / embed %.1fs）",
        batch, dict(stats), time.time() - t0,
        ctx.t_copy, ctx.t_register, ctx.t_index, ctx.t_embed,
    )
    return stats


# ------------------------------------------------------------------ 检索冒烟

def _cjk_query_from_text(text: str) -> str | None:
    body = split_frontmatter(text)[1] or text
    runs = re.findall(r"[\u4e00-\u9fff]{8,}", body)
    if runs:
        mid = max(runs, key=len)
        start = max(0, (len(mid) - 12) // 2)
        return mid[start:start + 12]
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", body)
    return words[0] if words else None


def smoke_b0(ctx: Ctx) -> bool:
    """B0 端到端冒烟：自查询命中源文件 + 3 个通用中文查询有命中。"""
    store = ctx.ensure_store()
    rows = [r for r in ctx.manifest.values()
            if r["batch"] == "B0" and r["state"] == "indexed" and r.get("chunks")]
    result: dict = {"at": utc_iso(), "generic": {}, "self": {}}
    ok = False
    if rows:
        pick = max(rows, key=lambda r: r["chunks"])
        text = (VAULT_ROOT / pick["target_rel_path"]).read_text(
            encoding="utf-8", errors="replace"
        )
        q = _cjk_query_from_text(text)
        if q:
            hits = store.search(q, limit=10)
            self_hit = any(h.file_id == pick["file_id"] for h in hits)
            result["self"] = {"query": q, "hit": self_hit,
                              "file_id": pick["file_id"],
                              "hits": len(hits)}
            ok = ok or self_hit
    for q in ("AI Agent", "知识库", "学习方法"):
        n = len(store.search(q, limit=5))
        result["generic"][q] = n
        ok = ok or n > 0
    result["pass"] = bool(ok)
    SMOKE_MARKER.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("B0 冒烟结果: %s", json.dumps(result, ensure_ascii=False))
    if not result["self"] or not result["self"].get("hit"):
        log.warning("B0 冒烟：自查询未命中源文件（需人工检查）")
    return bool(ok)


# ---------------------------------------------------------------------- 收尾

def prune_target(manifest: dict) -> dict:
    """删除目标区内不在 manifest target 集合中的多余文件与空目录（镜像精确保养）。

    用户授权口径：副本区旧内容在对账确认后删除，防止多份副本累积耗尽磁盘。
    保留例外：README.md（目标区说明文件）。仅操作 VAULT_ROOT 内部。
    """
    targets = {r["target_rel_path"] for r in manifest.values()}
    removed_files, removed_dirs = 0, 0
    for dirpath, dirnames, filenames in os.walk(VAULT_ROOT, topdown=False):
        rel_dir = Path(dirpath).relative_to(VAULT_ROOT).as_posix()
        for name in filenames:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if rel == "README.md" or rel in targets:
                continue
            (Path(dirpath) / name).unlink()
            removed_files += 1
        if rel_dir != ".":
            try:
                Path(dirpath).rmdir()  # 仅空目录成功
                removed_dirs += 1
            except OSError:
                pass
    log.info("prune_target 完成: 删除多余文件 %d 个、空目录 %d 个", removed_files, removed_dirs)
    return {"removed_files": removed_files, "removed_dirs": removed_dirs}


def finalize(ctx: Ctx) -> dict:
    """zvec optimize 一次 + FTS optimize/checkpoint + 最终冒烟 + 汇总报告。"""
    ctx.flush(force=True)
    timings: dict = {}
    t0 = time.time()
    if ctx.zvec is not None:
        ctx.zvec.flush()
        ctx.zvec.optimize()
        timings["zvec_optimize_sec"] = round(time.time() - t0, 1)
        log.info("zvec optimize 完成（%.1fs），count=%d",
                 timings["zvec_optimize_sec"], ctx.zvec.count())
    elif (STATE_DIR / "zvec_chunks" / "meta.json").is_file():
        z = ZvecStore.open(STATE_DIR / "zvec_chunks")
        try:
            z.flush()
            z.optimize()
            timings["zvec_optimize_sec"] = round(time.time() - t0, 1)
            log.info("zvec optimize 完成（%.1fs），count=%d",
                     timings["zvec_optimize_sec"], z.count())
        finally:
            z.close()

    store = ctx.ensure_store()
    conn = store.conn
    t1 = time.time()
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    conn.execute("PRAGMA optimize")
    ckpt = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    timings["fts_optimize_sec"] = round(time.time() - t1, 1)
    timings["wal_checkpoint"] = list(ckpt) if ckpt else None
    log.info("FTS optimize 完成（%.1fs）", timings["fts_optimize_sec"])

    report = build_report(ctx, timings)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("汇总报告: %s", REPORT_PATH)
    log.info("=== 汇总 ===\n%s", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def db_counts(conn: sqlite3.Connection) -> dict:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])
    return {
        "documents_total": one("SELECT COUNT(*) FROM documents"),
        "documents_active": one("SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL"),
        "documents_is_large": one(
            "SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL AND is_large = 1"
        ),
        "documents_by_status": {
            r[0]: int(r[1]) for r in conn.execute(
                "SELECT status, COUNT(*) FROM documents WHERE deleted_at IS NULL GROUP BY status"
            )
        },
        "chunks_total": one("SELECT COUNT(*) FROM chunks"),
        "chunks_by_kind": {
            r[0]: int(r[1]) for r in conn.execute(
                "SELECT kind, COUNT(*) FROM chunks GROUP BY kind"
            )
        },
        "fts_rows": one("SELECT COUNT(*) FROM chunks_fts"),
        "edges": one("SELECT COUNT(*) FROM edges"),
        "file_id_min": one("SELECT COALESCE(MIN(file_id), 0) FROM documents"),
        "file_id_max": one("SELECT COALESCE(MAX(file_id), 0) FROM documents"),
        "file_id_distinct": one("SELECT COUNT(DISTINCT file_id) FROM documents"),
    }


def build_report(ctx: Ctx, timings: dict) -> dict:
    by_state = Counter(r["state"] for r in ctx.manifest.values())
    by_batch: dict[str, dict] = {}
    for r in ctx.manifest.values():
        b = by_batch.setdefault(r["batch"], Counter())
        b[r["state"]] += 1
        if r["state"] == "indexed" and r.get("chunks") is not None:
            b["_chunks"] += r["chunks"]
    included = [r for r in ctx.manifest.values() if r["file_id"] is not None]
    completed = [
        r for r in included
        if r["state"] == "indexed"
        or (r["state"] == "registered" and not index_eligible(r))
    ]
    expected_chunks = sum(r.get("chunks") or 0 for r in ctx.manifest.values()
                          if r["state"] == "indexed")
    expected_vec = sum(r.get("chunks") or 0 for r in ctx.manifest.values()
                       if r["state"] == "indexed" and vector_eligible(r))
    failed_rows = [r for r in ctx.manifest.values() if r["state"] == "failed"]
    conflict_rows = [r for r in ctx.manifest.values() if r["state"] == "conflict"]
    sha_fail = [r for r in failed_rows if "sha256" in (r.get("error") or "")]
    store = ctx.ensure_store()
    counts = db_counts(store.conn)
    conservation = {
        "manifest_included": len(included),
        "manifest_completed": len(completed),
        "db_documents": counts["documents_active"],
        "manifest_chunks": expected_chunks,
        "db_chunks": counts["chunks_total"],
        "manifest_vector_chunks": expected_vec,
        "file_ids_contiguous": (
            counts["file_id_min"] == 1
            and counts["file_id_distinct"] == len(included)
            and counts["file_id_max"] == len(included)
        ),
    }
    conservation["ok"] = (
        conservation["manifest_included"] == conservation["manifest_completed"]
        == conservation["db_documents"]
        and conservation["manifest_chunks"] == conservation["db_chunks"]
    )
    errors_path = _errors_path()
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(
        json.dumps(
            {
                "failed": [
                    {k: r.get(k) for k in ("source_rel_path", "target_rel_path",
                                           "batch", "state", "error", "attempts")}
                    for r in failed_rows
                ],
                "conflict": [
                    {k: r.get(k) for k in ("source_rel_path", "target_rel_path",
                                           "batch", "state", "error")}
                    for r in conflict_rows
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "finished_at": utc_iso(),
        "manifest": str(MANIFEST_PATH),
        "totals_by_state": dict(sorted(by_state.items())),
        "by_batch": {
            b: {k: v for k, v in dict(c).items()} for b, c in sorted(by_batch.items())
        },
        "db": counts,
        "vector": {"dir": str(STATE_DIR / "zvec_chunks"),
                   "count": ctx.zvec.count() if ctx.zvec else None},
        "conservation": conservation,
        "failed_count": len(failed_rows),
        "conflict_count": len(conflict_rows),
        "sha256_verify_failed": len(sha_fail),
        "errors_file": str(errors_path),
        "fm": {"rewritten": ctx.fm_rewritten,
               "decode_skip": ctx.fm_skipped_decode},
        "timings": timings,
    }


def smoke_final(queries: list[str]) -> None:
    """kv-search 冒烟：3 个中文查询跨目录（FTS + 向量 + 真实 kv CLI 一发）。"""
    store = Store()
    embedder = BGEM3Embedder()
    zvec_dir = STATE_DIR / "zvec_chunks"
    z = ZvecStore.open(zvec_dir) if (zvec_dir / "meta.json").is_file() else None
    try:
        for q in queries:
            fts = store.search(q, limit=5)
            print(f"\n=== 查询: {q!r} ===")
            print(f"  FTS 命中 {len(fts)} 条：")
            for h in fts:
                d = store.get_file(h.file_id)
                print(f"    [FTS] f{h.file_id} {d.file_path if d else '?'} (bm25={h.score:.2f})")
            if z is not None:
                vec = embedder.encode([q])
                hits = z.search(vec[0], topk=5)
                print(f"  向量命中 {len(hits)} 条：")
                for h in hits:
                    d = store.get_file(h.file_id)
                    print(f"    [VEC] {h.doc_id} {d.file_path if d else '?'} (dist={h.score:.4f})")
    finally:
        if z is not None:
            z.close()
        store.close()
    # 真实 kv CLI 入口一发（--no-rerank 提速；验证 kv-search 端到端可用）
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_HOME", str(HF_HOME))
    kv_bin = Path(sys.executable).parent / "kv"
    if kv_bin.exists():
        try:
            out = subprocess.run(
                [str(kv_bin), "kv-search", queries[0], "--no-rerank", "--top-j", "5"],
                capture_output=True, text=True, timeout=600, env=env,
            )
            lines = [l for l in out.stdout.splitlines() if l.strip()]
            print(f"\n=== kv CLI kv-search({queries[0]!r}, --no-rerank) ===")
            print(f"  退出码={out.returncode}，JSONL 输出 {len(lines)} 行")
            for l in lines[:3]:
                print("    " + l[:200])
            if out.returncode != 0:
                print("  stderr:", out.stderr[-500:])
        except Exception as exc:  # noqa: BLE001
            print(f"kv CLI 冒烟失败: {exc}")
    else:
        print(f"kv CLI 不存在: {kv_bin}（跳过）")


# ----------------------------------------------------------------------- main

def cmd_run(only: str | None, batches: str | None, *, embed_devices: str = "cpu",
            embed_threads: int = 8, embed_batch: int = 8,
            embed_batch_texts: int = 64, prune_target: bool = False) -> None:
    ctx = Ctx(embed_devices=embed_devices, embed_threads=embed_threads,
              embed_batch=embed_batch, embed_batch_texts=embed_batch_texts)
    ctx.manifest = load_manifest()

    def _handle_stop(signum, frame):  # noqa: ANN001
        ctx.stop = True
        log.warning("收到信号 %s，将在当前文件处理后停止", signum)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    store = ctx.ensure_store()
    counts = db_counts(store.conn)
    log.info("库现状: %s", json.dumps(counts))

    if only:
        sequence = [only]
    elif batches:
        want = {b.strip() for b in batches.split(",") if b.strip()}
        sequence = [b for b in BATCH_ORDER if b in want]
    else:
        sequence = list(BATCH_ORDER)

    # 全量模式（无 --only/--batches）：确保 B0 完成且冒烟通过后再继续
    if not only and not batches:
        b0_rows = [r for r in ctx.manifest.values() if r["batch"] == "B0"]
        if any(needs_work(r) for r in b0_rows):
            run_batch(ctx, "B0", b0_rows)
        if not SMOKE_MARKER.exists():
            ok = smoke_b0(ctx)
            if not ok:
                raise SystemExit("B0 检索冒烟未通过，终止全量（详见 ingest_run.log）")
        else:
            log.info("B0 冒烟标记已存在，跳过: %s", SMOKE_MARKER)

    total_stats: Counter = Counter()
    try:
        for b in sequence:
            rows = [r for r in ctx.manifest.values() if r["batch"] == b]
            stats = run_batch(ctx, b, rows)
            total_stats.update(stats)
            if ctx.stop:
                break
        if only == "B0":
            smoke_b0(ctx)
        elif not only and not batches and not ctx.stop:
            if ctx.prune_target:
                ctx.prune_stats = prune_target(ctx.manifest)
            finalize(ctx)
    finally:
        ctx.flush(force=True)
        ctx.close()
    log.info("本次运行合计: %s", dict(total_stats))


def cmd_status() -> None:
    manifest = load_manifest()
    by_state = Counter(r["state"] for r in manifest.values())
    by_batch: dict[str, Counter] = {}
    for r in manifest.values():
        by_batch.setdefault(r["batch"], Counter())[r["state"]] += 1
    print("by_state:", dict(sorted(by_state.items())))
    for b in sorted(by_batch):
        print(f"  {b}: {dict(sorted(by_batch[b].items()))}")


def cmd_report() -> None:
    ctx = Ctx()
    ctx.manifest = load_manifest()
    try:
        report = build_report(ctx, {})
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        ctx.close()


def main(argv: list[str] | None = None) -> int:
    # embedding 环境注入（部署实例：离线 HF + 本地缓存）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = argparse.ArgumentParser(
        description="knowledge-vault 入库（默认落位模式：源 obsidian_file → 目标 knowledge-vault_file；--in-place 原地登记）"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_gen = sub.add_parser("generate", help="生成 manifest（全量 sha256 + file_id 预分配）")
    p_gen.add_argument("--regenerate", action="store_true")
    p_run = sub.add_parser("run", help="执行批次（默认 B0→B7 全量 + 收尾）")
    p_run.add_argument("--only", choices=BATCH_ORDER, default=None)
    p_run.add_argument("--batches", default=None, help="逗号分隔，如 B2,B3（不收尾）")
    p_run.add_argument("--embed-devices", default="cpu", choices=["cpu", "mps"])
    p_run.add_argument("--embed-threads", type=int, default=8)
    p_run.add_argument("--embed-batch", type=int, default=8,
                       help="encode 内部 forward 批大小")
    p_run.add_argument("--embed-batch-texts", type=int, default=64,
                       help="跨文件攒批触发阈值（队列文本数）")
    p_run.add_argument("--prune-target", action="store_true",
                       help="全量收尾前删除目标区内不在 manifest 中的多余文件（镜像精确保养，README.md 除外）")
    p_smoke = sub.add_parser("smoke-final", help="最终检索冒烟（3 中文查询跨目录）")
    p_smoke.add_argument("--queries", default=None, help="分号分隔，覆盖默认查询")
    sub.add_parser("status", help="manifest 状态速览")
    sub.add_parser("report", help="基于 manifest + 库的汇总报告")
    for p in sub.choices.values():
        p.add_argument("--source", default=None, metavar="PATH",
                       help="入库源根目录（默认 ~/Documents/obsidian_file）")
        p.add_argument("--vault-root", default=None, metavar="PATH",
                       help="目标区根目录（默认 ~/Documents/knowledge-vault_file；--in-place 时忽略）")
        p.add_argument("--in-place", action="store_true",
                       help="原地登记：零拷贝、绝不写源文件（frontmatter 改写强制关闭）")
        p.add_argument("--no-fm-write", action="store_true",
                       help="跳过 frontmatter 改写（登记信息只进索引层）")
        p.add_argument("--exclude-dir", action="append", default=None, metavar="RELPATH",
                       help="用户级排除目录（相对源根，可重复）；另读 KV_EXCLUDE_DIRS（逗号分隔）")
    args = parser.parse_args(argv)

    # CLI 参数 → 运行模式全局覆盖（须在调度前完成）
    global SNAPSHOT_ROOT, VAULT_ROOT, IN_PLACE, FM_WRITE, USER_EXCLUDE_RELPATHS
    if args.source:
        SNAPSHOT_ROOT = Path(args.source).expanduser()
    if args.vault_root:
        VAULT_ROOT = Path(args.vault_root).expanduser()
    IN_PLACE = bool(args.in_place)
    FM_WRITE = (not IN_PLACE) and (not args.no_fm_write)
    if IN_PLACE:
        VAULT_ROOT = SNAPSHOT_ROOT  # 原地登记：目标区即源区
    USER_EXCLUDE_RELPATHS = {p.strip().strip("/") for p in (args.exclude_dir or []) if p.strip()}
    for part in os.environ.get("KV_EXCLUDE_DIRS", "").split(","):
        if part.strip():
            USER_EXCLUDE_RELPATHS.add(part.strip().strip("/"))
    log.info(
        "模式: %s | 源: %s | 目标: %s | fm_write=%s | 用户排除: %s",
        "in-place" if IN_PLACE else "copy-to-vault", SNAPSHOT_ROOT, VAULT_ROOT,
        FM_WRITE, sorted(USER_EXCLUDE_RELPATHS) or "无",
    )

    t0 = time.time()
    if args.command == "generate":
        cmd_generate(args.regenerate)
    elif args.command == "run":
        cmd_run(
            args.only, args.batches,
            embed_devices=args.embed_devices,
            embed_threads=args.embed_threads,
            embed_batch=args.embed_batch,
            embed_batch_texts=args.embed_batch_texts,
            prune_target=args.prune_target,
        )
    elif args.command == "smoke-final":
        queries = (
            [q.strip() for q in args.queries.split(";") if q.strip()]
            if args.queries else DEFAULT_SMOKE_QUERIES
        )
        smoke_final(queries)
    elif args.command == "status":
        cmd_status()
    elif args.command == "report":
        cmd_report()
    log.info("命令 %s 完成，总耗时 %.1fs", args.command, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

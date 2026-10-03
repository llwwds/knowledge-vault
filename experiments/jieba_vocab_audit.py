#!/usr/bin/env python3
"""Spike 5：jieba 切分质量抽查与自定义词典初版。

从真实 Obsidian vault（只读）分层抽取 20 个 md 文件，剥离 frontmatter 与代码块后
用 jieba 精确模式分词，自动检测「技术词被切碎」的模式；可选加载 userdict 复测，
输出 before/after 对比。真实样本与含真实内容的明细只写设备本地 testdata（永不进 git），
仓库内不出现任何真实笔记标题、原句、用户名、私人路径。

用法（容器内）：
    docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/jieba_vocab_audit.py
    docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python \
        experiments/jieba_vocab_audit.py --userdict experiments/userdict_v0.txt

默认路径为容器内挂载点：--vault /vault（只读）、--out /app_runtime/testdata/jieba_audit。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jieba

DEFAULT_VAULT = "/vault"
DEFAULT_OUT = "/app_runtime/testdata/jieba_audit"
DEFAULT_USERDICT = "/repo/experiments/userdict_v0.txt"

SEED = 20261003
TOTAL_SAMPLE = 20
PER_STRATUM_CAP = 5
MIN_STRATUM_FILES = 5  # 一级目录 md 数低于该值不参与分层（附件目录、根部零散文件）

# 只读排除项：软删除 / 快照 / 配置 / 版本库 / 隐藏目录
EXCLUDED_DIRS = {
    "99 废纸篓",
    ".obsidian",
    ".trash",
    ".git",
    "obsidian仓库快照备份",
    ".laneweave",
    ".cairn",
}

# 抽样隐私护栏：文件名疑似凭据的文件不进入样本池（计数上报，不落路径）
CREDENTIAL_NAME = re.compile(
    r"(api[_ -]?key|secret|token|password|passwd|credential|密钥|凭据|密码)", re.I
)

# jieba 切分里的 ASCII 技术字符集（并上 jieba re_han 已含的 + # & . % -，另含路径/下划线分隔符）
TECH_RUN = re.compile(r"^[A-Za-z0-9_+#./&%\-]+$")

# 分隔符两类：-/./+/#/&/% 在 jieba re_han 内（userdict 可合并）；/ _ 不在（正则级边界，词典无法合并）
SEPS_FIXABLE = "-.+#&%"
SEPS_REGEX = "/_"

NUMERIC_SEQ = re.compile(r"^\d+(?:[.\-/]\d+)+$")  # 纯数字+分隔（日期/版本号），视为合理
ALNUM_MIX = re.compile(r"[A-Za-z]\d|\d[A-Za-z]")  # 字母数字混排（K8s、X25519 型）
CAMEL = re.compile(r"[a-z][A-Z]")  # 驼峰边界

FRONTMATTER_OPEN = re.compile(r"^---\s*$")
FENCE_CLOSED = re.compile(r"(?ms)^(```|~~~)[^\n]*\n.*?^\1[^\n]*$")
FENCE_OPEN_EOF = re.compile(r"(?ms)^(```|~~~)[^\n]*\n.*\Z")
WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
MD_AUTOLINK = re.compile(r"<https?://[^>]+>")


def _wikilink_text(m: re.Match) -> str:
    if m.group(2):
        return m.group(2)
    base = m.group(1).split("#")[0].rsplit("/", 1)[-1]
    return re.sub(r"\.md$", "", base)


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_body(text: str) -> str:
    """剥离 YAML frontmatter 与围栏代码块，折叠链接目标，返回用于分词的正文。"""
    body = text.lstrip("\ufeff\n")
    lines = body.split("\n")
    if lines and FRONTMATTER_OPEN.match(lines[0]):
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                body = "\n".join(lines[i + 1 :])
                break
    body = FENCE_CLOSED.sub(" ", body)
    body = FENCE_OPEN_EOF.sub(" ", body)
    body = MD_AUTOLINK.sub(" ", body)
    body = WIKILINK.sub(_wikilink_text, body)
    body = MD_LINK.sub(lambda m: m.group(1), body)
    return body


def iter_md_files(vault_root: Path):
    """递归收集 (relpath, first_level, size)，跳过排除目录与 vault 根部零散文件。"""
    candidates, root_stray, privacy_skipped = [], [], 0
    for path in vault_root.rglob("*.md"):
        rel = path.relative_to(vault_root)
        parts = rel.parts
        if any(p in EXCLUDED_DIRS for p in parts[:-1]) or parts[0] in EXCLUDED_DIRS:
            continue
        if CREDENTIAL_NAME.search(rel.name):
            privacy_skipped += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0:
            continue
        if len(parts) == 1:
            root_stray.append((str(rel), size))  # vault 根部定义/零散文件，不抽样
            continue
        candidates.append((str(rel), parts[0], size))
    return candidates, root_stray, privacy_skipped


def allocate(counts: dict[str, int], total: int, cap: int) -> dict[str, int]:
    """按 sqrt(文件数) 理想配比 + [1, cap] 截断分配抽样配额，最大余数法补齐到 total。"""
    items = dict(sorted(counts.items()))
    if len(items) > total:  # 层数超过样本量时按权重取前 total 层
        keep = sorted(items, key=lambda k: (-math.sqrt(items[k]), k))[:total]
        items = {k: items[k] for k in keep}
    weights = {k: math.sqrt(v) for k, v in items.items()}
    wsum = sum(weights.values())
    ideal = {k: total * w / wsum for k, w in weights.items()}
    alloc = {k: 1 for k in items}
    remaining = total - len(items)
    while remaining > 0:
        pool = [k for k in items if alloc[k] < cap]
        if not pool:
            break
        best = max(pool, key=lambda k: (ideal[k] - alloc[k], k))
        alloc[best] += 1
        remaining -= 1
    return alloc


def pick_samples(candidates, seed: int, total: int, cap: int):
    """分层 + 中位附近大小过滤 + 固定 seed 随机抽取。返回 (样本列表, 分层计数, 分层配额)。"""
    strata: dict[str, list] = {}
    for rel, first, size in candidates:
        strata.setdefault(first, []).append((rel, size))
    active = {k: sorted(v, key=lambda x: x[0]) for k, v in strata.items() if len(v) >= MIN_STRATUM_FILES}
    counts = {k: len(v) for k, v in active.items()}
    quotas = allocate(counts, total, cap)

    rng = random.Random(seed)
    picked = []
    for stratum in sorted(active):
        pool = active[stratum]
        quota = min(quotas[stratum], len(pool))
        sizes = [s for _, s in pool]
        median = statistics.median(sizes)
        band = pool
        for lo, hi in ((1 / 4, 4), (1 / 10, 10)):  # 先取中位 ±4 倍带，不够再放宽到 ±10 倍
            cand = [x for x in pool if lo * median <= x[1] <= hi * median]
            if len(cand) >= quota:
                band = cand
                break
        for rel, _size in rng.sample(band, quota):
            picked.append((rel, stratum))
    return picked, counts, quotas


def tokenize(body: str):
    """jieba 精确模式（含 HMM）分词，返回非空白 token 的 (word, start, end)。"""
    out = []
    for word, start, end in jieba.tokenize(body, mode="default"):
        if word.strip():
            out.append((word, start, end))
    return out


def classify_span(tokens: list[str]) -> str:
    span = "".join(tokens)
    if NUMERIC_SEQ.match(span) or re.match(r"^\d+\.$", span):
        return "NUMERIC_SEQUENCE"  # 日期/版本号/有序列表标记，合理切分，单列统计
    if not re.search(r"[A-Za-z0-9]", span):
        return "SEPS_ONLY"  # 纯分隔符（注释符等），噪音，不计切错
    if len(tokens) == 2 and tokens[1] == "." and re.match(r"^[A-Za-z0-9]+$", tokens[0]):
        return "TRAILING_PERIOD"  # 句末句点与英文词粘连，检测噪音，不计切错
    if re.match(r"^\.[A-Za-z0-9_]+$", span):
        return "EXT_ONLY"  # 点前缀：文件扩展名/属性访问，词典可修
    if re.search(r"[/_]", span):
        return "SEP_REGEX_BOUNDARY"  # / 或 _ 边界：不在 jieba re_han 内，词典无法修复
    if re.search(r"[-.+#&%]", span):
        return "SEP_FIXABLE"  # - . + 等分隔：userdict 可修复
    if ALNUM_MIX.search(span):
        return "ALNUM_MIX"  # 字母数字混排被拆散
    if CAMEL.search(span):
        return "CASE_SPLIT"  # 驼峰词被拆散
    if span.isalpha() and len(span) >= 5:
        return "ALPHA_FRAG"  # 连写英文词被切成碎片
    return "OTHER"


def detect_suspicious(tokens) -> list[dict]:
    """把连续相邻、字符全属 ASCII 技术集的 token run 拼成 span，多 token 即疑似切碎。"""
    spans = []
    run: list[tuple[str, int, int]] = []
    for tok in tokens + [("", -1, -1)]:  # 哨兵收尾
        contiguous = run and tok[1] == run[-1][2]
        if run and (not contiguous or not TECH_RUN.match(tok[0])):
            span_text = "".join(w for w, _, _ in run)
            if len(run) >= 2 and "://" not in span_text and not span_text.lower().startswith("www."):
                words = [w for w, _, _ in run]
                start, end = run[0][1], run[-1][2]
                spans.append(
                    {
                        "span": span_text,
                        "tokens": words,
                        "pattern": "|".join(w.lower() for w in words),
                        "type": classify_span(words),
                        "start": start,
                        "end": end,
                    }
                )
            run = []
        if tok[0] and TECH_RUN.match(tok[0]):
            run.append(tok)
    return spans


def audit_files(samples: list[tuple[str, str]], vault_root: Path, out_dir: Path):
    """复制样本到 testdata 并做基线切分审计。返回明细列表。"""
    samples_dir = out_dir / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)

    details = []
    for rel, stratum in samples:
        src = vault_root / rel
        raw = src.read_bytes()
        dst = samples_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)  # 原样复制，绝不改动 /vault
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(dst.read_bytes()).hexdigest()

        body = clean_body(raw.decode("utf-8", errors="replace"))
        tokens = tokenize(body)
        spans = detect_suspicious(tokens)
        for sp in spans:
            ctx = body[max(0, sp["start"] - 60) : sp["end"] + 60]
            sp["context"] = re.sub(r"\s+", " ", ctx).strip()
        details.append(
            {
                "relpath": rel,
                "stratum": stratum,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "body_chars": len(body),
                "token_count": len(tokens),
                "suspicious": spans,
            }
        )
    return details


NON_ERROR_TYPES = {"NUMERIC_SEQUENCE", "SEPS_ONLY", "TRAILING_PERIOD"}  # 合理切分/检测噪音，不计切错


def aggregate(details: list[dict], key: str = "suspicious") -> dict:
    by_type = Counter()
    info = Counter()
    patterns: dict[str, dict] = {}
    for d in details:
        for sp in d[key]:
            if sp["type"] in NON_ERROR_TYPES:
                info[sp["type"]] += 1
                continue
            by_type[sp["type"]] += 1
            p = patterns.setdefault(sp["pattern"], {"count": 0, "files": set(), "examples": []})
            p["count"] += 1
            p["files"].add(d["relpath"])
            if len(p["examples"]) < 5:
                p["examples"].append(sp["span"])
    return {
        "total_spans": sum(by_type.values()),
        "files_with_hits": sum(1 for d in details if any(s["type"] not in NON_ERROR_TYPES for s in d[key])),
        "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "info_counts": dict(sorted(info.items(), key=lambda x: -x[1])),
        "patterns": [
            {
                "pattern": pat,
                "count": p["count"],
                "files": len(p["files"]),
                "examples": p["examples"],
            }
            for pat, p in sorted(patterns.items(), key=lambda x: (-x[1]["count"], x[0]))
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Spike 5: jieba 切分质量抽查")
    ap.add_argument("--vault", default=DEFAULT_VAULT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--userdict", default=None, help="加载 userdict 做 before/after 复测")
    ap.add_argument("--total", type=int, default=TOTAL_SAMPLE)
    ap.add_argument("--cap", type=int, default=PER_STRATUM_CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    try:
        jieba.setLogLevel(60)
    except Exception:
        pass

    vault_root = Path(args.vault)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not vault_root.is_dir():
        log(f"错误：vault 目录不存在：{args.vault}")
        return 1

    candidates, root_stray, privacy_skipped = iter_md_files(vault_root)
    picked, stratum_counts, quotas = pick_samples(candidates, args.seed, args.total, args.cap)
    log(f"候选 {len(candidates)} 个 md；分层 {len(stratum_counts)} 个；隐私护栏跳过 {privacy_skipped} 个；根部零散 {len(root_stray)} 个")
    log(f"配额 {quotas}")

    details = audit_files(picked, vault_root, out_dir)

    before = aggregate(details)
    log(f"基线（无 userdict）：切碎 span {before['total_spans']} 个，命中文件 {before['files_with_hits']}/{len(details)}")

    after = None
    userdict_entries = 0
    if args.userdict:
        ud = Path(args.userdict)
        if ud.is_file():
            jieba.load_userdict(str(ud))
            userdict_entries = sum(
                1 for line in ud.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            after_details = []
            for d in details:
                body = (out_dir / "samples" / d["relpath"]).read_text(encoding="utf-8", errors="replace")
                body = clean_body(body)
                tokens = tokenize(body)
                spans = detect_suspicious(tokens)
                for sp in spans:
                    ctx = body[max(0, sp["start"] - 60) : sp["end"] + 60]
                    sp["context"] = re.sub(r"\s+", " ", ctx).strip()
                after_details.append({**d, "suspicious_after": spans, "token_count_after": len(tokens)})
            after = aggregate(after_details, "suspicious_after")
            log(f"userdict（{userdict_entries} 条）后：切碎 span {after['total_spans']} 个")
            details = [
                {**d, "suspicious_after": a["suspicious_after"], "token_count_after": a["token_count_after"]}
                for d, a in zip(details, after_details)
            ]
        else:
            log(f"警告：--userdict 指定的文件不存在，跳过复测：{args.userdict}")

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "vault_root": args.vault,
            "excluded_dirs": sorted(EXCLUDED_DIRS),
            "jieba_version": jieba.__version__,
            "python": sys.version.split()[0],
            "candidate_files": len(candidates),
            "privacy_guard_skipped": privacy_skipped,
            "root_stray_files": len(root_stray),
            "min_stratum_files": MIN_STRATUM_FILES,
            "userdict": args.userdict,
            "userdict_entries": userdict_entries,
        },
        "sampling": {
            "stratum_files": stratum_counts,
            "quotas": quotas,
            "sampled": [
                {"relpath": rel, "stratum": st} for rel, st in picked
            ],
        },
        "aggregate_before": before,
        "aggregate_after": after,
        "files": details,
    }
    out_json = out_dir / "audit_detail.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"明细已写入 {out_json}（含真实内容，仅存 testdata，永不进 git）")

    if after is not None:
        removed = before["total_spans"] - after["total_spans"]
        pct = removed / max(before["total_spans"], 1) * 100
        log(f"改善：{before['total_spans']} -> {after['total_spans']}（消除 {removed}，-{pct:.1f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

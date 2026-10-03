#!/usr/bin/env python3
"""Spike ③ — bge-m3 本地 CPU embedding 吞吐 + 与 bge-reranker-v2-m3 同驻内存基准。

在 knowledge-vault-dev 容器内（linux/arm64，torch CPU，8 线程，Docker VM 内存上限 8GB）测量：
  a) BAAI/bge-m3 下载体积（HF 缓存软链按 (dev,ino) 去重）、加载耗时、加载后 RSS；
  b) 单文件 embedding 吞吐：合成「典型笔记文件」逐个 encode（batched，max_length=1024，
     return_dense=True），每文件毫秒 / 每 chunk 毫秒 / 全库外推；
  c) 批大小敏感性：全部 chunks 一批 64 vs 8×8 分批；
  d) 同驻内存峰值：同一进程先 load BGEM3FlagModel 推理一次，再 load FlagReranker 推理一次，
     记录 ru_maxrss；若进程被 OOM kill，回退为「两进程分别 RSS 相加」估算。

use_fp16 口径（与实验②对照）：FlagEmbedding 1.4.2 中
  - BaseReranker（实验②）：init 不 half()，compute_score_single_gpu 有 device==cpu → use_fp16=False
    分支，CPU 上库默认实际 fp32；
  - M3Embedder（本实验，导出名 BGEM3FlagModel）：use_fp16 **默认 True**，init 通过
    get_model_torch_dtype() 把 torch_dtype=float16 传给 from_pretrained（CPU 上也加载为 fp16），
    但 encode_single_device 开头有 `if device == "cpu": self.model.float()` —— 首次 encode 时
    就地转回 fp32。即库默认路径在 CPU 上「fp16 加载 → fp32 推理」。
  本实验主测量显式 use_fp16=False（干净 fp32 加载，与实验②口径一致），另设 dtype_probe 子进程
  用纯库默认参数实例化，实证上述 fp16→fp32 行为。

运行（容器内形式）：
    docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_bge_m3_local.py --quick  # 自检（3 文件）
    docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_bge_m3_local.py          # 正式（10 文件）

产物：
    /root/llwwds_application/knowledge-vault/state/spike3_bge_m3/results.json  汇总结果（各阶段完成即增量落盘）
    同目录 frag_*.json 为各子进程原始片段（保留作证据，编排进程中断后仍可核对）。
    模型缓存进 HF_HOME（/app_runtime/hf_cache），bge-reranker-v2-m3 复用实验②缓存，不重复下载。

下载来源说明（2026-10-04）：实验执行时容器与宿主均无法连通 huggingface.co 与 hf-mirror.com
（TLS 握手被重置，VPN 未开）；bge-m3 改从 ModelScope 镜像（modelscope.cn，BAAI 官方同步仓库）
下载全部所需文件（config/tokenizer/sentencepiece/colbert_linear.pt/sparse_linear.pt/
pytorch_model.bin 及 sentence-transformers 元数据，跳过 onnx/ 与 imgs/），逐文件 sha256 校验后
按 HF 缓存规范手工构建 blobs/snapshots/refs 布局（revision=modelscope-master），并在
HF_HUB_OFFLINE=1 下运行。实验②的 reranker 缓存为 HF 原源下载，不受影响。

实现要点：
    - torch.set_num_threads(8)（与容器 CPU 数一致，实验②选优档）。
    - 每个测量阶段独立子进程：ru_maxrss 是进程级峰值，子进程隔离后互不污染；
      同驻测量（cores）本身要求两模型同进程，故为单一子进程。
    - 合成数据全部为模板 + 固定词表 + 固定种子（seed=20261004）的通用技术文本，
      不含任何真实 vault 内容、用户名、私人路径。
    - 慢轮自适应：批 64 单轮超过 slow-threshold 后停止加测（只保留已完成的轮次）。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "1"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
SEED = 20261004
THREADS = 8
ENCODE_BATCH_MAIN = 8          # 主吞吐路径批大小（实验②结论：CPU 上小批优）
ENCODE_MAX_LENGTH = 1024
CHUNK_HARD_CAP = 512           # 单 chunk token 上限（含 <s></s>）
CHUNK_TARGET_MIN = 460         # 凑到此 token 数即收口
BODY_TARGET_CHARS = 4100       # 每个合成文件正文字符数（XLM-R 实测 ~0.77 token/字符 → 6-8 个 512-token chunk）
QUERY_MAX_PAIRS = 64           # 同驻测量中 reranker 的 pair 数（与实验②同口径）

STATE_DIR = Path(os.environ.get("SPIKE3_STATE_DIR", "/root/llwwds_application/knowledge-vault/state/spike3_bge_m3"))
RESULTS_PATH = STATE_DIR / "results.json"


# ---------------------------------------------------------------------------
# 合成数据（固定种子；全部为通用技术词，不含真实 vault 内容、用户名、私人路径）
# ---------------------------------------------------------------------------

TOPICS = [
    "向量索引", "倒排索引", "摄入管线", "切块策略", "重排模型", "混合检索",
    "图扩展", "登记层", "版本管理", "增量同步",
]

NOUNS = [
    "向量索引", "HNSW 图", "倒排索引", "BM25 打分", "分词器", "嵌入模型", "交叉编码器",
    "RRF 融合", "候选池", "递归 CTE", "预写日志", "检查点", "缓存行", "工作线程",
    "批大小", "量化算子", "推理延迟", "召回率", "余弦相似度", "归一化层", "停用词表",
    "数据分片", "断点续跑", "幂等写入", "读写锁", "内存峰值", "稠密向量", "稀疏权重",
]

VERBS = ["评估", "重建", "校验", "预热", "分批", "裁剪", "合并", "压测", "冻结", "回放"]

PARA_TEMPLATES = [
    "在{a}的评审里，工程师决定先{v}{b}，再估算{c}的内存峰值，避免查询路径出现尾延迟抖动。这轮改动只动了索引层，登记层的 schema 保持不变。",
    "基准显示，把{a}的批大小从{m}调到{n}之后，{b}的吞吐提升了约{p}%，但{c}的 RSS 也随之上升。最后选择了中间档位，兼顾延迟与余量。",
    "如果{a}与{b}的输出没有对齐，{v}出来的{c}就会偏离预期，这时应先固定随机种子再做对比。经验是先把输入冻结成快照，再逐项核对。",
    "文档约定：{a}的写入必须幂等，{v}{b}时以{m}为键去重，重复导入不产生第二个副本。删除走软删，重建索引时统一回收。",
    "线上巡检脚本每{m}分钟采集一次{a}与{b}的指标，异常时先{v}{c}，再决定是否回滚到上一个构建号。告警阈值按最近七天的分位数动态取值。",
    "重构计划分三步：先冻结{a}的对外接口，然后{v}内部的{b}，最后用{c}的回归集验证结果一致。每一步都要留下可复现的基准数字。",
    "经验教训：不要在{a}未预热完成时开压测，{b}的首轮延迟往往包含懒加载，{c}应当单独统计。冷启动的数字只用于评估加载成本。",
    "为控制{a}的体积，团队把{b}改成按需加载，并在{v}{c}之前检查磁盘余量与配额。大文件只登记元数据，本体留在真源目录。",
    "当{a}与{b}同时竞争工作线程时，调度器优先保障{c}的实时性，其余任务让位并稍后批量处理。离线批处理安排在夜间窗口执行。",
    "代码评审的结论是：{a}应当只缓存派生数据，真值留在源头；{v}{b}时用{c}做一致性抽样。任何缓存都要写明失效条件。",
    "这节笔记记录{a}的调参过程：先把{n}档参数逐一试过，再固定{b}，只让{c}变化。矩阵里最快的一组被标记为推荐档。",
    "排查{a}抖动时发现，{b}的锁竞争才是根因；把{c}改成无共享的分片之后，p95 下降了约{p}%。教训是先测锁再换算法。",
]

CLOSINGS = [
    "小结：先用最小改动验证方向，再决定是否值得引入更重的依赖。",
    "小结：数字要在同一口径下比较，混合口径的对比只会误导决策。",
    "小结：任何优化都要写明触发条件与回退路径，避免把临时方案固化成永久债。",
    "小结：索引层永远可以重建，真源才是唯一可信的数据来源。",
]

FILLER_SENT = "该结论在八核容器环境与桌面宿主上都复测过，数字方向一致，量级差在两倍以内。"


def synth_file(idx: int, seed: int) -> dict:
    """合成一个「典型笔记文件」：标题 + ~BODY_TARGET_CHARS 字正文，模板拼接，确定性。"""
    rng = random.Random(seed)
    topic = TOPICS[idx % len(TOPICS)]
    title = f"关于{topic}的整理笔记（样例{idx + 1:02d}）"
    paras: list[str] = []
    length = 0
    p_idx = 0
    while length < BODY_TARGET_CHARS:
        n_sents = rng.randint(1, 2)
        sents: list[str] = []
        for _ in range(n_sents):
            sents.append(rng.choice(PARA_TEMPLATES).format(
                a=rng.choice(NOUNS), b=rng.choice(NOUNS), c=rng.choice(NOUNS),
                v=rng.choice(VERBS), m=rng.randint(2, 64), n=rng.randint(65, 256),
                p=rng.randint(5, 180),
            ))
        para = "".join(sents)
        paras.append(para)
        length += len(para)
        p_idx += 1
        if p_idx % rng.randint(3, 5) == 0:
            filler = FILLER_SENT if rng.random() < 0.5 else rng.choice(CLOSINGS)
            paras.append(filler)
            length += len(filler)
        if length >= BODY_TARGET_CHARS and rng.random() < 0.8:
            paras.append(rng.choice(CLOSINGS))
            length += len(paras[-1])
    body = "\n\n".join(paras)
    return {"title": title, "body": body, "body_chars": len(body)}


# 同驻测量中 reranker 使用的 query 与 passage（与实验②同风格，固定种子）
RERANK_QUERY = "个人知识库的混合检索管线里，怎样把向量召回、关键词召回和图扩展的候选融合之后交给重排模型排序？"


def synth_passages(n: int, seed: int = SEED) -> list[str]:
    """合成 n 条 150-400 字符的中文技术 passage，模拟待重排 chunk。"""
    rng = random.Random(seed)
    passages: list[str] = []
    for _ in range(n):
        sents: list[str] = []
        length = 0
        target = rng.randint(150, 400)
        while length < target:
            s = rng.choice(PARA_TEMPLATES).format(
                a=rng.choice(NOUNS), b=rng.choice(NOUNS), c=rng.choice(NOUNS),
                v=rng.choice(VERBS), m=rng.randint(2, 64), n=rng.randint(65, 256),
                p=rng.randint(5, 180),
            )
            if length >= 150 and length + len(s) > 400:
                break
            sents.append(s)
            length += len(s)
        p = "".join(sents)[:400]
        if len(p) < 150:
            p = (p + FILLER_SENT * 4)[:max(150, len(p))]
        passages.append(p)
    return passages


# ---------------------------------------------------------------------------
# 切块（按 tokenizer 计数的贪心句子打包）
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")


def chunk_text(tok, text: str, hard_cap: int = CHUNK_HARD_CAP, target_min: int = CHUNK_TARGET_MIN) -> list[str]:
    """把一篇笔记切成 ≤ hard_cap token 的 chunk：句子贪心打包，标题随正文一起切。"""
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    # 单句超长兜底：按逗号/字符硬切
    pieces: list[str] = []
    for s in sents:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if n + 2 <= hard_cap:
            pieces.append(s)
            continue
        for sub in re.split(r"(?<=[，,；;])", s):
            while len(tok(sub, add_special_tokens=False)["input_ids"]) + 2 > hard_cap:
                step = max(8, len(sub) // 2)
                pieces.append(sub[:step])
                sub = sub[step:]
            if sub:
                pieces.append(sub)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0  # 含 <s></s> 两个特殊 token
    for s in pieces:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if cur and cur_len + n + 2 > hard_cap:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n + (2 if cur_len == 0 else 0)
        if cur_len >= target_min:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append("".join(cur))
    return chunks


def chunk_token_lens(tok, chunks: list[str]) -> list[int]:
    return [len(tok(c, add_special_tokens=True)["input_ids"]) for c in chunks]


# ---------------------------------------------------------------------------
# 度量工具
# ---------------------------------------------------------------------------

def ru_maxrss_kb() -> int:
    """linux 上 ru_maxrss 单位为 KB。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def vm_rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def vm_hwm_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def kb_to_gib(kb: int) -> float:
    return round(kb / 1024 / 1024, 3)


def model_cache_dir(model_id: str) -> Path:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    return Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))


def dir_size_bytes(d: Path) -> int:
    """统计目录真实字节数。HF 缓存布局中 snapshot 与 blobs 大多是符号链接
    （大文件实体在共享的 hub/blobs/ 分片子目录），因此对符号链接解析到真实文件，
    并按 (dev, ino) 去重，避免同一实体经 snapshot + blobs 两条路径被重复计数
    （实验②踩坑：不解析软链会把缓存体积算成 2.90KB）。"""
    total = 0
    seen: set[tuple[int, int]] = set()
    for root, _dirs, files in os.walk(d):
        for f in files:
            p = Path(root) / f
            try:
                st = p.stat()  # stat() 跟随符号链接；悬空链接抛 OSError 跳过
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += st.st_size
            except OSError:
                pass
    return total


def cache_ready(d: Path) -> bool:
    if not d.exists():
        return False
    return any((snap / "model.safetensors").exists() or (snap / "pytorch_model.bin").exists()
               for snap in d.glob("snapshots/*")
               if dir_size_bytes(snap) > 0)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def versions() -> dict:
    def v(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "?"
    return {
        "python": platform.python_version(),
        "torch": v("torch"),
        "FlagEmbedding": v("FlagEmbedding"),
        "transformers": v("transformers"),
        "huggingface_hub": v("huggingface_hub"),
        "tokenizers": v("tokenizers"),
    }


def collect_meta(args: argparse.Namespace) -> dict:
    import torch  # noqa: PLC0415

    cgroup_mem = "?"
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if Path(p).exists():
            cgroup_mem = p and Path(p).read_text().strip()
            break
    meminfo = "?"
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            meminfo = line.strip()
            break
    return {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "script_version": SCRIPT_VERSION,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "platform": f"{platform.system()}/{platform.machine()}",
        **versions(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cgroup_mem_limit": cgroup_mem,
        "vm_meminfo_memtotal": meminfo,
        "hf_home": os.environ.get("HF_HOME", "?"),
        "embed_cache_dir": str(model_cache_dir(EMBED_MODEL)),
        "rerank_cache_dir": str(model_cache_dir(RERANK_MODEL)),
        "threads": THREADS,
        "seed": SEED,
        "encode_max_length": ENCODE_MAX_LENGTH,
        "encode_batch_main": ENCODE_BATCH_MAIN,
        "quick": args.quick,
        "n_files": args.files,
        "notes": [
            "容器内 CPU 推理（linux/arm64 NEON，无 CUDA/MPS，torch 未用 macOS Accelerate/AMX），数字为保守上界；宿主 M1 16GB 的 macOS 原生路径预计更快，部署前需宿主 venv 复测校准。",
            "主测量 use_fp16=False（fp32 口径，与实验②一致）；use_fp16 默认行为的实证见 dtype_probe 段。",
            "同驻内存判定受 Docker VM 8GB 上限影响：容器内不 OOM 只说明 ≤8GB，宿主 16GB 判断需对照两模型分别 RSS 与权重和。",
            "bge-m3 缓存来自 ModelScope 镜像（huggingface.co 当时不可达），sha256 逐文件校验后按 HF 缓存规范构建，HF_HUB_OFFLINE=1 运行；权重为 pytorch_model.bin（该仓库无 safetensors）。",
        ],
    }


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# worker：各测量阶段（独立子进程）
# ---------------------------------------------------------------------------

def worker_download(args: argparse.Namespace) -> dict:
    from huggingface_hub import snapshot_download

    cache_dir = model_cache_dir(EMBED_MODEL)
    existed = cache_ready(cache_dir)
    t0 = time.perf_counter()
    local = snapshot_download(
        EMBED_MODEL, ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"]
    )
    download_s = time.perf_counter() - t0
    files = []
    for snap in sorted(Path(local).glob("*")):
        st = snap.stat()
        files.append({"file": snap.name, "bytes": st.st_size})
    total = dir_size_bytes(cache_dir)
    return {
        "tag": "download",
        "cache_existed_before": existed,
        "download_s": round(download_s, 1),
        "local_path": str(local),
        "snapshot_files": files,
        "cache_bytes_dedup": total,
        "cache_human": human(total),
    }


def worker_dtype_probe(args: argparse.Namespace) -> dict:
    """用纯库默认参数实例化 BGEM3FlagModel，实证 use_fp16 默认值在 CPU 上的真实行为。"""
    import torch

    torch.set_num_threads(THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from FlagEmbedding import BGEM3FlagModel

    t0 = time.perf_counter()
    emb = BGEM3FlagModel(EMBED_MODEL)  # 纯库默认：use_fp16 默认 True
    load_s = time.perf_counter() - t0
    dtype_at_load = str(next(emb.model.parameters()).dtype)
    lib_use_fp16 = bool(getattr(emb, "use_fp16", None))
    rss_after_load = vm_rss_kb()

    out = emb.encode(
        ["这是一句用于验证 dtype 行为的短句。", "第二句：批大小为一，长度很短。"],
        batch_size=2, max_length=256, return_dense=True,
    )
    dtype_after_encode = str(next(emb.model.parameters()).dtype)
    vecs = out["dense_vecs"]
    norms = [float((v * v).sum() ** 0.5) for v in vecs]
    return {
        "tag": "dtype_probe",
        "lib_use_fp16_default": lib_use_fp16,
        "model_dtype_at_load": dtype_at_load,
        "model_dtype_after_encode_on_cpu": dtype_after_encode,
        "load_s": round(load_s, 2),
        "vm_rss_after_load_kb": rss_after_load,
        "vm_rss_after_encode_kb": vm_rss_kb(),
        "ru_maxrss_final_kb": ru_maxrss_kb(),
        "dense_dim": int(vecs.shape[1]),
        "vec_l2_norms": [round(n, 6) for n in norms],
        "behavior_note": "M3Embedder init 经 get_model_torch_dtype() 把 use_fp16 默认 True 传给 from_pretrained（CPU 上也加载为 fp16）；encode_single_device 开头 `if device == 'cpu': self.model.float()` 就地转 fp32 —— 与 BaseReranker 的 CPU 分支机制不同，净效果同为 fp32 推理。",
    }


def worker_bench(args: argparse.Namespace) -> dict:
    """主吞吐：模型信息 + 逐文件 encode + 批大小敏感性 + 全库外推。"""
    import numpy as np
    import torch

    torch.set_num_threads(THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from FlagEmbedding import BGEM3FlagModel

    # --- 合成文件与切块（模型加载前先准备好文本） ---
    files = [synth_file(i, SEED + 1000 + i) for i in range(args.files)]

    t0 = time.perf_counter()
    emb = BGEM3FlagModel(EMBED_MODEL, use_fp16=False, devices="cpu")  # fp32 干净加载
    load_s = time.perf_counter() - t0
    model_dtype = str(next(emb.model.parameters()).dtype)
    param_numel = sum(p.numel() for p in emb.model.parameters())
    weights_bytes = sum(p.numel() * p.element_size() for p in emb.model.parameters())
    rss_after_load_kb = vm_rss_kb()
    maxrss_after_load_kb = ru_maxrss_kb()

    tok = emb.tokenizer
    for f in files:
        text = f["title"] + "\n\n" + f["body"]
        f["chunks"] = chunk_text(tok, text)
        f["chunk_lens"] = chunk_token_lens(tok, f["chunks"])
        assert all(l <= CHUNK_HARD_CAP for l in f["chunk_lens"]), "chunk 超 token 上限"
        f["tokens_total"] = sum(f["chunk_lens"])
    chunks_total = sum(len(f["chunks"]) for f in files)
    all_lens = [l for f in files for l in f["chunk_lens"]]

    # --- 预热：文件 0 的前 4 个 chunk ---
    warm_chunks = files[0]["chunks"][:4]
    t = time.perf_counter()
    emb.encode(warm_chunks, batch_size=min(4, len(warm_chunks)), max_length=ENCODE_MAX_LENGTH, return_dense=True)
    warm_s = time.perf_counter() - t
    rss_after_warm_kb = vm_rss_kb()
    dense_dim = None

    # --- b) 逐文件吞吐：2 轮 × 每文件一次 encode ---
    per_file_rounds: list[list[float]] = [[] for _ in files]
    for _r in range(args.rounds_file):
        for i, f in enumerate(files):
            t = time.perf_counter()
            out = emb.encode(f["chunks"], batch_size=ENCODE_BATCH_MAIN,
                             max_length=ENCODE_MAX_LENGTH, return_dense=True)
            dt_ms = (time.perf_counter() - t) * 1000
            vecs = out["dense_vecs"]
            assert vecs.shape[0] == len(f["chunks"]), "encode 返回数量不符"
            dense_dim = int(vecs.shape[1])
            per_file_rounds[i].append(dt_ms)

    per_file = []
    for i, f in enumerate(files):
        p50 = percentile(per_file_rounds[i], 50)
        per_file.append({
            "file_idx": i,
            "title": f["title"],
            "body_chars": f["body_chars"],
            "n_chunks": len(f["chunks"]),
            "tokens_total": f["tokens_total"],
            "rounds_ms": [round(x, 1) for x in per_file_rounds[i]],
            "p50_ms": round(p50, 1),
            "per_chunk_p50_ms": round(p50 / len(f["chunks"]), 1),
        })

    file_p50s = [pf["p50_ms"] for pf in per_file]
    per_file_p50_ms = percentile(file_p50s, 50)
    chunk_p50_ms = percentile([pf["per_chunk_p50_ms"] for pf in per_file], 50)
    mean_chunk_tokens = sum(all_lens) / len(all_lens)
    per_1000_token_ms = chunk_p50_ms / mean_chunk_tokens * 1000

    # --- c) 批大小敏感性：全部 chunks，一批 64 vs 8×8（模型已热，不再单独预热） ---
    all_chunks = [c for f in files for c in f["chunks"]]
    batch_modes: dict[str, dict] = {}
    mode_vecs: dict[str, np.ndarray] = {}
    for bs in (64, ENCODE_BATCH_MAIN):
        rounds: list[float] = []
        for k in range(args.rounds_batch):
            t = time.perf_counter()
            out = emb.encode(all_chunks, batch_size=bs, max_length=ENCODE_MAX_LENGTH, return_dense=True)
            dt = time.perf_counter() - t
            rounds.append(dt)
            if k == 0:
                mode_vecs[str(bs)] = np.asarray(out["dense_vecs"])
            # 慢轮自适应：批 64 单轮超阈值则不再加测
            if dt > args.slow_threshold_s and k >= args.rounds_batch - 2 and bs == 64:
                break
        batch_modes[str(bs)] = {
            "rounds_s": [round(r, 3) for r in rounds],
            "p50_s": round(percentile(rounds, 50), 3),
            "per_chunk_ms": round(percentile(rounds, 50) * 1000 / len(all_chunks), 1),
        }

    v64, v8 = mode_vecs["64"], mode_vecs[str(ENCODE_BATCH_MAIN)]
    cos = (v64 * v8).sum(axis=1)  # dense vecs 已 L2 归一化
    max_cos_dist = float(1.0 - cos.min())

    # --- 全库外推 ---
    ASSUME_FILES = 50000
    ASSUME_CHUNKS_PER_FILE = 20
    total_chunks_assumed = ASSUME_FILES * ASSUME_CHUNKS_PER_FILE
    p50_b8 = batch_modes[str(ENCODE_BATCH_MAIN)]["p50_s"]
    p50_b64 = batch_modes["64"]["p50_s"]
    hours_b8 = total_chunks_assumed * (p50_b8 / len(all_chunks)) / 3600
    hours_b64 = total_chunks_assumed * (p50_b64 / len(all_chunks)) / 3600
    per_file_ms_basis = per_file_p50_ms
    chunks_within_4h = int(4 * 3600 / (p50_b8 / len(all_chunks)))
    files_within_4h = int(chunks_within_4h / ASSUME_CHUNKS_PER_FILE)
    seconds_50_files = 50 * ASSUME_CHUNKS_PER_FILE * (p50_b8 / len(all_chunks))
    seconds_10_files = 10 * ASSUME_CHUNKS_PER_FILE * (p50_b8 / len(all_chunks))

    return {
        "tag": "bench",
        "load_s": round(load_s, 2),
        "model_dtype": model_dtype,
        "lib_use_fp16_flag": bool(getattr(emb, "use_fp16", None)),
        "param_count_million": round(param_numel / 1e6, 1),
        "weights_bytes_in_memory": weights_bytes,
        "weights_human": human(weights_bytes),
        "dense_dim": dense_dim,
        "vm_rss_after_load_kb": rss_after_load_kb,
        "ru_maxrss_after_load_kb": maxrss_after_load_kb,
        "vm_rss_after_warmup_kb": rss_after_warm_kb,
        "warmup": {"n_chunks": len(warm_chunks), "s": round(warm_s, 2)},
        "chunking": {
            "files": len(files),
            "chunks_total": chunks_total,
            "chunk_counts": [len(f["chunks"]) for f in files],
            "body_chars": [f["body_chars"] for f in files],
            "token_len_p50": int(percentile([float(x) for x in all_lens], 50)),
            "token_len_mean": round(mean_chunk_tokens, 1),
            "token_len_max": max(all_lens),
        },
        "per_file": per_file,
        "per_file_summary": {
            "per_file_p50_ms": round(per_file_p50_ms, 1),
            "per_file_min_ms": round(min(file_p50s), 1),
            "per_file_max_ms": round(max(file_p50s), 1),
            "per_chunk_p50_ms": round(chunk_p50_ms, 1),
            "mean_chunk_tokens": round(mean_chunk_tokens, 1),
            "per_1000_token_ms": round(per_1000_token_ms, 2),
        },
        "batch_sensitivity": {
            **{f"batch_{k}": v for k, v in batch_modes.items()},
            "b64_over_b8_p50_ratio": round(p50_b64 / p50_b8, 3) if p50_b8 else None,
            "vec_consistency_max_cosine_dist": round(max_cos_dist, 7),
            "note": "模型经逐文件阶段已充分预热，批对比未再单独预热；批 64 慢轮自适应。",
        },
        "extrapolation": {
            "assumed_files": ASSUME_FILES,
            "chunks_per_file": ASSUME_CHUNKS_PER_FILE,
            "total_chunks": total_chunks_assumed,
            "basis": f"batch_size={ENCODE_BATCH_MAIN} 全集 p50",
            "full_library_hours_b8": round(hours_b8, 1),
            "full_library_hours_b64": round(hours_b64, 1),
            "chunks_within_4h": chunks_within_4h,
            "files_within_4h": files_within_4h,
            "incremental_10_files_s": round(seconds_10_files, 1),
            "incremental_50_files_s": round(seconds_50_files, 1),
            "token_basis_note": f"实测 chunk token p50={int(percentile([float(x) for x in all_lens], 50))}；真实库 chunk 更短时按 per_1000_token_ms 线性缩放。",
        },
        "ru_maxrss_final_kb": ru_maxrss_kb(),
        "vm_hwm_final_kb": vm_hwm_kb(),
    }


def worker_cores(args: argparse.Namespace) -> dict:
    """同驻：同一进程 BGEM3FlagModel 加载 + 推理一次，再 FlagReranker 加载 + 推理一次。"""
    import torch

    torch.set_num_threads(THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from FlagEmbedding import BGEM3FlagModel, FlagReranker

    # 阶段 1：embedding 模型
    t0 = time.perf_counter()
    emb = BGEM3FlagModel(EMBED_MODEL, use_fp16=False, devices="cpu")
    emb_load_s = time.perf_counter() - t0
    emb_dtype = str(next(emb.model.parameters()).dtype)
    emb_rss_after_load_kb = vm_rss_kb()

    files = [synth_file(i, SEED + 1000 + i) for i in range(args.files)]
    chunks: list[str] = []
    for f in files:
        chunks.extend(chunk_text(emb.tokenizer, f["title"] + "\n\n" + f["body"]))
        if len(chunks) >= 8:
            break
    chunks = chunks[:8]

    t = time.perf_counter()
    out = emb.encode(chunks, batch_size=ENCODE_BATCH_MAIN, max_length=ENCODE_MAX_LENGTH, return_dense=True)
    emb_encode_s = time.perf_counter() - t
    assert out["dense_vecs"].shape[0] == len(chunks)
    emb_rss_after_encode_kb = vm_rss_kb()
    maxrss_after_phase1_kb = ru_maxrss_kb()

    # 阶段 2：同进程再加载 reranker（库默认路径，实验②证实 CPU 上实际 fp32）
    t0 = time.perf_counter()
    rr = FlagReranker(RERANK_MODEL, use_fp16=True, devices="cpu")
    rr_load_s = time.perf_counter() - t0
    rr_dtype = str(next(rr.model.parameters()).dtype)
    rr_rss_after_load_kb = vm_rss_kb()

    pairs = [(RERANK_QUERY, p) for p in synth_passages(QUERY_MAX_PAIRS)]
    t = time.perf_counter()
    scores = rr.compute_score(pairs, batch_size=ENCODE_BATCH_MAIN)
    rr_score_s = time.perf_counter() - t
    assert len(scores) == QUERY_MAX_PAIRS and all(math.isfinite(float(s)) for s in scores)

    return {
        "tag": "cores",
        "mode": "same_process",
        "oom": False,
        "embedding_phase": {
            "load_s": round(emb_load_s, 2),
            "model_dtype": emb_dtype,
            "vm_rss_after_load_kb": emb_rss_after_load_kb,
            "encode_n_chunks": len(chunks),
            "encode_s": round(emb_encode_s, 2),
            "vm_rss_after_encode_kb": emb_rss_after_encode_kb,
        },
        "reranker_phase": {
            "load_s": round(rr_load_s, 2),
            "model_dtype": rr_dtype,
            "vm_rss_after_load_kb": rr_rss_after_load_kb,
            "score_n_pairs": QUERY_MAX_PAIRS,
            "batch_size": ENCODE_BATCH_MAIN,
            "score_s": round(rr_score_s, 2),
            "per_pair_ms": round(rr_score_s * 1000 / QUERY_MAX_PAIRS, 1),
        },
        "ru_maxrss_peak_kb": ru_maxrss_kb(),
        "ru_maxrss_peak_gib": kb_to_gib(ru_maxrss_kb()),
        "vm_hwm_final_kb": vm_hwm_kb(),
        "vm_rss_final_kb": vm_rss_kb(),
        "maxrss_after_phase1_kb": maxrss_after_phase1_kb,
        "cgroup_mem_limit_note": "Docker VM 内存上限 8GB（cgroup 未单独限额）；峰值 ≤8GB 只能证明同驻在该上限内可行，宿主 16GB 判断见文档外推。",
    }


def worker_cores_emb_only(args: argparse.Namespace) -> dict:
    """OOM 回退：单进程只载 embedding 模型并推理一次，记录峰值 RSS。"""
    import torch

    torch.set_num_threads(THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from FlagEmbedding import BGEM3FlagModel

    t0 = time.perf_counter()
    emb = BGEM3FlagModel(EMBED_MODEL, use_fp16=False, devices="cpu")
    load_s = time.perf_counter() - t0
    dtype = str(next(emb.model.parameters()).dtype)
    files = [synth_file(i, SEED + 1000 + i) for i in range(2)]
    chunks: list[str] = []
    for f in files:
        chunks.extend(chunk_text(emb.tokenizer, f["title"] + "\n\n" + f["body"]))
        if len(chunks) >= 8:
            break
    chunks = chunks[:8]
    t = time.perf_counter()
    emb.encode(chunks, batch_size=ENCODE_BATCH_MAIN, max_length=ENCODE_MAX_LENGTH, return_dense=True)
    encode_s = time.perf_counter() - t
    return {
        "tag": "cores_emb_only",
        "load_s": round(load_s, 2),
        "model_dtype": dtype,
        "encode_n_chunks": len(chunks),
        "encode_s": round(encode_s, 2),
        "vm_rss_after_encode_kb": vm_rss_kb(),
        "ru_maxrss_peak_kb": ru_maxrss_kb(),
        "ru_maxrss_peak_gib": kb_to_gib(ru_maxrss_kb()),
    }


def worker_cores_rerank_only(args: argparse.Namespace) -> dict:
    """OOM 回退：单进程只载 reranker 并推理一次，记录峰值 RSS（对照实验②）。"""
    import torch

    torch.set_num_threads(THREADS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from FlagEmbedding import FlagReranker

    t0 = time.perf_counter()
    rr = FlagReranker(RERANK_MODEL, use_fp16=True, devices="cpu")
    load_s = time.perf_counter() - t0
    dtype = str(next(rr.model.parameters()).dtype)
    pairs = [(RERANK_QUERY, p) for p in synth_passages(QUERY_MAX_PAIRS)]
    t = time.perf_counter()
    scores = rr.compute_score(pairs, batch_size=ENCODE_BATCH_MAIN)
    score_s = time.perf_counter() - t
    assert len(scores) == QUERY_MAX_PAIRS
    return {
        "tag": "cores_rerank_only",
        "load_s": round(load_s, 2),
        "model_dtype": dtype,
        "score_n_pairs": QUERY_MAX_PAIRS,
        "batch_size": ENCODE_BATCH_MAIN,
        "score_s": round(score_s, 2),
        "per_pair_ms": round(score_s * 1000 / QUERY_MAX_PAIRS, 1),
        "ru_maxrss_peak_kb": ru_maxrss_kb(),
        "ru_maxrss_peak_gib": kb_to_gib(ru_maxrss_kb()),
    }


WORKERS = {
    "download": worker_download,
    "dtype_probe": worker_dtype_probe,
    "bench": worker_bench,
    "cores": worker_cores,
    "cores_emb_only": worker_cores_emb_only,
    "cores_rerank_only": worker_cores_rerank_only,
}


# ---------------------------------------------------------------------------
# 编排：子进程跑各阶段 → 增量合并 results.json
# ---------------------------------------------------------------------------

def run_worker(tag: str, args: argparse.Namespace) -> tuple[int, dict | None, float]:
    """运行单个 worker 子进程，返回 (returncode, frag 或 None, wall_s)。"""
    frag_path = STATE_DIR / f"frag_{tag}.json"
    if frag_path.exists():
        frag_path.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", tag,
           "--files", str(args.files), "--rounds-file", str(args.rounds_file),
           "--rounds-batch", str(args.rounds_batch), "--slow-threshold-s", str(args.slow_threshold_s)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    wall = time.perf_counter() - t0
    frag = None
    if proc.returncode == 0 and frag_path.exists():
        try:
            frag = json.loads(frag_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            frag = None
    return proc.returncode, frag, wall


def kernel_oom_evidence() -> str | None:
    """尽力获取内核 OOM 证据（容器内 dmesg 常无权限，失败则返回 None）。"""
    try:
        r = subprocess.run(["dmesg", "--time-format", "iso"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            r = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            hits = [l for l in r.stdout.splitlines() if "killed process" in l.lower() or "oom" in l.lower()]
            if hits:
                return " | ".join(hits[-3:])[:500]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="自检模式：3 个文件")
    parser.add_argument("--files", type=int, default=None, help="合成文件数（默认 10，quick 3）")
    parser.add_argument("--rounds-file", type=int, default=2, help="逐文件吞吐计测轮数")
    parser.add_argument("--rounds-batch", type=int, default=2, help="批大小对比每档计测轮数上限")
    parser.add_argument("--slow-threshold-s", type=float, default=420.0, help="批 64 单轮超此值则停止加测")
    parser.add_argument("--worker", choices=sorted(WORKERS), default=None, help="内部：以子进程运行指定阶段")
    args = parser.parse_args()
    if args.files is None:
        args.files = 3 if args.quick else 10

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if args.worker:
        frag = WORKERS[args.worker](args)
        write_json(STATE_DIR / f"frag_{args.worker}.json", frag)
        print(f"[worker:{args.worker}] done")
        return

    # ---- 编排 ----
    def log(msg: str) -> None:
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)

    results: dict = {"meta": collect_meta(args)}
    embed_cache = model_cache_dir(EMBED_MODEL)
    rerank_cache = model_cache_dir(RERANK_MODEL)
    results["meta"]["rerank_cache_ready"] = cache_ready(rerank_cache)
    write_json(RESULTS_PATH, results)
    log(f"meta 落盘；embed 缓存就绪={cache_ready(embed_cache)}，rerank 缓存就绪={results['meta']['rerank_cache_ready']}")

    # 阶段 0：模型下载（已缓存则只统计体积）
    if cache_ready(embed_cache):
        total = dir_size_bytes(embed_cache)
        results["download"] = {
            "tag": "download", "cache_existed_before": True, "download_s": 0.0,
            "cache_bytes_dedup": total, "cache_human": human(total),
        }
        log(f"bge-m3 缓存已存在（{human(total)}），跳过下载")
    else:
        log("开始下载 bge-m3（~2.3GB）……")
        rc, frag, wall = run_worker("download", args)
        results["download"] = frag or {"tag": "download", "returncode": rc, "wall_s": round(wall, 1)}
        log(f"下载完成 rc={rc} wall={wall:.0f}s")
    write_json(RESULTS_PATH, results)

    # 阶段 1：dtype probe（use_fp16 默认行为实证）
    log("dtype probe：库默认参数实例化（fp16 加载 → CPU encode 转 fp32）……")
    rc, frag, wall = run_worker("dtype_probe", args)
    results["dtype_probe"] = frag or {"tag": "dtype_probe", "returncode": rc, "wall_s": round(wall, 1)}
    log(f"dtype probe rc={rc} wall={wall:.0f}s")
    write_json(RESULTS_PATH, results)

    # 阶段 2：主吞吐
    log(f"bench：{args.files} 个合成文件逐个 encode + 批大小对比……")
    rc, frag, wall = run_worker("bench", args)
    results["bench"] = frag or {"tag": "bench", "returncode": rc, "wall_s": round(wall, 1)}
    log(f"bench rc={rc} wall={wall:.0f}s")
    write_json(RESULTS_PATH, results)

    # 阶段 3：同驻内存（同一进程顺序加载两模型并各推理一次）
    log("cores：同进程 BGEM3FlagModel → FlagReranker 顺序加载与推理……")
    rc, frag, wall = run_worker("cores", args)
    if rc == 0 and frag:
        results["coresidency"] = frag
        log(f"cores rc=0 wall={wall:.0f}s 峰值={frag['ru_maxrss_peak_gib']} GiB")
    else:
        oom = rc in (-9, 137)
        log(f"cores rc={rc}（{'疑似 OOM kill' if oom else '失败'}），回退为两进程分别 RSS 相加估算")
        fallback: dict = {
            "mode": "two_process_estimate",
            "oom": oom,
            "cores_exit_code": rc,
            "kernel_oom_evidence": kernel_oom_evidence() if oom else None,
            "note": "同驻 worker 在容器 8GB VM 上限内被杀/失败；改为两进程分别测峰值 RSS 后相加估算同驻需求。该观测受 8GB 上限约束，宿主 16GB 下同驻可能仍可行。",
        }
        for tag in ("cores_emb_only", "cores_rerank_only"):
            rc2, frag2, wall2 = run_worker(tag, args)
            fallback[tag] = frag2 or {"tag": tag, "returncode": rc2, "wall_s": round(wall2, 1)}
            log(f"{tag} rc={rc2} wall={wall2:.0f}s")
        e = fallback.get("cores_emb_only") or {}
        r = fallback.get("cores_rerank_only") or {}
        if isinstance(e, dict) and isinstance(r, dict) and e.get("ru_maxrss_peak_kb") and r.get("ru_maxrss_peak_kb"):
            s = e["ru_maxrss_peak_kb"] + r["ru_maxrss_peak_kb"]
            fallback["estimated_coresident_gib"] = kb_to_gib(s)
        results["coresidency"] = fallback

    write_json(RESULTS_PATH, results)
    log(f"results.json 写入完成：{RESULTS_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Spike ② — bge-reranker-v2-m3 在 CPU 上的延迟与内存基准（knowledge-vault-dev 容器内运行）。

测量查询路径末级重排的成本：64 对 query-passage，CPU fp16/fp32，64 一批 vs 8×8 分批，
torch threads=4 vs 8 对比。全部计算在容器内（linux/arm64，无 CUDA/MPS）完成。

运行（容器内形式）：
    docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_reranker_cpu.py --quick  # 自检（轮次少）
    docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/bench_reranker_cpu.py          # 正式

产物：
    /root/llwwds_application/knowledge-vault/state/spike2_reranker/results.json  汇总结果
    同目录 frag_*.json 为各配置子进程的原始片段（保留作证据，用于断点续跑）。

实现要点：
    - use_fp16=True 传入 FlagReranker（库默认路径，precision="auto"）。注意 FlagEmbedding 1.4.2 的
      BaseReranker.compute_score_single_gpu 内有 `if device == "cpu": self.use_fp16 = False` 分支
      （init 本身不调用 half()），即库"打算"在 CPU 上禁用 fp16；但模型实际 dtype 以
      transformers from_pretrained 的结果为准（worker 记录 model_dtype 字段）。
      precision="half" / "fp32" 为强制 model.half() / model.float() 的对照分支，
      编排器按 main 阶段实测 dtype 自动选择相反方向作对照。
    - 每个测量配置在独立子进程中运行：torch.set_num_threads 必须在重计算前生效，且
      resource.ru_maxrss 是进程级峰值，子进程隔离后各配置互不污染。
    - 64 对 query-passage 由模板 + 固定词表 + 固定种子合成，不含任何真实 vault 内容。
    - 慢轮自适应：若单轮耗时超过 --slow-threshold-s（默认 20s），后续轮次降为
      --rounds-slow（默认 5），并在结果中记录 rounds_slow_applied=true（p95 口径随之注明样本数）。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "1"
MODEL = "BAAI/bge-reranker-v2-m3"
N_PAIRS = 64
SEED = 20261004

STATE_DIR = Path(os.environ.get("SPIKE2_STATE_DIR", "/root/llwwds_application/knowledge-vault/state/spike2_reranker"))
RESULTS_PATH = STATE_DIR / "results.json"

# ---------------------------------------------------------------------------
# 合成数据（固定种子；全部为通用技术词，不含真实 vault 内容、用户名、私人路径）
# ---------------------------------------------------------------------------

QUERY = "个人知识库的混合检索管线里，怎样把向量召回、关键词召回和图扩展的候选融合之后交给重排模型排序？"

NOUNS = [
    "向量索引", "HNSW 图", "倒排索引", "BM25 打分", "分词器", "嵌入模型", "交叉编码器",
    "RRF 融合", "候选池", "递归 CTE", "预写日志", "检查点", "缓存行", "工作线程",
    "批大小", "量化算子", "推理延迟", "召回率", "余弦相似度", "归一化层", "停用词表",
    "数据分片", "断点续跑", "幂等写入", "读写锁", "内存峰值", "构建号", "语义化版本",
]

VERBS = ["评估", "重建", "校验", "预热", "分批", "裁剪", "合并", "压测", "冻结", "回放"]

TEMPLATES = [
    "在{a}的评审里，工程师决定先{v}{b}，再估算{c}的内存峰值，避免查询路径出现尾延迟抖动。",
    "基准显示，把{a}的批大小从{m}调到{n}之后，{b}的吞吐提升了约{p}%，但{c}的RSS也随之上升。",
    "如果{a}与{b}的输出没有对齐，{v}出来的{c}就会偏离预期，这时应先固定随机种子再做对比。",
    "文档约定：{a}的写入必须幂等，{v}{b}时以{m}为键去重，重复导入不产生第二个副本。",
    "线上巡检脚本每{m}分钟采集一次{a}与{b}的指标，异常时先{v}{c}，再决定是否回滚到上一个构建号。",
    "重构计划分三步：先冻结{a}的对外接口，然后{v}内部的{b}，最后用{c}的回归集验证结果一致。",
    "经验教训：不要在{a}未预热完成时开压测，{b}的首轮延迟往往包含懒加载，{c}应当单独统计。",
    "为控制{a}的体积，团队把{b}改成按需加载，并在{v}{c}之前检查磁盘余量与配额。",
    "当{a}与{b}同时竞争工作线程时，调度器优先保障{c}的实时性，其余任务让位并稍后批量处理。",
    "代码评审的结论是：{a}应当只缓存派生数据，真值留在源头；{v}{b}时用{c}做一致性抽样。",
]

GENERIC_FILLER = "该结论在八核容器环境与桌面宿主上都复测过，数字方向一致，量级差在两倍以内。"


def synth_passages(n: int, seed: int = SEED) -> list[str]:
    """合成 n 条 150-400 字符的中文技术 passage，模拟笔记 chunk。"""
    rng = random.Random(seed)
    passages: list[str] = []
    for _ in range(n):
        sents: list[str] = []
        length = 0
        target = rng.randint(150, 400)
        while length < target:
            s = rng.choice(TEMPLATES).format(
                a=rng.choice(NOUNS), b=rng.choice(NOUNS), c=rng.choice(NOUNS),
                v=rng.choice(VERBS), m=rng.randint(2, 64), n=rng.randint(65, 256),
                p=rng.randint(5, 180),
            )
            if length >= 150 and length + len(s) > 400:
                break
            sents.append(s)
            length += len(s)
        if rng.random() < 0.5 and length + len(GENERIC_FILLER) <= 400:
            sents.append(GENERIC_FILLER)
        p = "".join(sents)
        assert 150 <= len(p) <= 400, f"passage 长度越界: {len(p)}"
        passages.append(p)
    return passages


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
    """线性插值百分位。"""
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def model_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    return Path(hf_home) / "hub" / "models--BAAI--bge-reranker-v2-m3"


def dir_size_bytes(d: Path) -> int:
    """统计目录真实字节数。HF 新版缓存布局中 snapshot 与 blobs 大多是符号链接
    （大文件实体在共享的 hub/blobs/ 分片子目录），因此对符号链接解析到真实文件，
    并按 (dev, ino) 去重，避免同一实体经 snapshot + blobs 两条路径被重复计数。"""
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


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


# ---------------------------------------------------------------------------
# worker：单配置测量（独立子进程）
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from FlagEmbedding import FlagReranker

    t0 = time.perf_counter()
    reranker = FlagReranker(MODEL, use_fp16=True, devices="cpu")  # 纯 CPU；use_fp16 按 spec 传 True，实际 dtype 以 model_dtype 为准
    lib_fp16_active = bool(getattr(reranker, "use_fp16", False))
    if args.precision == "half":
        reranker.model.half()  # 强制 fp16 对照
    elif args.precision == "fp32":
        reranker.model.float()  # 强制 fp32 对照
    model_dtype = str(next(reranker.model.parameters()).dtype)
    param_numel = sum(p.numel() for p in reranker.model.parameters())
    weights_bytes = sum(p.numel() * p.element_size() for p in reranker.model.parameters())
    load_s = time.perf_counter() - t0

    rss_after_load_kb = vm_rss_kb()
    maxrss_after_load_kb = ru_maxrss_kb()

    pairs = [(QUERY, p) for p in synth_passages(N_PAIRS)]
    tok = reranker.tokenizer
    tok_lens = [
        len(tok(q, p, add_special_tokens=True, truncation=True, max_length=512)["input_ids"])
        for q, p in pairs
    ]

    def one_round(batch_size: int):
        t = time.perf_counter()
        scores = reranker.compute_score(pairs, batch_size=batch_size)
        dt = time.perf_counter() - t
        assert len(scores) == N_PAIRS and all(math.isfinite(float(s)) for s in scores)
        return dt, [float(s) for s in scores]

    timed_target = args.rounds
    rounds: list[float] = []
    first_b64_scores: list[float] | None = None
    first_b8_scores: list[float] | None = None
    slow_applied = False

    for _ in range(args.warmup):
        one_round(args.batch_size)

    i = 0
    while i < timed_target:
        dt, scores = one_round(args.batch_size)
        if args.batch_size == 64 and first_b64_scores is None:
            first_b64_scores = scores
        rounds.append(dt)
        i += 1
        # 慢轮自适应：第 2 个计测轮后判定，若单轮已远超预算则减少后续轮次
        if i == 2 and len(rounds) >= 2 and (sum(rounds) / len(rounds)) > args.slow_threshold_s \
                and timed_target > args.rounds_slow:
            timed_target = args.rounds_slow
            slow_applied = True

    b8_rounds: list[float] = []
    if args.also_batch8:
        for _ in range(max(1, args.warmup - 1)):
            one_round(8)
        j = 0
        b8_target = timed_target
        while j < b8_target:
            dt, scores = one_round(8)
            if first_b8_scores is None:
                first_b8_scores = scores
            b8_rounds.append(dt)
            j += 1
            if j == 2 and (sum(b8_rounds) / len(b8_rounds)) > args.slow_threshold_s and b8_target > args.rounds_slow:
                b8_target = args.rounds_slow

    diff = None
    if first_b64_scores and first_b8_scores:
        diff = max(abs(a - b) for a, b in zip(first_b64_scores, first_b8_scores))

    result = {
        "tag": args.tag,
        "threads": args.threads,
        "effective_num_threads": torch.get_num_threads(),
        "precision": args.precision,
        "lib_use_fp16_active": lib_fp16_active,
        "model_dtype": model_dtype,
        "param_numel": param_numel,
        "param_count_million": round(param_numel / 1e6, 1),
        "weights_bytes_in_memory": weights_bytes,
        "weights_human": human(weights_bytes),
        "batch_mode": "64-single" if args.batch_size == 64 else "8x8",
        "warmup": args.warmup,
        "rounds_requested": args.rounds,
        "rounds_actual": len(rounds),
        "rounds_slow_applied": slow_applied,
        "rounds_s": [round(r, 4) for r in rounds],
        "p50_s": round(percentile(rounds, 50), 4),
        "p95_s": round(percentile(rounds, 95), 4),
        "max_s": round(max(rounds), 4),
        "per_pair_p50_ms": round(percentile(rounds, 50) * 1000 / N_PAIRS, 2),
        "per_pair_p95_ms": round(percentile(rounds, 95) * 1000 / N_PAIRS, 2),
        "batch8_rounds_s": [round(r, 4) for r in b8_rounds] or None,
        "batch8_p50_s": round(percentile(b8_rounds, 50), 4) if b8_rounds else None,
        "batch8_per_pair_p50_ms": round(percentile(b8_rounds, 50) * 1000 / N_PAIRS, 2) if b8_rounds else None,
        "score_batch_consistency_max_absdiff": round(diff, 6) if diff is not None else None,
        "score_min_max": [min(first_b64_scores or first_b8_scores or [0.0]),
                          max(first_b64_scores or first_b8_scores or [0.0])],
        "load_s": round(load_s, 2),
        "vm_rss_after_load_kb": rss_after_load_kb,
        "ru_maxrss_after_load_kb": maxrss_after_load_kb,
        "ru_maxrss_final_kb": ru_maxrss_kb(),
        "vm_hwm_final_kb": vm_hwm_kb(),
        "token_len_max": max(tok_lens),
        "token_len_p50": int(percentile([float(x) for x in tok_lens], 50)),
        "token_len_mean": round(sum(tok_lens) / len(tok_lens), 1),
    }
    return result


# ---------------------------------------------------------------------------
# 编排：子进程跑各配置 → 合并 results.json
# ---------------------------------------------------------------------------

def fingerprint(worker_args: dict) -> dict:
    return {"script_version": SCRIPT_VERSION, "model": MODEL, "worker": worker_args}


def spawn_worker(worker_args: dict) -> dict:
    tag = worker_args["tag"]
    frag_path = STATE_DIR / f"frag_{tag}.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--tag", tag, "--threads", str(worker_args["threads"]),
        "--precision", worker_args["precision"], "--batch-size", str(worker_args["batch_size"]),
        "--warmup", str(worker_args["warmup"]), "--rounds", str(worker_args["rounds"]),
        "--rounds-slow", str(worker_args["rounds_slow"]),
    ]
    if worker_args.get("also_batch8"):
        cmd.append("--also-batch8")
    print(f"[orchestrator] start {tag}: {' '.join(cmd[2:])}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], file=sys.stderr)
        print(proc.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"worker {tag} failed (exit {proc.returncode})")
    out = proc.stdout.strip().splitlines()[-1]
    frag = json.loads(out)
    frag["fingerprint"] = fingerprint(worker_args)
    frag_path.write_text(json.dumps(frag, ensure_ascii=False, indent=2))
    print(f"[orchestrator] {tag} done: p50={frag['p50_s']}s p95={frag['p95_s']}s", flush=True)
    return frag


def reuse_frag(worker_args: dict) -> dict | None:
    frag_path = STATE_DIR / f"frag_{worker_args['tag']}.json"
    if frag_path.exists():
        try:
            frag = json.loads(frag_path.read_text())
            if frag.get("fingerprint") == fingerprint(worker_args):
                print(f"[orchestrator] reuse existing frag for {worker_args['tag']}", flush=True)
                return frag
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def cgroup_mem_limit_bytes() -> str:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v and v != "max":
                return f"{int(v)} ({human(int(v))})"
            if v == "max":
                return "max (未单独限额，随 Docker VM)"
        except (OSError, ValueError):
            continue
    return "unknown"


def collect_meta() -> dict:
    cache_dir = model_cache_dir()
    downloaded = cache_dir.exists() and (
        any(cache_dir.rglob("*.safetensors")) or any(cache_dir.rglob("*.bin"))
    )
    return {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": MODEL,
        "platform": f"{platform.system()}/{platform.machine()}",
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "FlagEmbedding": importlib.metadata.version("FlagEmbedding"),
        "transformers": importlib.metadata.version("transformers"),
        "cpu_count": os.cpu_count(),
        "cuda_available": False,  # 容器内实测（见报告），此处占位由 worker 校正
        "cgroup_mem_limit": cgroup_mem_limit_bytes(),
        "model_cache_dir": str(cache_dir),
        "model_cache_downloaded": bool(downloaded),
        "model_cache_bytes": dir_size_bytes(cache_dir) if cache_dir.exists() else 0,
        "model_cache_human": human(dir_size_bytes(cache_dir)) if cache_dir.exists() else "0 B",
    }


def main() -> None:
    global STATE_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="减少轮次：协议 1 预热 + 3 计测；仍是真实模型")
    ap.add_argument("--force", action="store_true", help="忽略已有 frag_*.json 全部重跑")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--threads-probe", default="4,8", help="threads 探测档位，逗号分隔")
    # worker 专用参数
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--tag", default="worker")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--precision", choices=["auto", "half", "fp32"], default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--rounds-slow", type=int, default=5)
    ap.add_argument("--slow-threshold-s", type=float, default=20.0)
    ap.add_argument("--also-batch8", action="store_true", help="同一 worker 内追加 8×8 分批对比")
    args = ap.parse_args()

    STATE_DIR = Path(args.state_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if args.worker:
        print(json.dumps(run_worker(args), ensure_ascii=False))
        return

    # ---- 编排模式 ----
    torch_check = subprocess.run(
        [sys.executable, "-c", "import torch;print(int(torch.cuda.is_available()), int(torch.backends.mps.is_available()))"],
        capture_output=True, text=True,
    )
    cuda_mps = torch_check.stdout.split()
    cuda_ok, mps_ok = (cuda_mps + ["0", "0"])[:2]

    proto_warmup, proto_rounds = (1, 3) if args.quick else (2, 10)
    probe_warmup, probe_rounds = 1, 2

    meta = collect_meta()
    meta["cuda_available"] = cuda_ok == "1"
    meta["mps_available"] = mps_ok == "1"
    meta["quick"] = args.quick
    meta["notes"] = [
        "precision=auto 为库默认路径（use_fp16=True 传入 FlagReranker）；FlagEmbedding 1.4.2 的 "
        "compute_score_single_gpu 内有 device==cpu 时 use_fp16=False 的分支，但模型实际 dtype 以各 worker "
        "的 model_dtype 字段为准（transformers 5.18 from_pretrained 决定）。contrast 阶段按实测 dtype 取相反精度对照。",
        "normalize 使用库默认（False）；compute_score 其余默认（max_length=512）。",
        "查询固定 1 条中文技术 query，passage 为 64 条模板合成中文技术文本（150-400 字符，seed=20261004），无真实 vault 内容。",
    ]
    print(f"[orchestrator] meta: cache={meta['model_cache_human']} cuda={meta['cuda_available']} mps={meta['mps_available']}", flush=True)

    # 阶段 1：threads 探测（各 warmup 1 + 计测 2，64 一批，库默认精度）
    probes: dict[int, dict] = {}
    for th in [int(x) for x in args.threads_probe.split(",")]:
        wa = {"tag": f"probe_t{th}", "threads": th, "precision": "auto", "batch_size": 64,
              "warmup": probe_warmup, "rounds": probe_rounds, "rounds_slow": probe_rounds}
        frag = None if args.force else reuse_frag(wa)
        probes[th] = frag or spawn_worker(wa)
    winner_threads = min(probes, key=lambda th: probes[th]["p50_s"])
    print(f"[orchestrator] threads winner = {winner_threads} "
          f"(p50: " + ", ".join(f"t{th}={probes[th]['p50_s']}s" for th in sorted(probes)) + ")", flush=True)

    # 阶段 2：胜出 threads 上跑全量协议（64 一批 + 8×8 分批）
    wa_main = {"tag": f"main_t{winner_threads}", "threads": winner_threads, "precision": "auto",
               "batch_size": 64, "warmup": proto_warmup, "rounds": proto_rounds,
               "rounds_slow": min(proto_rounds, 5), "also_batch8": True}
    frag_main = None if args.force else reuse_frag(wa_main)
    if frag_main is None:
        frag_main = spawn_worker(wa_main)

    # 阶段 3：精度对照（按 main 阶段实测 dtype 选相反方向：fp16 实测则测 fp32，fp32 实测则测 fp16）
    main_dtype = frag_main.get("model_dtype", "torch.float32")
    contrast_precision = "fp32" if "float16" in main_dtype else "half"
    contrast_warmup, contrast_rounds = (1, 2) if args.quick else (1, 3)
    wa_contrast = {"tag": f"{contrast_precision}_t{winner_threads}", "threads": winner_threads,
                   "precision": contrast_precision, "batch_size": 64,
                   "warmup": contrast_warmup, "rounds": contrast_rounds, "rounds_slow": max(2, contrast_rounds - 1)}
    frag_contrast = None if args.force else reuse_frag(wa_contrast)
    if frag_contrast is None:
        try:
            frag_contrast = spawn_worker(wa_contrast)
        except RuntimeError as e:
            frag_contrast = {"tag": wa_contrast["tag"], "error": str(e)[:500]}
            (STATE_DIR / f"frag_{wa_contrast['tag']}.json").write_text(json.dumps(frag_contrast, ensure_ascii=False))

    # 补一次元数据（模型此时必然已下载）
    meta = collect_meta()
    meta["cuda_available"] = cuda_ok == "1"
    meta["mps_available"] = mps_ok == "1"

    results = {
        "meta": meta,
        "probe": {f"t{th}": probes[th] for th in sorted(probes)},
        "main": frag_main,
        "contrast": frag_contrast,
        "winner_threads": winner_threads,
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[orchestrator] results -> {RESULTS_PATH}", flush=True)
    print(json.dumps({k: results[k] for k in ("winner_threads",)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

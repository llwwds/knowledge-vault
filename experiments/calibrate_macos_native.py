#!/usr/bin/env python3
"""macOS 原生最小校准点 — 校准 spike ②③ 容器数字的系统性偏差。

背景：spike ②③跑在 linux/arm64 容器（无 Accelerate/AMX，VM 内存 8GB 上限），
判定对象是 M1 本机（16GB）。本脚本在宿主机 macOS venv（规范第 1 条允许的受控环境）
跑最小单点对照，产出「macOS/容器」加速比，用于校准容器数字：
  a) reranker 64 对：库默认路径（fp32）+ 强制 half()（测 M1 AMX fp16 可用性）
  b) bge-m3 单文件 encode（~7-8 chunks）
  c) 同进程双模型同驻 RSS（16GB 硬件口径，无 VM 上限失真）

模型缓存复用 HF_HOME（与容器共享，已含 bge-m3 与 bge-reranker-v2-m3），
全程 HF_HUB_OFFLINE=1。结果 JSON 追加写入 state/spike_calibration/results.json。

用法：
    HF_HUB_OFFLINE=1 HF_HOME=~/llwwds_application/knowledge-vault/hf_cache \
        ~/llwwds_application/knowledge-vault/venv-mac/bin/python \
        experiments/calibrate_macos_native.py [--skip-resident]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import torch

MODEL_DIR = os.path.expanduser(
    "~/llwwds_application/knowledge-vault/state/spike_calibration"
)

# ----------------------------- 合成文本（与 spike ②③ 同风格，无真实内容） -----------------------------

TOPICS = ["向量数据库", "知识图谱", "文档解析", "权限管理", "异步队列", "全文检索",
          "嵌入模型", "重排序", "数据管道", "版本控制", "备份策略", "缓存设计"]
FILLER = ["该模块采用分层架构设计", "在实现时需要注意边界条件", "通过配置注入解耦依赖",
          "写入路径与查询路径分离", "索引层可随时从真源重建", "单机串行模型避免锁竞争",
          "按需加载降低常驻内存", "批处理摊薄固定开销"]


def synth_passages(n: int, seed: int = 20261004) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sents = [rng.choice(FILLER) + "，涉及" + rng.choice(TOPICS) + "。"
                 for _ in range(8)]
        out.append(f"（合成样例 {i}）" + "".join(sents))
    return out


def rss_gib() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 3)  # macOS ru_maxrss 单位为字节


def save(results: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {path}")


# ----------------------------- 测量段 -----------------------------

def bench_reranker() -> dict:
    from FlagEmbedding import FlagReranker

    pairs = [
        ["如何为个人知识库选择向量数据库与全文检索的组合方案？", p]
        for p in synth_passages(64)
    ]
    out: dict = {}

    t0 = time.perf_counter()
    rr = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    out["load_seconds"] = time.perf_counter() - t0
    out["default_dtype"] = str(next(rr.model.parameters()).dtype)
    out["rss_after_load_gib"] = round(rss_gib(), 2)

    # 库默认路径（CPU 分支实为 fp32）
    rr.compute_score(pairs[:8], batch_size=8)  # 预热
    lat = []
    for _ in range(3):
        t0 = time.perf_counter()
        rr.compute_score(pairs, batch_size=8)
        lat.append(time.perf_counter() - t0)
    out["fp32_default_64pairs_b8"] = {
        "rounds_s": [round(x, 2) for x in lat],
        "p50_s": round(float(np.median(lat)), 2),
    }

    # 强制 half()：测 M1 CPU fp16（AMX）可用性与加速比
    try:
        rr.model.half()
        t0 = time.perf_counter()
        rr.compute_score(pairs, batch_size=8)
        half_s = time.perf_counter() - t0
        out["fp16_forced_64pairs_b8_s"] = round(half_s, 2)
        rr.model.float()  # 复原，供同驻段继续用
    except Exception as e:  # noqa: BLE001
        out["fp16_forced_error"] = repr(e)[:200]
    out["peak_rss_gib"] = round(rss_gib(), 2)
    return out


def bench_bge_m3() -> dict:
    from FlagEmbedding import BGEM3FlagModel

    texts = synth_passages(8, seed=20261005)  # 模拟一个 ~8 chunk 的笔记文件
    out: dict = {}
    t0 = time.perf_counter()
    m3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    out["load_seconds"] = time.perf_counter() - t0
    out["default_dtype"] = str(next(m3.model.parameters()).dtype)

    m3.encode(texts[:2], max_length=1024)  # 预热
    t0 = time.perf_counter()
    res = m3.encode(texts, max_length=1024, batch_size=8)
    dt = time.perf_counter() - t0
    out["one_file_8chunks"] = {
        "seconds": round(dt, 2),
        "ms_per_chunk": round(dt * 1000 / len(texts), 1),
        "dense_dim": int(np.asarray(res["dense_vecs"]).shape[1]),
    }
    out["peak_rss_gib"] = round(rss_gib(), 2)
    return out


# ----------------------------- 主流程 -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-resident", action="store_true",
                    help="跳过双模型同驻段（调试用）")
    args = ap.parse_args()

    torch.set_num_threads(8)
    print(f"python {platform.python_version()} | torch {torch.__version__} | "
          f"macOS {platform.mac_ver()[0]} | arm64 | threads={torch.get_num_threads()}")

    results = {
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "host_mem_gb": 16,
            "offline": os.environ.get("HF_HUB_OFFLINE", ""),
        },
        "note": "macOS 原生校准点：单点对照，用于校准 spike②③ 容器数字，非全量复跑",
    }

    results["reranker"] = bench_reranker()
    print("reranker:", json.dumps(results["reranker"], ensure_ascii=False))

    if not args.skip_resident:
        results["bge_m3"] = bench_bge_m3()
        print("bge_m3:", json.dumps(results["bge_m3"], ensure_ascii=False))
        results["resident_peak_rss_gib"] = round(rss_gib(), 2)

    save(results)

    print("===== 校准摘要 =====")
    rr = results["reranker"]
    fp32 = rr["fp32_default_64pairs_b8"]["p50_s"]
    print(f"reranker 64对 b8: fp32 p50={fp32}s"
          + (f" | fp16 强制={rr.get('fp16_forced_64pairs_b8_s')}s"
             if "fp16_forced_64pairs_b8_s" in rr else ""))
    if "bge_m3" in results:
        m3o = results["bge_m3"]["one_file_8chunks"]
        print(f"bge-m3 单文件(8 chunks): {m3o['seconds']}s "
              f"({m3o['ms_per_chunk']} ms/chunk), dense_dim={m3o['dense_dim']}")
    if "resident_peak_rss_gib" in results:
        print(f"同驻进程峰值 RSS: {results['resident_peak_rss_gib']} GiB / 16GB 宿主")
    return 0


if __name__ == "__main__":
    sys.exit(main())

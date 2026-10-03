#!/usr/bin/env python3
"""Spike 1 — zvec 0.7.0 自证基准：50 万 chunk 规模的插入 / 查询 / 删插 / 重建。

模拟 knowledge-vault 索引层选型验证：
- 数据全部为合成数据（dim=1024 随机单位向量，模拟 bge-m3 输出）；
- payload 设计定稿为「正文不入向量库」：只存 file_id / chunk_seq / kind；
- 主键为 zvec 要求的字符串形式：f"f{file_id:05d}-c{chunk_seq:03d}"。

阶段：
  1. 分批插入（每批 10,000 条）并计时，观察峰值内存；
  2. flush + optimize（compaction）计时；
  3. 查询延迟：100 条随机查询 top_k=10（默认 ef=300 与 ef=100 各测一轮），
     另测 20 条带 file_id 过滤的查询；20 条「以库内某 chunk 自身向量为查询」的自召回 sanity；
  4. 按 file_id 删插：随机 100 个 file（=5,000 条）delete(ids) 计时后 upsert 重插计时；
     另随机 100 个 file 用 delete_by_filter 逐 file 删除计时（两种删除语义对照）；
  5. 全量重建：drop 目录 + 新建 + 全量重插 + flush + optimize 端到端计时；
  6. 磁盘占用：collection 目录 du。

用法：
    python experiments/bench_zvec_500k.py           # 500k 正式基准（预计 10-40 分钟）
    python experiments/bench_zvec_500k.py --quick   # 1 万条快速自检版（约 1 分钟）

幂等：启动时清理本脚本自己的 collection 目录（state/spike1_zvec/bench_main）。
结果 JSON 写入 state/spike1_zvec/bench_results_{quick|full}.json。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import zvec
from zvec import (
    CollectionSchema,
    DataType,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    MetricType,
    Query,
    VectorSchema,
)

# ----------------------------- 常量与参数 -----------------------------

SEED = 20261003
DIM = 1024                    # bge-m3 输出维度
VEC_FIELD = "embedding"
CHUNKS_PER_FILE = 50          # 每文件 50 个 chunk
KIND = "chunk"

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

INSERT_CALL_SIZE = 1000      # zvec 单次 Insert/Upsert 上限 1024 条，取 1000 稳妥

STATE_DIR = os.path.expanduser(
    "~/llwwds_application/knowledge-vault/state/spike1_zvec"
)
MAIN_DIR = os.path.join(STATE_DIR, "bench_main")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb() -> float:
    """进程峰值 RSS（MB）。macOS 的 ru_maxrss 单位是字节，Linux 是 KB。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def du_kb(path: str) -> int:
    out = subprocess.run(
        ["du", "-sk", path], capture_output=True, text=True, check=True
    )
    return int(out.stdout.split()[0])


# ----------------------------- 合成数据 -----------------------------

def gen_batch(batch_idx: int, batch_size: int) -> np.ndarray:
    """确定性生成一批 (batch_size, DIM) 随机单位向量。

    以 [SEED, batch_idx] 为种子，因此任意全局 chunk 下标 g 的向量都可以通过
    重生成第 g // batch_size 批还原（供删插阶段复用同一批向量）。
    """
    rng = np.random.default_rng([SEED, batch_idx])
    vecs = rng.standard_normal((batch_size, DIM), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def chunk_id(file_id: int, chunk_seq: int) -> str:
    return f"f{file_id:05d}-c{chunk_seq:03d}"


def build_docs(batch_vecs: np.ndarray, first_g: int) -> list:
    """把一批向量包装成 zvec.Doc（含 payload 字段）。"""
    docs = []
    for r in range(batch_vecs.shape[0]):
        g = first_g + r
        file_id, seq = divmod(g, CHUNKS_PER_FILE)
        docs.append(
            zvec.Doc(
                id=chunk_id(file_id, seq),
                vectors={VEC_FIELD: batch_vecs[r]},
                fields={"file_id": file_id, "chunk_seq": seq, "kind": KIND},
            )
        )
    return docs


def vectors_for_files(file_ids: list[int], batch_size: int) -> np.ndarray:
    """按确定性生成器还原给定 file 的全部 chunk 向量，返回 (len(file_ids)*50, DIM)。"""
    cache: dict[int, np.ndarray] = {}
    rows = []
    for f in file_ids:
        for seq in range(CHUNKS_PER_FILE):
            g = f * CHUNKS_PER_FILE + seq
            b, r = divmod(g, batch_size)
            if b not in cache:
                cache = {b: gen_batch(b, batch_size)}  # 只保留最近一批，控内存
            rows.append(cache[b][r])
    return np.stack(rows)


# ----------------------------- collection -----------------------------

def make_schema() -> CollectionSchema:
    return CollectionSchema(
        name="chunks",
        fields=[
            # file_id 是过滤主路径，加倒排索引
            FieldSchema(
                "file_id",
                DataType.INT64,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema("chunk_seq", DataType.INT32),
            FieldSchema("kind", DataType.STRING),
        ],
        vectors=[
            VectorSchema(
                VEC_FIELD,
                data_type=DataType.VECTOR_FP32,
                dimension=DIM,
                index_param=HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    m=HNSW_M,
                    ef_construction=HNSW_EF_CONSTRUCTION,
                ),
            )
        ],
    )


def fresh_collection(path: str):
    shutil.rmtree(path, ignore_errors=True)
    return zvec.create_and_open(path, make_schema())


# ----------------------------- 阶段实现 -----------------------------

def stage_insert(coll, n_docs: int, batch_size: int) -> dict:
    log(f"阶段1 插入开始: {n_docs} 条, 每批 {batch_size}")
    per_batch_gen, per_batch_build, per_batch_insert = [], [], []
    done = 0
    t_all = time.perf_counter()
    n_batches = n_docs // batch_size
    for b in range(n_batches):
        t0 = time.perf_counter()
        vecs = gen_batch(b, batch_size)
        t1 = time.perf_counter()
        docs = build_docs(vecs, first_g=b * batch_size)
        t2 = time.perf_counter()
        t_ins = 0.0
        for j in range(0, len(docs), INSERT_CALL_SIZE):
            t_s = time.perf_counter()
            statuses = coll.insert(docs[j : j + INSERT_CALL_SIZE])
            t_ins += time.perf_counter() - t_s
            bad = [s for s in statuses if not s.ok()]
            if bad:
                raise RuntimeError(f"batch {b}: {len(bad)} 条插入失败: {bad[0].message()}")
        t3 = time.perf_counter()
        per_batch_gen.append(t1 - t0)
        per_batch_build.append(t2 - t1)
        per_batch_insert.append(t_ins)
        done += batch_size
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            rate = done / (time.perf_counter() - t_all)
            log(
                f"  batch {b + 1}/{n_batches}  累计 {done} 条  "
                f"{rate:.0f} docs/s  peak_rss={rss_mb():.0f} MB"
            )
    total = time.perf_counter() - t_all
    count = coll.stats.doc_count
    assert count == n_docs, f"doc_count={count}, 期望 {n_docs}"
    result = {
        "total_seconds": total,
        "docs": n_docs,
        "throughput_docs_per_s": n_docs / total,
        "insert_call_only_seconds": sum(per_batch_insert),
        "insert_only_throughput_docs_per_s": n_docs / sum(per_batch_insert),
        "gen_seconds": sum(per_batch_gen),
        "build_docs_seconds": sum(per_batch_build),
        "per_batch_insert_ms_p50": float(np.percentile(per_batch_insert, 50)) * 1000,
        "per_batch_insert_ms_p95": float(np.percentile(per_batch_insert, 95)) * 1000,
        "peak_rss_mb_after": rss_mb(),
        "doc_count": count,
    }
    log(
        f"阶段1 完成: 端到端 {total:.1f}s ({result['throughput_docs_per_s']:.0f} docs/s), "
        f"纯 insert 调用 {result['insert_call_only_seconds']:.1f}s "
        f"({result['insert_only_throughput_docs_per_s']:.0f} docs/s), "
        f"peak_rss={result['peak_rss_mb_after']:.0f} MB"
    )
    return result


def stage_flush_optimize(coll) -> dict:
    log("阶段2 flush + optimize")
    t0 = time.perf_counter()
    coll.flush()
    t_flush = time.perf_counter() - t0
    t0 = time.perf_counter()
    coll.optimize()
    t_opt = time.perf_counter() - t0
    stats = coll.stats
    result = {
        "flush_seconds": t_flush,
        "optimize_seconds": t_opt,
        "doc_count": stats.doc_count,
        "index_completeness": dict(stats.index_completeness),
        "disk_kb_after_optimize": du_kb(coll.path),
        "peak_rss_mb": rss_mb(),
    }
    log(
        f"阶段2 完成: flush={t_flush:.2f}s optimize={t_opt:.2f}s "
        f"disk={result['disk_kb_after_optimize'] / 1024:.1f} MB "
        f"completeness={result['index_completeness']}"
    )
    return result


def pct(lat_ms: list[float]) -> dict:
    a = np.asarray(lat_ms)
    return {
        "n": len(a),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
    }


def run_knn_sweep(coll, qvecs: np.ndarray, ef: int | None) -> list[float]:
    param = HnswQueryParam(ef=ef) if ef is not None else HnswQueryParam()
    lat = []
    for i in range(qvecs.shape[0]):
        t0 = time.perf_counter()
        res = coll.query(
            queries=Query(field_name=VEC_FIELD, vector=qvecs[i], param=param),
            topk=10,
        )
        lat.append((time.perf_counter() - t0) * 1000)
        if len(res) != 10:
            raise RuntimeError(f"query {i} 返回 {len(res)} 条")
    return lat


def stage_query(coll, n_queries: int, n_filtered: int, batch_size: int) -> dict:
    log("阶段3 查询延迟")
    rng = np.random.default_rng([SEED, 10**9])
    qvecs = rng.standard_normal((n_queries, DIM), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)

    # 预热 5 条（含 mmap 页与首查路径）
    for i in range(5):
        coll.query(queries=Query(field_name=VEC_FIELD, vector=qvecs[i]), topk=10)

    result: dict = {}
    lat_def = run_knn_sweep(coll, qvecs, ef=None)
    result["knn_default_ef300"] = pct(lat_def)
    lat_100 = run_knn_sweep(coll, qvecs, ef=100)
    result["knn_ef100"] = pct(lat_100)
    log(
        f"  topk=10 默认ef: p50={result['knn_default_ef300']['p50_ms']:.2f}ms "
        f"p95={result['knn_default_ef300']['p95_ms']:.2f}ms "
        f"p99={result['knn_default_ef300']['p99_ms']:.2f}ms | "
        f"ef=100: p50={result['knn_ef100']['p50_ms']:.2f}ms "
        f"p95={result['knn_ef100']['p95_ms']:.2f}ms"
    )

    n_files_total = coll.stats.doc_count // CHUNKS_PER_FILE
    fids = rng.choice(n_files_total, size=n_filtered, replace=False).tolist()
    lat_f, checked = [], 0
    for i, fid in enumerate(fids):
        t0 = time.perf_counter()
        res = coll.query(
            queries=Query(field_name=VEC_FIELD, vector=qvecs[i]),
            topk=10,
            filter=f"file_id = {int(fid)}",
        )
        lat_f.append((time.perf_counter() - t0) * 1000)
        for d in res:
            assert d.fields["file_id"] == int(fid), f"过滤失效: {d.id}"
            checked += 1
    result["knn_filtered_by_file_id"] = pct(lat_f)
    log(
        f"  file_id 过滤: p50={result['knn_filtered_by_file_id']['p50_ms']:.2f}ms "
        f"p95={result['knn_filtered_by_file_id']['p95_ms']:.2f}ms "
        f"(校验 {checked} 条结果均属于目标 file)"
    )

    # 自召回 sanity：以库内 chunk 自身向量为查询，应命中自身（top10）
    sample_gs = rng.choice(
        coll.stats.doc_count, size=20, replace=False
    ).tolist()
    hits = 0
    for g in sample_gs:
        b, r = divmod(int(g), batch_size)
        v = gen_batch(b, batch_size)[r]
        res = coll.query(queries=Query(field_name=VEC_FIELD, vector=v), topk=10)
        fid, seq = divmod(int(g), CHUNKS_PER_FILE)
        if any(d.id == chunk_id(fid, seq) for d in res):
            hits += 1
    result["self_recall_at10"] = hits / len(sample_gs)
    log(f"  自召回 sanity (20 条精确向量): self-recall@10 = {hits}/{len(sample_gs)}")
    return result


def files_vectors_pairs(file_ids: list[int], batch_size: int) -> list:
    """还原 file -> (file_id, chunk_seq, vector) 列表，供重插。"""
    vecs = vectors_for_files(file_ids, batch_size)
    docs_meta = []
    k = 0
    for f in file_ids:
        for seq in range(CHUNKS_PER_FILE):
            docs_meta.append((f, seq, vecs[k]))
            k += 1
    return docs_meta


def reinsert(coll, meta: list, label: str, upsert_batch: int = INSERT_CALL_SIZE) -> float:
    t0 = time.perf_counter()
    for i in range(0, len(meta), upsert_batch):
        part = meta[i : i + upsert_batch]
        docs = [
            zvec.Doc(
                id=chunk_id(f, seq),
                vectors={VEC_FIELD: v},
                fields={"file_id": f, "chunk_seq": seq, "kind": KIND},
            )
            for (f, seq, v) in part
        ]
        statuses = coll.upsert(docs)
        bad = [s for s in statuses if not s.ok()]
        if bad:
            raise RuntimeError(f"{label}: 重插失败 {bad[0].message()}")
    dt = time.perf_counter() - t0
    log(f"  {label}: 重插 {len(meta)} 条耗时 {dt:.2f}s ({len(meta) / dt:.0f} docs/s)")
    return dt


def stage_delete_reinsert(coll, n_files: int, batch_size: int) -> dict:
    log(f"阶段4 按 file_id 删插: {n_files} 个 file x {CHUNKS_PER_FILE} chunk")
    rng = np.random.default_rng([SEED, 10**9 + 1])
    n_files_total = coll.stats.doc_count // CHUNKS_PER_FILE
    chosen = rng.choice(n_files_total, size=2 * n_files, replace=False)
    ids_files = sorted(int(f) for f in chosen[:n_files])           # delete(ids) 组
    filter_files = sorted(int(f) for f in chosen[n_files:])        # delete_by_filter 组

    result: dict = {}
    before = coll.stats.doc_count

    # (a) delete(ids)：一次调用删 5000 条
    id_list = [
        chunk_id(f, seq) for f in ids_files for seq in range(CHUNKS_PER_FILE)
    ]
    t0 = time.perf_counter()
    statuses = coll.delete(id_list)
    dt_delete_ids = time.perf_counter() - t0
    bad = [s for s in statuses if not s.ok()]
    if bad:
        raise RuntimeError(f"delete(ids) 失败 {len(bad)} 条: {bad[0].message()}")
    after_ids = coll.stats.doc_count
    assert after_ids == before - len(id_list), (
        f"删除后 doc_count={after_ids}, 期望 {before - len(id_list)}"
    )
    result["delete_ids"] = {
        "files": n_files,
        "docs": len(id_list),
        "seconds": dt_delete_ids,
        "docs_per_s": len(id_list) / dt_delete_ids,
        "doc_count_after": after_ids,
    }
    log(
        f"  delete(ids) 一次调用删 {len(id_list)} 条: {dt_delete_ids:.3f}s "
        f"({len(id_list) / dt_delete_ids:.0f} docs/s), doc_count={after_ids}"
    )

    # (b) delete_by_filter：逐 file 调用（knowledge-vault 实际会用到的语义）
    t0 = time.perf_counter()
    for f in filter_files:
        coll.delete_by_filter(f"file_id = {f}")
    dt_delete_filter = time.perf_counter() - t0
    after_filter = coll.stats.doc_count
    assert after_filter == after_ids - n_files * CHUNKS_PER_FILE
    result["delete_by_filter"] = {
        "files": n_files,
        "docs": n_files * CHUNKS_PER_FILE,
        "seconds": dt_delete_filter,
        "docs_per_s": n_files * CHUNKS_PER_FILE / dt_delete_filter,
        "per_call_ms_p50": float(
            np.percentile([dt_delete_filter / n_files], 50)
        ) * 1000,
        "doc_count_after": after_filter,
    }
    log(
        f"  delete_by_filter 逐 file 删 {n_files} 个 file"
        f"（{n_files * CHUNKS_PER_FILE} 条）: {dt_delete_filter:.3f}s "
        f"({n_files * CHUNKS_PER_FILE / dt_delete_filter:.0f} docs/s)"
    )

    # 重插（upsert，向量与删除前完全相同）
    meta_ids = files_vectors_pairs(ids_files, batch_size)
    result["reinsert_upsert_ids_group"] = {
        "docs": len(meta_ids),
        "seconds": reinsert(coll, meta_ids, "delete(ids)组"),
    }
    meta_filter = files_vectors_pairs(filter_files, batch_size)
    result["reinsert_upsert_filter_group"] = {
        "docs": len(meta_filter),
        "seconds": reinsert(coll, meta_filter, "delete_by_filter组"),
    }
    final = coll.stats.doc_count
    assert final == before, f"重插后 doc_count={final}, 期望 {before}"
    result["doc_count_final"] = final
    result["peak_rss_mb"] = rss_mb()
    log(f"  重插完成, doc_count 恢复到 {final}, peak_rss={rss_mb():.0f} MB")
    return result


def stage_rebuild(n_docs: int, batch_size: int) -> dict:
    log("阶段5 全量重建: drop 目录 + 新建 + 全量重插")
    result: dict = {}
    t0 = time.perf_counter()
    shutil.rmtree(MAIN_DIR, ignore_errors=True)
    result["drop_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    coll = zvec.create_and_open(MAIN_DIR, make_schema())
    result["create_seconds"] = time.perf_counter() - t0
    ins = stage_insert(coll, n_docs, batch_size)
    t0 = time.perf_counter()
    coll.flush()
    result["flush_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    coll.optimize()
    result["optimize_seconds"] = time.perf_counter() - t0
    result["insert"] = ins
    result["end_to_end_seconds"] = (
        result["drop_seconds"]
        + result["create_seconds"]
        + ins["total_seconds"]
        + result["flush_seconds"]
        + result["optimize_seconds"]
    )
    result["disk_kb_final"] = du_kb(coll.path)
    result["peak_rss_mb"] = rss_mb()
    try:
        result["hnsw_storage_mode"] = coll._obj._debug_hnsw_storage_mode(VEC_FIELD)
    except Exception as e:  # noqa: BLE001 - debug API，缺失不影响结论
        result["hnsw_storage_mode"] = f"unavailable: {e}"
    log(
        f"阶段5 完成: 端到端 {result['end_to_end_seconds']:.1f}s "
        f"(drop {result['drop_seconds']:.2f}s + create {result['create_seconds']:.2f}s "
        f"+ insert {ins['total_seconds']:.1f}s + flush {result['flush_seconds']:.2f}s "
        f"+ optimize {result['optimize_seconds']:.2f}s), "
        f"disk={result['disk_kb_final'] / 1024:.1f} MB"
    )
    return coll, result


# ----------------------------- 主流程 -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="1 万条快速自检版（默认 50 万条正式基准）"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="覆盖每批插入条数（默认 full=10000 / quick=2000）"
    )
    args = parser.parse_args()

    if args.quick:
        n_docs, batch_size = 10_000, 2_000
        n_query, n_filtered, n_del_files = 100, 20, 20
        mode = "quick"
    else:
        n_docs, batch_size = 500_000, 10_000
        n_query, n_filtered, n_del_files = 100, 20, 100
        mode = "full"
    if args.batch:
        batch_size = args.batch
    assert batch_size % CHUNKS_PER_FILE == 0, "batch 必须整除每 file chunk 数"
    assert n_docs % batch_size == 0

    os.makedirs(STATE_DIR, exist_ok=True)
    log(f"zvec {zvec.version('zvec')} | python {platform.python_version()} | "
        f"{platform.machine()} | macOS {platform.mac_ver()[0]}")
    log(f"mode={mode} n_docs={n_docs} batch={batch_size} "
        f"files={n_docs // CHUNKS_PER_FILE} dim={DIM} "
        f"hnsw(m={HNSW_M}, efc={HNSW_EF_CONSTRUCTION})")

    results = {
        "mode": mode,
        "env": {
            "zvec": zvec.version("zvec"),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "numpy": np.__version__,
            "mem_gb": 16,
        },
        "params": {
            "n_docs": n_docs,
            "batch_size": batch_size,
            "dim": DIM,
            "chunks_per_file": CHUNKS_PER_FILE,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "metric": "COSINE",
            "vector_dtype": "VECTOR_FP32",
            "payload_fields": ["file_id(INT64,倒排)", "chunk_seq(INT32)", "kind(STRING)"],
            "collection_option": "默认 (enable_mmap=True)",
        },
    }

    # 阶段 1-2：建库插入
    coll = fresh_collection(MAIN_DIR)
    results["insert"] = stage_insert(coll, n_docs, batch_size)
    results["flush_optimize"] = stage_flush_optimize(coll)

    # 阶段 3：查询
    results["query"] = stage_query(coll, n_query, n_filtered, batch_size)

    # 阶段 4：删插
    results["delete_reinsert"] = stage_delete_reinsert(coll, n_del_files, batch_size)

    # 收尾：关闭后重开验证持久化
    coll.close()
    t0 = time.perf_counter()
    coll2 = zvec.open(MAIN_DIR)
    results["reopen_seconds"] = time.perf_counter() - t0
    results["doc_count_after_reopen"] = coll2.stats.doc_count
    log(f"重开 collection: {results['reopen_seconds']:.2f}s, "
        f"doc_count={results['doc_count_after_reopen']}")
    coll2.close()

    # 阶段 5：全量重建
    coll3, rebuild = stage_rebuild(n_docs, batch_size)

    # 重建后复测查询
    log("阶段5 附带: 重建后复测 100 条查询")
    rng = np.random.default_rng([SEED, 10**9])
    qvecs = rng.standard_normal((n_query, DIM), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
    for i in range(5):
        coll3.query(queries=Query(field_name=VEC_FIELD, vector=qvecs[i]), topk=10)
    rebuild["query_after_rebuild"] = pct(run_knn_sweep(coll3, qvecs, ef=None))
    disk_kb = du_kb(coll3.path)
    rebuild["disk_kb_final"] = disk_kb
    log(f"  重建后查询 p50={rebuild['query_after_rebuild']['p50_ms']:.2f}ms "
        f"p95={rebuild['query_after_rebuild']['p95_ms']:.2f}ms")
    coll3.close()

    results["rebuild"] = rebuild

    out_path = os.path.join(STATE_DIR, f"bench_results_{mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"结果已写入 {out_path}")

    log("===== 汇总 =====")
    ins = results["insert"]
    q = results["query"]
    d = results["delete_reinsert"]
    rb = results["rebuild"]
    print(json.dumps({
        "插入吞吐(docs/s, 端到端)": round(ins["throughput_docs_per_s"]),
        "插入吞吐(docs/s, 纯insert调用)": round(ins["insert_only_throughput_docs_per_s"]),
        "查询p50/p95/p99_ms(默认ef300)": [
            round(q["knn_default_ef300"]["p50_ms"], 2),
            round(q["knn_default_ef300"]["p95_ms"], 2),
            round(q["knn_default_ef300"]["p99_ms"], 2)],
        "过滤查询p50/p95_ms": [
            round(q["knn_filtered_by_file_id"]["p50_ms"], 2),
            round(q["knn_filtered_by_file_id"]["p95_ms"], 2)],
        "self_recall_at10": q["self_recall_at10"],
        "删5000条耗时_s": round(d["delete_ids"]["seconds"], 3),
        "重插5000条耗时_s": round(d["reinsert_upsert_ids_group"]["seconds"], 2),
        "delete_by_filter_5000条耗时_s": round(d["delete_by_filter"]["seconds"], 3),
        "全量重建端到端_s": round(rb["end_to_end_seconds"], 1),
        "磁盘占用_MB": round(rb["disk_kb_final"] / 1024, 1),
        "峰值RSS_MB": round(rb["peak_rss_mb"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Spike 4：递归 CTE 多跳图遍历基准（全合成数据，无任何真实 vault 内容）。

目的：在本地 SQLite 自证「十万边量级下，递归 CTE 做 1/2/3 跳有向遍历」的
延迟是否可用（个人库单用户标准：p95 数百 ms 内可接受，秒级需调整）。

方法：
- 合成随机有向图，edges 表 WITHOUT ROWID 主键 (src,dst) + dst 反向索引。
- 递归 CTE 从随机起点做 x 跳扩展，统计单次延迟 p50/p95/max 与
  返回的 DISTINCT 节点数（扇出规模参照）。
- 附加对照：UNION vs UNION ALL（去重在 CTE 内 vs 外）、
  PRAGMA cache_size/temp_store/mmap 开关前后对比。

运行：
    ~/llwwds_application/knowledge-vault/venv-mac/bin/python \\
        experiments/bench_recursive_cte.py          # 10 万边 + 5 万边
    ~/llwwds_application/knowledge-vault/venv-mac/bin/python \\
        experiments/bench_recursive_cte.py --quick  # 仅 1 万边快检

运行时产物（数据库）只写 ~/llwwds_application/knowledge-vault/state/spike4_cte/，
重跑前自动删除本脚本自己生成的 db 重建，幂等。
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

SEED = 20261003          # 图生成与起点抽样统一 seed，保证可复现
N_STARTS = 100           # 主矩阵：每个 (规模, 跳数) 的随机起点数
N_STARTS_EXTRA = 50      # 附加对照使用的起点数（取主矩阵前 N_STARTS_EXTRA 个）

STATE_DIR = Path.home() / "llwwds_application" / "knowledge-vault" / "state" / "spike4_cte"

# 规模名 -> (目标边数, 节点数, 数据库文件名)
SCALES = {
    "100k": (100_000, 20_000, "edges_100k.db"),
    "50k": (50_000, 10_000, "edges_50k.db"),
    "10k": (10_000, 2_000, "edges_10k.db"),
}

DDL = """
CREATE TABLE edges(
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    PRIMARY KEY(src, dst)
) WITHOUT ROWID;
CREATE INDEX idx_edges_dst ON edges(dst);
"""

# 基准模板：UNION 在 CTE 内按 (node, depth) 去重，外层再对 node 去重计数
SQL_BASE = """
WITH RECURSIVE walk(node, depth) AS (
    SELECT :start, 0
    UNION
    SELECT e.dst, w.depth + 1
    FROM walk w JOIN edges e ON e.src = w.node
    WHERE w.depth < :max_depth
)
SELECT COUNT(DISTINCT node) FROM walk
"""

# 对照：UNION ALL 遍历期间不去重，外层一次性 DISTINCT 后计数
SQL_UNION_ALL = """
WITH RECURSIVE walk(node, depth) AS (
    SELECT :start, 0
    UNION ALL
    SELECT e.dst, w.depth + 1
    FROM walk w JOIN edges e ON e.src = w.node
    WHERE w.depth < :max_depth
)
SELECT COUNT(*) FROM (SELECT DISTINCT node FROM walk)
"""

# 附加对照开启的 PRAGMA（对比默认配置）
PERF_PRAGMAS = (
    ("cache_size=-65536（64MB）", "PRAGMA cache_size=-65536"),
    ("temp_store=MEMORY", "PRAGMA temp_store=MEMORY"),
    ("mmap_size=256MB", "PRAGMA mmap_size=268435456"),
)

# ---------------------------------------------------------------------------
# 建库
# ---------------------------------------------------------------------------


def build_db(path: Path, n_edges: int, n_nodes: int, seed: int) -> tuple[int, int]:
    """删除旧库后重建合成随机图，返回 (实际边数, db 文件字节数)。幂等。"""
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")  # 一次性基准库，构建阶段不需要持久性保证
    conn.executescript(DDL)

    rng = random.Random(seed)
    t0 = time.perf_counter()
    # 先在内存里去重采样，保证落库边数精确等于目标值
    edge_set: set[tuple[int, int]] = set()
    while len(edge_set) < n_edges:
        edge_set.add((rng.randrange(n_nodes), rng.randrange(n_nodes)))
    conn.executemany("INSERT OR IGNORE INTO edges(src, dst) VALUES(?, ?)", edge_set)
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    build_s = time.perf_counter() - t0

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db_bytes = path.stat().st_size
    conn.close()
    print(f"  建库完成：{total} 边 / {n_nodes} 节点，耗时 {build_s:.1f}s，db {db_bytes / 1e6:.1f} MB")
    return total, db_bytes


def open_conn(path: Path, perf: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    if perf:
        for label, sql in PERF_PRAGMAS:
            conn.execute(sql)
            print(f"  PRAGMA ON: {label}")
    return conn


# ---------------------------------------------------------------------------
# 基准执行
# ---------------------------------------------------------------------------


def bench(conn: sqlite3.Connection, sql: str, starts: list[int], max_depth: int) -> tuple[list[float], list[int]]:
    """逐起点执行 CTE，返回 (单次毫秒延迟列表, DISTINCT 节点数列表)。"""
    lats: list[float] = []
    cnts: list[int] = []
    for s in starts:
        t0 = time.perf_counter()
        (n,) = conn.execute(sql, {"start": s, "max_depth": max_depth}).fetchone()
        lats.append((time.perf_counter() - t0) * 1000.0)
        cnts.append(n)
    return lats, cnts


def summarize(lat: list[float]) -> tuple[float, float, float]:
    """返回 (p50, p95, max)，单位 ms。"""
    s = sorted(lat)
    p50 = statistics.median(s)
    p95 = s[max(0, min(len(s) - 1, round(0.95 * (len(s) - 1))))]
    return p50, p95, s[-1]


def fmt_row(scale: str, variant: str, hop: int, lats: list[float], cnts: list[int]) -> str:
    p50, p95, mx = summarize(lats)
    return (
        f"{scale:>5} | {variant:<28} | {hop} 跳 | "
        f"p50 {p50:8.2f} | p95 {p95:8.2f} | max {mx:8.2f} | "
        f"节点数均值 {statistics.mean(cnts):9.1f} | n={len(lats)}"
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser(description="递归 CTE 多跳遍历基准（合成数据）")
    ap.add_argument("--quick", action="store_true", help="仅跑 1 万边快检规模")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]} / sqlite3 {sqlite3.sqlite_version} / seed {SEED}")
    print(f"state 目录: {STATE_DIR}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    scale_keys = ["10k"] if args.quick else ["100k", "50k"]
    # 汇总行：(规模, 变体, 跳数, 延迟列表, 节点数列表)
    results: list[tuple[str, str, int, list[float], list[int]]] = []

    for key in scale_keys:
        n_edges, n_nodes, fname = SCALES[key]
        path = STATE_DIR / fname
        print(f"\n=== 规模 {key}：目标 {n_edges} 边 / {n_nodes} 节点（{fname}）===")
        build_db(path, n_edges, n_nodes, seed=SEED)

        rng = random.Random(SEED + 1)
        starts_all = [rng.randrange(n_nodes) for _ in range(N_STARTS)]
        starts_extra = starts_all[:N_STARTS_EXTRA]

        # --- 变体 1：基准模板（UNION 去重在 CTE 内，默认 PRAGMA），100 起点 ---
        conn = open_conn(path)
        print(f"-- 基准 UNION（CTE 内去重，默认 PRAGMA），{N_STARTS} 起点 --")
        for hop in (1, 2, 3):
            lats, cnts = bench(conn, SQL_BASE, starts_all, hop)
            results.append((key, "base UNION(内去重)", hop, lats, cnts))
            print("  " + fmt_row(key, "base UNION(内去重)", hop, lats, cnts), flush=True)
        conn.close()

        # --- 变体 2：UNION ALL（去重挪到外层），50 起点 ---
        conn = open_conn(path)
        print(f"-- 对照 UNION ALL（外层 DISTINCT），{N_STARTS_EXTRA} 起点 --")
        for hop in (1, 2, 3):
            lats, cnts = bench(conn, SQL_UNION_ALL, starts_extra, hop)
            results.append((key, "UNION ALL(外去重)", hop, lats, cnts))
            print("  " + fmt_row(key, "UNION ALL(外去重)", hop, lats, cnts), flush=True)
        conn.close()

        # --- 变体 3：基准模板 + 性能 PRAGMA，50 起点 ---
        conn = open_conn(path, perf=True)
        print(f"-- 基准 UNION + 性能 PRAGMA，{N_STARTS_EXTRA} 起点 --")
        for hop in (1, 2, 3):
            lats, cnts = bench(conn, SQL_BASE, starts_extra, hop)
            results.append((key, "UNION+PRAGMA", hop, lats, cnts))
            print("  " + fmt_row(key, "UNION+PRAGMA", hop, lats, cnts), flush=True)
        conn.close()

    # --- 汇总表 ---
    print("\n=== 汇总（附加对照统一取前 50 个起点，与主矩阵同 seed 可直接对比）===")
    print("规模 | 变体 | 跳数 | 延迟 ms (p50/p95/max) | DISTINCT 节点数均值 | 样本数")
    for key in scale_keys:
        for variant in ("base UNION(内去重)", "UNION ALL(外去重)", "UNION+PRAGMA"):
            for hop in (1, 2, 3):
                rows = [(l, c) for (s, v, h, l, c) in results if s == key and v == variant and h == hop]
                if not rows:
                    continue
                lat_flat, cnt_flat = rows[0]
                if variant != "base UNION(内去重)":
                    lat_flat, cnt_flat = lat_flat[:N_STARTS_EXTRA], cnt_flat[:N_STARTS_EXTRA]
                p50, p95, mx = summarize(lat_flat)
                print(
                    f"{key} | {variant} | {hop} | "
                    f"{p50:.2f} / {p95:.2f} / {mx:.2f} | "
                    f"{statistics.mean(cnt_flat):.1f} | {len(lat_flat)}"
                )

    print(f"\n总耗时 {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

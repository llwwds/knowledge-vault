"""edges 图：写入与递归 CTE 多跳扩展。

口径沿用 docs/spike-4-recursive-cte.md 的基准模板：
- CTE 内用 ``UNION``（非 UNION ALL）按 ``(node, depth)`` 去重，防止环形图的行数
  失控；同一节点在不同深度仍可能出现多行（UNION 去重按整行），出口处按节点取
  ``MIN(depth)`` 折叠为"首达深度"。
- 出口排除起点本身（扩展语义是"取邻居上下文"），``distinct_count`` 即去重后
  的邻居数。
- 方向默认双向（``both``）：正向走 PK 前缀（``e.src = w.node``），反向走
  ``idx_edges_dst``（``e.dst = w.node``），两条 UNION 臂。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExpandResult:
    """图扩展结果。

    ``nodes``: ``(file_id, depth)`` 列表，depth 为首达深度（1 起），
    按 (depth, node) 升序；``distinct_count`` = 去重后的节点数。
    """

    nodes: list[tuple[int, int]]
    distinct_count: int


def add_edge(conn: sqlite3.Connection, src: int, dst: int, edge_type: str) -> bool:
    """加一条有向边（幂等：重复边忽略）。返回是否新写入。"""
    cur = conn.execute(
        "INSERT OR IGNORE INTO edges(src, dst, edge_type) VALUES (?, ?, ?)",
        (src, dst, edge_type),
    )
    return cur.rowcount > 0


def remove_edges_by_src(
    conn: sqlite3.Connection, src: int, edge_type: str | None = None
) -> int:
    """删除某节点的出边（``edge_type=None`` 删全部类型），返回删除条数。"""
    if edge_type is None:
        cur = conn.execute("DELETE FROM edges WHERE src = ?", (src,))
    else:
        cur = conn.execute(
            "DELETE FROM edges WHERE src = ? AND edge_type = ?", (src, edge_type)
        )
    return cur.rowcount


def expand(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    max_hops: int = 1,
    edge_types: Sequence[str] | None = None,
    direction: str = "both",
) -> ExpandResult:
    """从 ``file_id`` 出发做最多 ``max_hops`` 跳的图扩展（递归 CTE）。

    ``direction``: ``out``（沿 src→dst）/ ``in``（沿 dst→src）/ ``both``（默认）。
    ``edge_types`` 为 None 或空列表时不过滤边类型。
    """
    if max_hops < 0:
        raise ValueError(f"max_hops 不能为负，收到 {max_hops}")
    direction = direction.lower()
    if direction not in ("out", "in", "both"):
        raise ValueError(f"direction 必须是 out|in|both，收到 {direction!r}")

    type_sql = ""
    type_params: list[object] = []
    if edge_types:
        type_sql = " AND e.edge_type IN (%s)" % ", ".join("?" for _ in edge_types)
        type_params = list(edge_types)

    arms: list[tuple[str, list[object]]] = []
    if direction in ("out", "both"):
        arms.append((
            "SELECT e.dst, w.depth + 1 FROM walk w "
            "JOIN edges e ON e.src = w.node "
            "WHERE w.depth < ?" + type_sql,
            [max_hops, *type_params],
        ))
    if direction in ("in", "both"):
        arms.append((
            "SELECT e.src, w.depth + 1 FROM walk w "
            "JOIN edges e ON e.dst = w.node "
            "WHERE w.depth < ?" + type_sql,
            [max_hops, *type_params],
        ))

    parts = ["WITH RECURSIVE walk(node, depth) AS (", "    SELECT ?, 0"]
    params: list[object] = [file_id]
    for arm_sql, arm_params in arms:
        parts.append("    UNION")
        parts.append("    " + arm_sql)
        params.extend(arm_params)
    parts.append(")")
    parts.append(
        "SELECT node, MIN(depth) AS depth FROM walk "
        "WHERE node != ? GROUP BY node ORDER BY depth, node"
    )
    params.append(file_id)

    rows = conn.execute("\n".join(parts), params).fetchall()
    nodes = [(int(row["node"]), int(row["depth"])) for row in rows]
    return ExpandResult(nodes=nodes, distinct_count=len(nodes))

"""graph：边写入与递归 CTE 多跳扩展（spike-4 口径）。"""

from __future__ import annotations

import pytest

from knowledge_vault import add_edge, expand, remove_edges_by_src


class TestEdgeWriting:
    def test_add_edge_idempotent(self, conn):
        assert add_edge(conn, 1, 2, "link") is True
        assert add_edge(conn, 1, 2, "link") is False
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1

    def test_same_pair_different_types_distinct(self, conn):
        add_edge(conn, 1, 2, "link")
        assert add_edge(conn, 1, 2, "mention") is True
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2

    def test_remove_edges_by_src(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 1, 3, "mention")
        add_edge(conn, 2, 3, "link")

        assert remove_edges_by_src(conn, 1, "mention") == 1
        assert remove_edges_by_src(conn, 1) == 1          # 剩余的 link 出边
        assert remove_edges_by_src(conn, 1) == 0
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


class TestExpand:
    def test_chain_hops_1_to_3(self, conn):
        for src, dst in [(1, 2), (2, 3), (3, 4)]:
            add_edge(conn, src, dst, "link")

        assert expand(conn, 1, max_hops=1).nodes == [(2, 1)]
        assert expand(conn, 1, max_hops=2).nodes == [(2, 1), (3, 2)]
        three = expand(conn, 1, max_hops=3)
        assert three.nodes == [(2, 1), (3, 2), (4, 3)]
        assert three.distinct_count == 3

    def test_max_hops_zero_returns_empty(self, conn):
        add_edge(conn, 1, 2, "link")
        result = expand(conn, 1, max_hops=0)
        assert result.nodes == []
        assert result.distinct_count == 0

    def test_start_node_excluded(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 2, 1, "link")                      # 环回到起点
        assert [n for n, _ in expand(conn, 1, max_hops=2).nodes] == [2]

    def test_direction(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 3, 1, "link")

        assert expand(conn, 1, max_hops=1, direction="out").nodes == [(2, 1)]
        assert expand(conn, 1, max_hops=1, direction="in").nodes == [(3, 1)]
        assert expand(conn, 1, max_hops=1).nodes == [(2, 1), (3, 1)]  # 默认双向

    def test_min_depth_on_multiple_paths(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 2, 3, "link")
        add_edge(conn, 1, 3, "link")                      # 3 可在深度 1 和 2 触达
        assert expand(conn, 1, max_hops=2).nodes == [(2, 1), (3, 1)]  # 首达深度

    def test_cycle_terminates(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 2, 3, "link")
        add_edge(conn, 3, 1, "link")
        result = expand(conn, 1, max_hops=3)              # UNION 去重保证终止
        assert [n for n, _ in result.nodes] == [2, 3]
        assert result.distinct_count == 2

    def test_edge_type_filter(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 1, 3, "mention")

        assert expand(conn, 1, max_hops=1, edge_types=["mention"]).nodes == [(3, 1)]
        both = expand(conn, 1, max_hops=1, edge_types=["link", "mention"])
        assert both.distinct_count == 2
        empty_filter = expand(conn, 1, max_hops=1, edge_types=[])
        assert empty_filter.distinct_count == 2           # 空列表 = 不过滤

    def test_edge_type_filter_multi_hop(self, conn):
        add_edge(conn, 1, 2, "link")
        add_edge(conn, 2, 3, "mention")
        assert expand(conn, 1, max_hops=2, edge_types=["link"]).nodes == [(2, 1)]
        assert expand(conn, 1, max_hops=2, edge_types=["link", "mention"]).nodes == [
            (2, 1), (3, 2),
        ]

    def test_isolated_node(self, conn):
        result = expand(conn, 42, max_hops=2)
        assert result.nodes == []
        assert result.distinct_count == 0

    def test_invalid_args(self, conn):
        with pytest.raises(ValueError):
            expand(conn, 1, max_hops=-1)
        with pytest.raises(ValueError):
            expand(conn, 1, direction="sideways")

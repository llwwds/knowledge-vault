"""RRF 融合的数学正确性（score = Σ 1/(k + rank)，rank 从 1 起）。"""

from __future__ import annotations

import pytest

from knowledge_vault.retrieve import rrf_fuse


class TestRrfFuse:
    def test_single_list_score_formula(self):
        fused = rrf_fuse([["a", "b", "c"]], k=60)
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 62)
        assert fused["c"] == pytest.approx(1 / 63)

    def test_rank_starts_at_one(self):
        # 单元素列表：rank=1 → 1/(k+1)
        assert rrf_fuse([["x"]], k=60) == {"x": pytest.approx(1 / 61)}

    def test_scores_descend_by_rank(self):
        fused = rrf_fuse([["a", "b", "c"]])
        assert fused["a"] > fused["b"] > fused["c"]

    def test_cross_list_summation(self):
        # 同一候选在两路分别排 rank1 / rank2 → 贡献相加
        fused = rrf_fuse([["a", "b"], ["c", "a"]], k=60)
        assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["b"] == pytest.approx(1 / 62)
        assert fused["c"] == pytest.approx(1 / 61)
        assert fused["a"] > fused["c"] > fused["b"]

    def test_three_way_summation(self):
        fused = rrf_fuse([["a"], ["a", "b"], ["b", "a"]], k=10)
        assert fused["a"] == pytest.approx(1 / 11 + 1 / 11 + 1 / 12)
        assert fused["b"] == pytest.approx(1 / 12 + 1 / 11)

    def test_k_parameter_changes_weights(self):
        assert rrf_fuse([["a"]], k=1)["a"] == pytest.approx(1 / 2)
        assert rrf_fuse([["a"]], k=1000)["a"] == pytest.approx(1 / 1001)

    def test_disjoint_lists_merge(self):
        fused = rrf_fuse([["a", "b"], ["c", "d"]])
        assert set(fused) == {"a", "b", "c", "d"}
        assert fused["a"] == pytest.approx(fused["c"])

    def test_empty_inputs(self):
        assert rrf_fuse([]) == {}
        assert rrf_fuse([[], []]) == {}
        assert rrf_fuse([["a"], []])["a"] == pytest.approx(1 / 61)

    def test_duplicate_within_one_list_counts_once(self):
        # 同一路内重复键只按首次出现名次计一次（防御性约定）
        fused = rrf_fuse([["a", "a", "b"]], k=60)
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 62)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            rrf_fuse([["a"]], k=0)
        with pytest.raises(ValueError):
            rrf_fuse([["a"]], k=-5)

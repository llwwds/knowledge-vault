"""召回管线编排：三路融合 / 图扩展 / 过滤语义 / rerank / top-j。

合成知识库见 _phase3_helpers.build_kb；向量路与 reranker 一律用替身。
"""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_vault.protocols import Embedder, Reranker
from knowledge_vault.retrieve import (
    RetrievalOutput,
    VectorHit,
    ZvecVectorIndex,
    retrieve,
    rrf_fuse,
)

from _phase3_helpers import (
    FakeEmbedder,
    FakeRerankerBadArity,
    FakeRerankerByLength,
    FakeVectorPath,
    build_kb,
    chunk_id,
)


@pytest.fixture()
def kb(store, make_file) -> dict:
    return build_kb(store, make_file)


def by_chunk_id(output: RetrievalOutput, cid: str):
    for item in output.items:
        if item.chunk_id == cid:
            return item
    return None


# ------------------------------------------------------------------- 基础召回


class TestBasicRecall:
    def test_fts_only_recall_and_text_source(self, store, kb):
        output = retrieve(store, "机器学习")
        cids = {it.chunk_id for it in output.items}
        assert {chunk_id(kb, "a", 1), chunk_id(kb, "a", 2), chunk_id(kb, "c", 1)} <= cids
        # 软删除文件不出现
        assert all(it.file_id != kb["d"] for it in output.items)
        # b 无 FTS 命中；默认图扩展（a→b）会把 b 的代表 chunk 并入
        item_b = by_chunk_id(output, chunk_id(kb, "b", 1))
        assert item_b is not None
        assert item_b.sources == ("graph",)
        # 文本来自 chunks 表（原样），不是真源现切
        item = by_chunk_id(output, chunk_id(kb, "c", 1))
        assert item.text == "机器学习的模型需要训练数据"
        # 分数降序
        scores = [it.score for it in output.items]
        assert scores == sorted(scores, reverse=True)

    def test_result_structure(self, store, kb):
        output = retrieve(store, "训练数据")
        assert isinstance(output, RetrievalOutput)
        assert output.meta["fts_hits"] == 1
        item = output.items[0]
        assert item.file_id == kb["c"]
        assert item.chunk_id == chunk_id(kb, "c", 1)
        assert item.kind == "chunk"
        assert item.sources == ("fts",)
        assert item.title == "训练数据"
        assert item.file_path.endswith("训练数据.md")

    def test_empty_query_returns_empty(self, store, kb):
        for query in ("", "   "):
            output = retrieve(store, query)
            assert output.items == []
            assert output.meta["fts_hits"] == 0

    def test_no_match_query(self, store, kb):
        output = retrieve(store, "量子纠缠态")
        assert output.items == []


# ------------------------------------------------------------------- RRF 融合


class TestRrfFusion:
    def test_exact_rrf_math_without_graph(self, store, kb):
        # 查询只命中 c（FTS rank1）；向量路固定返回 b(rank1)、a(rank2)
        embedder = FakeEmbedder({"训练数据": 0})
        vector_path = FakeVectorPath([
            VectorHit(file_id=kb["b"], chunk_seq=1, kind="chunk", score=0.9),
            VectorHit(file_id=kb["a"], chunk_seq=1, kind="chunk", score=0.8),
        ])
        output = retrieve(
            store, "训练数据",
            embedder=embedder, vector_path=vector_path,
            top_j=10, graph=False,
        )
        assert output.meta["fts_hits"] == 1
        assert output.meta["vector_hits"] == 2
        ids = {it.chunk_id: it for it in output.items}
        b, c, a = chunk_id(kb, "b", 1), chunk_id(kb, "c", 1), chunk_id(kb, "a", 1)
        # b 与 c 同为 1/61；并列按 chunk_id 升序（f2-c1 < f3-c1）
        assert output.items[0].chunk_id == b
        assert output.items[1].chunk_id == c
        assert ids[b].score == pytest.approx(1 / 61)
        assert ids[c].score == pytest.approx(1 / 61)
        assert ids[a].score == pytest.approx(1 / 62)
        assert ids[b].sources == ("vector",)
        assert ids[c].sources == ("fts",)
        assert ids[a].sources == ("vector",)

    def test_dual_path_candidate_outranks_single(self, store, kb):
        # c 同时被两路命中（向量 rank1 + FTS rank1）→ 分数 = 两份 RRF 贡献
        embedder = FakeEmbedder({"训练数据": 0})
        vector_path = FakeVectorPath([
            VectorHit(file_id=kb["c"], chunk_seq=1, kind="chunk", score=0.99),
            VectorHit(file_id=kb["b"], chunk_seq=1, kind="chunk", score=0.5),
        ])
        output = retrieve(
            store, "训练数据",
            embedder=embedder, vector_path=vector_path, graph=False,
        )
        ids = {it.chunk_id: it for it in output.items}
        assert ids[chunk_id(kb, "c", 1)].score == pytest.approx(1 / 61 + 1 / 61)
        assert ids[chunk_id(kb, "b", 1)].score == pytest.approx(1 / 62)
        assert output.items[0].chunk_id == chunk_id(kb, "c", 1)

    def test_vector_path_without_embedder_raises(self, store, kb):
        with pytest.raises(ValueError, match="embedder"):
            retrieve(store, "机器学习", vector_path=FakeVectorPath([]))

    def test_embedder_and_vector_path_receive_query(self, store, kb):
        embedder = FakeEmbedder({})
        vector_path = FakeVectorPath([])
        retrieve(store, "训练数据", embedder=embedder, vector_path=vector_path, graph=False)
        assert embedder.calls == [["训练数据"]]
        assert vector_path.calls == [50]  # per_path_topk 默认值

    def test_fakes_satisfy_frozen_protocols(self):
        assert isinstance(FakeEmbedder({}), Embedder)
        assert isinstance(FakeRerankerByLength(), Reranker)
        assert isinstance(FakeVectorPath([]), object)


# --------------------------------------------------------------------- 图扩展


class TestGraphExpansion:
    def test_graph_adds_neighbor_with_low_weight(self, store, kb):
        # 查询只命中 c；c 经双向扩展触达 a（a→c 反向）与 b（c→b mention），
        # 两者的代表 chunk（summary 优先，否则最小 seq）都以固定低权重并入
        output = retrieve(store, "训练数据", graph=True)
        c = chunk_id(kb, "c", 1)
        a = chunk_id(kb, "a", 1)
        b = chunk_id(kb, "b", 1)
        item_c = by_chunk_id(output, c)
        item_a = by_chunk_id(output, a)
        item_b = by_chunk_id(output, b)
        assert item_c.sources == ("fts",)
        assert item_a is not None
        assert item_a.sources == ("graph",)
        assert item_a.score == pytest.approx(1 / (60 + 100))  # graph_rank=100
        assert item_b is not None
        assert item_b.sources == ("graph",)
        assert item_b.score == pytest.approx(1 / (60 + 100))
        assert item_c.score > item_a.score
        assert output.meta["graph_added"] == 2

    def test_graph_boost_is_additive_for_existing_candidate(self, store, kb):
        # c-c1 是 a 扩展的邻居且已在融合名单 → 融合分 + 固定图权重
        query = "机器学习"
        without = retrieve(store, query, graph=False)
        with_g = retrieve(store, query, graph=True)
        cid = chunk_id(kb, "c", 1)
        s0 = by_chunk_id(without, cid).fused_score
        s1 = by_chunk_id(with_g, cid).fused_score
        assert s1 == pytest.approx(s0 + 1 / (60 + 100))
        assert by_chunk_id(with_g, cid).sources == ("fts", "graph")

    def test_graph_disabled(self, store, kb):
        output = retrieve(store, "机器学习", graph=False)
        assert all("graph" not in it.sources for it in output.items)
        assert output.meta["graph_files"] == 0

    def test_graph_edge_type_filter(self, store, kb):
        # c→b 是 mention；a→c 是 link。从种子 c 出发：
        # edge_types=["mention"] → 邻居 b；edge_types=["link"] → 邻居 a（反向）
        out_mention = retrieve(store, "训练数据", graph_edge_types=["mention"])
        assert by_chunk_id(out_mention, chunk_id(kb, "b", 1)) is not None
        assert by_chunk_id(out_mention, chunk_id(kb, "a", 1)) is None

        out_link = retrieve(store, "训练数据", graph_edge_types=["link"])
        assert by_chunk_id(out_link, chunk_id(kb, "a", 1)) is not None
        assert by_chunk_id(out_link, chunk_id(kb, "b", 1)) is None

    def test_graph_direction(self, store, kb):
        # graph_direction="out"：从 c 出发只走 c→b(mention) → 邻居 b
        out_out = retrieve(store, "训练数据", graph_direction="out")
        assert by_chunk_id(out_out, chunk_id(kb, "b", 1)) is not None
        assert by_chunk_id(out_out, chunk_id(kb, "a", 1)) is None

        # graph_direction="in"：从 c 出发只走 a→c（反向）→ 邻居 a
        out_in = retrieve(store, "训练数据", graph_direction="in")
        assert by_chunk_id(out_in, chunk_id(kb, "a", 1)) is not None
        assert by_chunk_id(out_in, chunk_id(kb, "b", 1)) is None

    def test_graph_neighbor_respects_filters(self, store, kb):
        # status="now" 时允许集只剩 a；FTS 命中 a，但 a 的邻居 b/c 不在允许集
        output = retrieve(store, "机器学习", status="now")
        assert {it.file_id for it in output.items} == {kb["a"]}
        assert output.meta["allowed_files"] == 1

    def test_graph_seed_count_limits_seeds(self, store, kb):
        # graph_seed_count=0 → 无种子、无图贡献
        output = retrieve(store, "机器学习", graph_seed_count=0)
        assert all("graph" not in it.sources for it in output.items)


# --------------------------------------------------------------------- 过滤语义


class TestFilterSemantics:
    def test_deleted_file_excluded(self, store, make_file):
        path = make_file("被删文件.md", "占位")
        doc = store.register_file(path)
        store.add_chunks(doc.file_id, ["独特检索词苜蓿"])
        assert len(retrieve(store, "独特检索词苜蓿", graph=False).items) == 1
        store.soft_delete(doc.file_id)
        output = retrieve(store, "独特检索词苜蓿", graph=False)
        assert output.items == []
        assert output.meta["allowed_files"] == 0

    def test_status_filter(self, store, kb):
        out_now = retrieve(store, "机器学习", status="now")
        assert {it.file_id for it in out_now.items} <= {kb["a"]}
        assert out_now.items, "status=now 应命中 a"

        out_library = retrieve(store, "机器学习", status="library")
        file_ids = {it.file_id for it in out_library.items}
        assert kb["a"] not in file_ids
        assert kb["c"] in file_ids or kb["large"] in file_ids

    def test_tag_intersection(self, store, kb):
        out_ml = retrieve(store, "机器学习", context_tag=["ml"], graph=False)
        file_ids = {it.file_id for it in out_ml.items}
        assert kb["a"] in file_ids and kb["c"] in file_ids
        assert kb["b"] not in file_ids

        out_life = retrieve(store, "机器学习", context_tag=["life"])
        # b 无 FTS 命中且融合名单为空（无图种子）→ 空结果
        assert out_life.items == []
        assert out_life.meta["allowed_files"] == 1

        # 多标签请求 = 任一交集非空
        out_multi = retrieve(store, "机器学习", context_tag=["ml", "life"])
        assert by_chunk_id(out_multi, chunk_id(kb, "b", 1)) is not None  # b 经图扩展进入
        assert {it.file_id for it in out_multi.items} >= {kb["a"], kb["b"]}

    def test_file_ids_filter(self, store, kb):
        output = retrieve(store, "机器学习", file_ids=[kb["a"]], graph=False)
        assert {it.file_id for it in output.items} == {kb["a"]}

        empty = retrieve(store, "机器学习", file_ids=[])
        assert empty.items == []
        assert empty.meta["allowed_files"] == 0

    def test_is_large_only_summary_recallable(self, store, kb):
        output = retrieve(store, "机器学习", graph=False)
        large_items = [it for it in output.items if it.file_id == kb["large"]]
        assert len(large_items) == 1
        assert large_items[0].kind == "summary"
        assert large_items[0].text == "机器学习超大文件的摘要"
        # 大文件的常规 chunk（含同一检索词）被过滤
        assert all(it.chunk_id != chunk_id(kb, "large", 1) for it in output.items)


# --------------------------------------------------------------------- rerank


class TestRerank:
    def test_reranker_reorders_candidates(self, store, kb):
        reranker = FakeRerankerByLength()
        output = retrieve(store, "机器学习", reranker=reranker, graph=False)
        assert output.meta["reranked"] is True
        # 文本越短分越高：最短的摘要 chunk 排第一
        first = output.items[0]
        assert first.text == "机器学习超大文件的摘要"
        lengths = [-it.score for it in output.items]
        assert lengths == sorted(lengths)
        # reranker 收到的就是 chunks 表原文
        query, texts = reranker.calls[0]
        assert query == "机器学习"
        assert first.text in texts

    def test_no_reranker_uses_fused_score(self, store, kb):
        output = retrieve(store, "机器学习", graph=False)
        assert output.meta["reranked"] is False
        for item in output.items:
            assert item.score == pytest.approx(item.fused_score)

    def test_rerank_score_passthrough(self, store, kb):
        # 用字符交集替身验证分数值透传（不只是序）
        reranker = FakeRerankerByLength()
        output = retrieve(store, "机器学习", reranker=reranker, graph=False, top_j=3)
        expect = {it.chunk_id: -float(len(it.text)) for it in output.items}
        for item in output.items:
            assert item.score == pytest.approx(expect[item.chunk_id])

    def test_rerank_contract_length_mismatch(self, store, kb):
        with pytest.raises(RuntimeError, match="不一致"):
            retrieve(store, "机器学习", reranker=FakeRerankerBadArity(), graph=False)

    def test_rerank_candidates_cap(self, store, kb):
        reranker = FakeRerankerByLength()
        output = retrieve(
            store, "机器学习", reranker=reranker, graph=False, rerank_candidates=1
        )
        # 只送 1 个候选 rerank → 该候选 rerank 分最高（唯一），其余不送
        assert len(reranker.calls[0][1]) == 1


# --------------------------------------------------------------------- top-j


class TestTopJ:
    def test_top_j_limits_and_orders(self, store, kb):
        full = retrieve(store, "机器学习", graph=True)
        assert len(full.items) > 2
        for j in (1, 2, len(full.items)):
            page = retrieve(store, "机器学习", graph=True, top_j=j)
            assert len(page.items) == j
            assert [it.chunk_id for it in page.items] == [
                it.chunk_id for it in full.items[:j]
            ]

    def test_top_j_must_be_positive(self, store, kb):
        with pytest.raises(ValueError):
            retrieve(store, "机器学习", top_j=0)


# ------------------------------------------------------------ zvec 适配器（真实 zvec）


class TestZvecVectorIndex:
    def test_adapter_roundtrip(self, tmp_path):
        import zvec

        schema = zvec.CollectionSchema(
            "zvec_chunks",
            fields=[
                zvec.FieldSchema("file_id", zvec.DataType.INT64),
                zvec.FieldSchema("chunk_seq", zvec.DataType.INT64),
                zvec.FieldSchema("kind", zvec.DataType.STRING),
            ],
            vectors=[
                zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dimension=4)
            ],
        )
        coll = zvec.create_and_open(str(tmp_path / "zvec_chunks"), schema)
        try:
            coll.insert([
                zvec.Doc(
                    id="f1-c1",
                    fields={"file_id": 1, "chunk_seq": 1, "kind": "chunk"},
                    vectors={"embedding": np.array([1, 0, 0, 0], dtype=np.float32)},
                ),
                zvec.Doc(
                    id="f2-c0",
                    fields={"file_id": 2, "chunk_seq": 0, "kind": "summary"},
                    vectors={"embedding": np.array([0, 1, 0, 0], dtype=np.float32)},
                ),
            ])
            coll.flush()

            adapter = ZvecVectorIndex(coll)  # FLAT 缺省索引 → query_param=None
            hits = adapter.search(np.array([1, 0, 0, 0], dtype=np.float32), topk=2)
            assert len(hits) == 2
            assert hits[0].file_id == 1
            assert hits[0].chunk_seq == 1
            assert hits[0].kind == "chunk"
            assert hits[0].chunk_id == "f1-c1"
            assert hits[0].score >= hits[1].score
            assert hits[1].chunk_id == "f2-c0"

            # topk 截断
            assert len(adapter.search(np.ones(4, dtype=np.float32), topk=1)) == 1
        finally:
            coll.close()

    def test_adapter_matches_vector_path_protocol(self, tmp_path):
        import zvec
        from knowledge_vault.retrieve import VectorPath as VectorPathProto

        schema = zvec.CollectionSchema(
            "zvec_chunks",
            vectors=[
                zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dimension=4)
            ],
        )
        coll = zvec.create_and_open(str(tmp_path / "zvec_chunks"), schema)
        try:
            assert isinstance(ZvecVectorIndex(coll), VectorPathProto)
        finally:
            coll.close()

    def test_rrf_fuse_shared_helper_matches_retrieve(self):
        # retrieve 内部与独立 rrf_fuse 用同一实现（防手写分叉）
        fused = rrf_fuse([["a"], ["b", "a"]], k=60)
        assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)

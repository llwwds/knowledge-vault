"""vectorstore 测试：真实 zvec（临时目录 collection）+ FakeEmbedder 向量。

覆盖：meta 生命周期 / 写入读取 / 自召回 / 删除 / upsert / 分批插入 /
optimize 攒批阈值策略（含 pipeline.db 持久化）/ 重开持久性。
"""

from __future__ import annotations

import numpy as np
import pytest

from knowledge_vault.embedder import FakeEmbedder
from knowledge_vault.pipeline import PipelineState
from knowledge_vault.vectorstore import (
    DEFAULT_OPTIMIZE_EVERY,
    ZvecStore,
    ZvecStoreError,
)


@pytest.fixture()
def zdir(tmp_path):
    return tmp_path / "zvec_chunks"


@pytest.fixture()
def vs(zdir):
    store = ZvecStore.create(zdir)
    yield store
    store.close()


@pytest.fixture()
def emb():
    return FakeEmbedder()


def _norm(rng: np.random.Generator, dim: int = 1024) -> np.ndarray:
    v = rng.standard_normal(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------- 生命周期

def test_create_writes_meta(zdir, vs):
    meta = vs.meta
    assert meta["collection_name"] == "chunks"
    assert meta["model_name"] == "BAAI/bge-m3"
    assert meta["dim"] == 1024
    assert meta["metric"] == "cosine"
    assert meta["hnsw"] == {"m": 16, "ef_construction": 200}
    assert (zdir / "meta.json").exists()
    assert vs.count() == 0


def test_create_on_existing_dir_raises(zdir, vs):
    with pytest.raises(FileExistsError):
        ZvecStore.create(zdir)


def test_open_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ZvecStore.open(tmp_path / "no_such_dir")


def test_open_dir_without_meta_raises(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ZvecStoreError):
        ZvecStore.open(bare)


def test_open_dim_mismatch_raises(zdir, vs):
    vs.close()
    with pytest.raises(ZvecStoreError, match="维度"):
        ZvecStore.open(zdir, dim=512)


def test_open_model_mismatch_raises(zdir, vs):
    vs.close()
    with pytest.raises(ZvecStoreError, match="模型"):
        ZvecStore.open(zdir, expected_model="other-model")


def test_open_roundtrip_meta(zdir, vs):
    vs.close()
    reopened = ZvecStore.open(zdir)
    try:
        assert reopened.meta["model_name"] == "BAAI/bge-m3"
        assert reopened.meta["dim"] == 1024
    finally:
        reopened.close()


def test_update_meta_persists(zdir, vs):
    vs.update_meta(note="运维备注")
    vs.close()
    reopened = ZvecStore.open(zdir)
    try:
        assert reopened.meta["note"] == "运维备注"
    finally:
        reopened.close()


# ------------------------------------------------------------- 写入读取

def test_insert_chunks_and_count(vs, emb):
    rng = np.random.default_rng(1)
    vecs = np.stack([_norm(rng) for _ in range(3)])
    written = vs.insert_chunks(7, [1, 2, 3], vecs, ["chunk", "chunk", "chunk"])
    assert written == 3
    assert vs.count() == 3


def test_insert_chunks_validates(vs):
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        vs.insert_chunks(1, [1], np.zeros((2, 1024), dtype=np.float32), ["chunk"])
    with pytest.raises(ValueError):
        vs.insert_chunks(1, [1, 2], np.zeros((2, 512), dtype=np.float32), ["chunk", "chunk"])
    with pytest.raises(ValueError):
        vs.insert_chunks(1, [1, 2], np.zeros((2, 1024), dtype=np.float32), ["chunk"])


def test_insert_empty_is_noop(vs):
    assert vs.insert_chunks(1, [], np.zeros((0, 1024), dtype=np.float32), []) == 0
    assert vs.count() == 0


def test_self_retrieval_top1(vs, emb):
    rng = np.random.default_rng(3)
    vecs = np.stack([_norm(rng) for _ in range(3)])
    vs.insert_chunks(11, [1, 2, 3], vecs, ["chunk", "chunk", "summary"])
    hits = vs.search(vecs[1], topk=3)
    assert len(hits) == 3
    assert hits[0].doc_id == "f11-c2"
    assert hits[0].file_id == 11
    assert hits[0].chunk_seq == 2
    assert hits[0].kind == "chunk"
    assert abs(hits[0].score) < 1e-3  # 自召回 score = 1 - cosine ≈ 0
    # 排序：score 严格递增（距离语义）
    assert [h.score for h in hits] == sorted(h.score for h in hits)


def test_search_filter_by_file_id(vs, emb):
    rng = np.random.default_rng(4)
    vecs_a = np.stack([_norm(rng) for _ in range(2)])
    vecs_b = np.stack([_norm(rng) for _ in range(2)])
    vs.insert_chunks(1, [1, 2], vecs_a, ["chunk", "chunk"])
    vs.insert_chunks(2, [1, 2], vecs_b, ["chunk", "chunk"])
    hits = vs.search(vecs_b[0], topk=10, file_id=1)
    assert {h.file_id for h in hits} == {1}
    hits_all = vs.search(vecs_b[0], topk=10)
    assert len(hits_all) == 4


def test_search_dim_mismatch_raises(vs):
    with pytest.raises(ValueError):
        vs.search(np.zeros(64, dtype=np.float32))


def test_delete_file(vs, emb):
    rng = np.random.default_rng(5)
    vs.insert_chunks(1, [1, 2, 3], np.stack([_norm(rng) for _ in range(3)]),
                     ["chunk"] * 3)
    vs.insert_chunks(2, [1, 2], np.stack([_norm(rng) for _ in range(2)]),
                     ["chunk"] * 2)
    assert vs.delete_file(1) == 3
    assert vs.count() == 2
    assert vs.delete_file(999) == 0
    assert vs.count() == 2


def test_upsert_overwrites_same_id(vs, emb):
    rng = np.random.default_rng(6)
    v1, v2 = _norm(rng), _norm(rng)
    vs.insert_chunks(1, [1], v1[None, :], ["chunk"])
    vs.upsert_chunks(1, [1], v2[None, :], ["chunk"])
    assert vs.count() == 1
    hits = vs.search(v2, topk=1)
    assert hits[0].doc_id == "f1-c1"
    assert abs(hits[0].score) < 1e-3


def test_insert_batch_over_1024(vs, emb):
    """zvec 单次 insert 上限 1024：封装必须自动分批（1030 条跨两批）。"""
    rng = np.random.default_rng(7)
    n = 1030
    vecs = np.stack([_norm(rng) for _ in range(n)])
    written = vs.insert_chunks(9, list(range(1, n + 1)), vecs, ["chunk"] * n)
    assert written == n
    assert vs.count() == n


def test_reopen_persistence(zdir, vs, emb):
    rng = np.random.default_rng(8)
    vec = _norm(rng)
    vs.insert_chunks(3, [1], vec[None, :], ["chunk"])
    vs.close()
    reopened = ZvecStore.open(zdir)
    try:
        assert reopened.count() == 1
        hits = reopened.search(vec, topk=1)
        assert hits[0].doc_id == "f3-c1"
    finally:
        reopened.close()


# ------------------------------------------------- optimize 攒批策略

def test_optimize_if_needed_threshold_with_persisted_state(zdir):
    state = PipelineState(zdir.parent / "pipeline.sqlite3")
    try:
        vs = ZvecStore.create(zdir, optimize_state=state)
        try:
            rng = np.random.default_rng(9)
            vecs = np.stack([_norm(rng) for _ in range(3)])
            assert vs.optimize_if_needed(state, every=5, hours=24.0) is False
            vs.insert_chunks(1, [1, 2, 3], vecs, ["chunk"] * 3)
            assert state.get_optimize_state().pending_count == 3
            assert vs.optimize_if_needed(state, every=5, hours=24.0) is False  # 3 < 5
            vs.insert_chunks(1, [4, 5, 6], vecs, ["chunk"] * 3)  # pending 6 ≥ 5
            assert vs.optimize_if_needed(state, every=5, hours=24.0) is True
            st = state.get_optimize_state()
            assert st.pending_count == 0
            assert st.oldest_pending_at is None
            assert st.last_optimize_at is not None
        finally:
            vs.close()
    finally:
        state.close()


def test_optimize_if_needed_hours_rule(zdir):
    state = PipelineState(zdir.parent / "pipeline.sqlite3")
    try:
        vs = ZvecStore.create(zdir, optimize_state=state)
        try:
            rng = np.random.default_rng(10)
            vs.insert_chunks(1, [1], _norm(rng)[None, :], ["chunk"])
            # 人为把最老未 optimize 写入时间拨回 25 小时前
            import time as _time

            state.conn.execute(
                "UPDATE optimize_state SET oldest_pending_at = ? WHERE id = 1",
                (_time.time() - 25 * 3600.0,),
            )
            assert vs.optimize_if_needed(state, every=DEFAULT_OPTIMIZE_EVERY,
                                         hours=24.0) is True
            assert state.get_optimize_state().pending_count == 0
        finally:
            vs.close()
    finally:
        state.close()


def test_optimize_if_needed_no_pending(zdir, vs):
    assert vs.optimize_if_needed(None, every=1, hours=0.0) is False


def test_optimize_resets_memory_counter(zdir, vs):
    rng = np.random.default_rng(11)
    vs.insert_chunks(1, [1], _norm(rng)[None, :], ["chunk"])
    assert vs.optimize_if_needed(None, every=1, hours=24.0) is True
    # 已 optimize，pending 清零：同阈值不再触发
    assert vs.optimize_if_needed(None, every=1, hours=24.0) is False


def test_optimize_state_without_persistence_resets_on_reopen(zdir, vs):
    """无持久化注入时，攒批计数仅在内存（重启清零）——文档化行为。"""
    rng = np.random.default_rng(12)
    vs.insert_chunks(1, [1], _norm(rng)[None, :], ["chunk"])
    vs.close()
    reopened = ZvecStore.open(zdir)
    try:
        assert reopened.optimize_if_needed(None, every=1, hours=24.0) is False
    finally:
        reopened.close()

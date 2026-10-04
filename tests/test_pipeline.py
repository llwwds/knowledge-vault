"""pipeline 测试：Store（登记+FTS）+ ZvecStore（向量）+ PipelineState + FakeEmbedder。

覆盖：状态机 / 断点续传跳过 / 修改重建 / 删除与移动语义 / 50MB 阈值 /
summary 记录 / 批量扫描对账 / 失败重试与上限 / optimize 联动。
全部合成数据 + tmp_path，不触碰真实 vault/state。
"""

from __future__ import annotations

import os

import pytest

from knowledge_vault.chunker import ChunkerParams
from knowledge_vault.embedder import FakeEmbedder
from knowledge_vault.pipeline import (
    LARGE_FILE_THRESHOLD_BYTES,
    IngestParams,
    IngestPipeline,
    PipelineState,
    PipelineStateError,
)
from knowledge_vault.registry import compute_content_hash
from knowledge_vault.vectorstore import ZvecStore


@pytest.fixture()
def state(store):
    s = PipelineState(store.config.state_dir / "pipeline.sqlite3")
    yield s
    s.close()


@pytest.fixture()
def vectors(store):
    v = ZvecStore.create(store.config.state_dir / "zvec_chunks")
    yield v
    v.close()


@pytest.fixture()
def params():
    # 小阈值便于测试：>64 字节即"超大文件"；切块参数缩小让多块场景可行
    return IngestParams(
        large_file_threshold=64,
        max_retries=2,
        chunker=ChunkerParams(target_size=48, overlap=8, min_chunk_size=4),
    )


@pytest.fixture()
def pipe(store, vectors, state, params):
    return IngestPipeline(
        store, vectors, state, embedder=FakeEmbedder(), params=params
    )


MD_BODY = (
    "# 甲\n\n检索算法 alpha。\n\n"
    "## 乙\n\n索引结构 beta。\n"
)
# 注意：上方 params 夹具把 large_file_threshold 压到 64 字节，
# MD_BODY 必须保持 < 64 UTF-8 字节（当前 59），否则会走"超大文件只登记"路径。


# ------------------------------------------------------- 状态机与 Happy Path

def test_process_new_md_indexed(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "note.md"
    path.write_text(MD_BODY, encoding="utf-8")
    result = pipe.process_file(path)
    assert result == "indexed"

    task = state.get_task(str(path))
    assert task is not None
    assert task.state == "indexed"
    assert task.content_hash == compute_content_hash(path)
    assert task.file_id is not None
    assert task.pending_at and task.parsed_at and task.embedded_at and task.indexed_at
    assert task.failed_at is None

    # 登记层：活跃记录存在
    doc = store.get_file(task.file_id)
    assert doc is not None and doc.deleted_at is None

    # FTS：正文可全文检索
    hits = store.search("检索算法")
    assert hits and hits[0].file_id == task.file_id

    # 向量：与 FTS chunk 数一致
    fts_count = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (task.file_id,)
    ).fetchone()[0]
    assert fts_count >= 2
    assert vectors.count() == fts_count


def test_process_unchanged_skips(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "same.md"
    path.write_text(MD_BODY, encoding="utf-8")
    assert pipe.process_file(path) == "indexed"
    count_before = vectors.count()
    task_before = state.get_task(str(path))
    assert pipe.process_file(path) == "skipped"
    assert vectors.count() == count_before
    # 任务行保持原状（updated_at 等不变）
    assert state.get_task(str(path)) == task_before


def test_process_modified_reindexes_same_file_id(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "grow.md"
    path.write_text("# 头\n\n旧内容 old_token_zz。\n", encoding="utf-8")
    assert pipe.process_file(path) == "indexed"
    task = state.get_task(str(path))
    old_file_id = task.file_id

    path.write_text("# 头\n\n新内容 new_token_yy，完全不同的正文。\n", encoding="utf-8")
    assert pipe.process_file(path) == "indexed"

    new_task = state.get_task(str(path))
    assert new_task.file_id == old_file_id  # 修改不换 file_id
    assert new_task.content_hash == compute_content_hash(path)

    # 旧文本从 FTS 与向量两侧消失，新文本就位
    assert store.search("old_token_zz") == []
    assert any("new_token_yy" in h.text for h in store.search("new_token_yy"))
    doc = store.get_file(old_file_id)
    assert doc.content_hash == compute_content_hash(path)  # 登记行已刷新
    fts_count = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (old_file_id,)
    ).fetchone()[0]
    assert vectors.count() == fts_count


# ------------------------------------------------------- 登记口径与阈值

def test_unknown_extension_registered_only(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "pic.png"
    path.write_bytes(b"\x89PNG fake bytes")
    assert pipe.process_file(path) == "indexed"
    task = state.get_task(str(path))
    doc = store.get_file(task.file_id)
    assert doc is not None
    assert store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (task.file_id,)
    ).fetchone()[0] == 0
    assert vectors.count() == 0


def test_large_file_registered_only(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "big.md"
    path.write_text("# 大文件\n\n" + "内容 " * 100, encoding="utf-8")  # > 64B 阈值
    assert pipe.process_file(path) == "indexed"
    task = state.get_task(str(path))
    assert store.get_file(task.file_id).size_bytes > 64
    assert store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (task.file_id,)
    ).fetchone()[0] == 0
    assert vectors.count() == 0


def test_default_threshold_is_50mb():
    assert LARGE_FILE_THRESHOLD_BYTES == 50 * 1024 * 1024


def test_summary_record_seq0(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "with_summary.md"
    path.write_text("# 正文\n\n正文内容 body_token。\n", encoding="utf-8")
    # 先带 summary 登记（summary 真源在登记行，由用户/API 填写）
    doc = store.register_file(path, summary="这是文件级摘要 summary_token。")
    assert pipe.process_file(path) == "indexed"

    hits = store.search("summary_token")
    assert hits and hits[0].kind == "summary"
    assert hits[0].chunk_seq == 0
    # 向量侧也有 summary 记录
    vhits = vectors.search(
        pipe.embedder.encode(["这是文件级摘要 summary_token。"])[0],
        topk=10, file_id=doc.file_id,
    )
    assert any(h.kind == "summary" and h.chunk_seq == 0 for h in vhits)
    # 正文 chunk 从 seq=1 起
    seqs = [r["chunk_seq"] for r in store.conn.execute(
        "SELECT chunk_seq FROM chunks WHERE file_id = ? ORDER BY chunk_seq",
        (doc.file_id,),
    )]
    assert seqs[0] == 0 and seqs[1] == 1


def test_txt_file_paragraph_chunks(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("纯文本段落一。\n\n纯文本段落二。\n", encoding="utf-8")
    assert pipe.process_file(path) == "indexed"
    task = state.get_task(str(path))
    hits = store.search("纯文本段落一")
    assert hits and hits[0].file_id == task.file_id
    assert vectors.count() >= 1


# ------------------------------------------------------- 删除与移动语义

def test_delete_semantics(pipe, store, vectors, state, tmp_path):
    path = tmp_path / "doomed.md"
    path.write_text(MD_BODY, encoding="utf-8")
    assert pipe.process_file(path) == "indexed"
    task = state.get_task(str(path))
    file_id = task.file_id
    assert vectors.count() > 0

    os.remove(path)
    assert pipe.handle_deleted(path) == "deleted"
    doc = store.get_file(file_id)
    assert doc.deleted_at is not None           # 登记 soft_delete
    assert store.conn.execute(                   # chunks 清空
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (file_id,)
    ).fetchone()[0] == 0
    assert vectors.count() == 0                  # 向量清空
    assert state.get_task(str(path)) is None     # task 行清除


def test_handle_deleted_unknown_path_is_noop(pipe, state, tmp_path):
    assert pipe.handle_deleted(tmp_path / "ghost.md") == "noop"


def test_move_semantics_new_file_id(pipe, store, vectors, state, tmp_path):
    old_path = tmp_path / "origin.md"
    old_path.write_text(MD_BODY, encoding="utf-8")
    assert pipe.process_file(old_path) == "indexed"
    old_task = state.get_task(str(old_path))
    old_file_id = old_task.file_id
    old_vec_count = vectors.count()

    new_path = tmp_path / "renamed.md"
    os.rename(old_path, new_path)
    new_file_id = pipe.handle_moved(old_path, new_path)
    assert new_file_id is not None and new_file_id != old_file_id

    # 旧记录软删、新记录活跃
    assert store.get_file(old_file_id).deleted_at is not None
    new_doc = store.get_file(new_file_id)
    assert new_doc.deleted_at is None
    assert new_doc.file_path == str(new_path)
    assert new_doc.content_hash == old_doc_hash(old_file_id, store)  # hash 继承

    # FTS 与向量都重建在新 file_id 下
    assert store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (old_file_id,)
    ).fetchone()[0] == 0
    new_fts = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (new_file_id,)
    ).fetchone()[0]
    assert new_fts > 0
    assert vectors.count() == new_fts == old_vec_count
    hits = store.search("检索算法")
    assert hits and hits[0].file_id == new_file_id
    assert state.get_task(str(old_path)) is None
    assert state.get_task(str(new_path)).state == "indexed"


def old_doc_hash(file_id, store):
    return store.get_file(file_id).content_hash


def test_handle_moved_unknown_old_registers_new(pipe, store, state, tmp_path):
    new_path = tmp_path / "fresh.md"
    new_path.write_text("# 全新\n\n全新内容。\n", encoding="utf-8")
    new_file_id = pipe.handle_moved(tmp_path / "unknown.md", new_path)
    assert new_file_id is not None
    assert store.get_file(new_file_id).file_path == str(new_path)


# ------------------------------------------------------- 批量扫描与对账

def test_process_all_reconciles(pipe, store, vectors, state):
    root = store.config.vault_root
    f1 = root / "one.md"
    f2 = root / "sub" / "two.md"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("# 一\n\n内容一。\n", encoding="utf-8")
    f2.write_text("# 二\n\n内容二。\n", encoding="utf-8")

    report1 = pipe.process_all()
    assert report1.indexed == 2 and report1.skipped == 0

    # 第二轮：无变化 → 全跳过
    report2 = pipe.process_all()
    assert report2.indexed == 0 and report2.skipped == 2

    # 删一个、增一个
    os.remove(f2)
    f3 = root / "three.md"
    f3.write_text("# 三\n\n内容三。\n", encoding="utf-8")
    report3 = pipe.process_all()
    assert report3.indexed == 1       # three.md
    assert report3.skipped == 1       # one.md
    assert report3.deleted == 1       # two.md 软删
    assert store.get_file(
        state.get_task(str(f1)).file_id
    ).deleted_at is None


def test_process_all_skips_hidden(pipe, store, state):
    root = store.config.vault_root
    hidden_dir = root / ".obsidian"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "config.md").write_text("# 隐藏目录\n", encoding="utf-8")
    (root / ".DS_Store").write_bytes(b"\x00\x01")
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "x.md").write_text("# 缓存\n", encoding="utf-8")
    real = root / "real.md"
    real.write_text("# 真文件\n\n真内容。\n", encoding="utf-8")

    report = pipe.process_all()
    assert report.indexed == 1
    rows = store.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert rows == 1


def test_process_all_limit(pipe, store):
    root = store.config.vault_root
    root.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        p = root / f"f{i}.md"
        p.write_text(f"# {i}\n\n内容 {i}。\n", encoding="utf-8")
    report = pipe.process_all(limit=2)
    assert report.indexed == 2


# ------------------------------------------------------- 失败与重试

class ExplodingEmbedder:
    """固定抛错的 Embedder（协议鸭子类型，测失败路径）。"""

    dim = 1024

    def encode(self, texts):
        raise RuntimeError("模型炸了")


def test_failure_marks_failed_with_retry_count(pipe, state, tmp_path):
    path = tmp_path / "boom.md"
    path.write_text(MD_BODY, encoding="utf-8")
    pipe.embedder = ExplodingEmbedder()
    assert pipe.process_file(path) == "failed"
    task = state.get_task(str(path))
    assert task.state == "failed"
    assert task.retries == 1
    assert "模型炸了" in task.error
    assert task.failed_at is not None
    # 半途状态不残留：FTS/向量都应干净（删除先行）
    assert store_or_vector_clean(pipe, task)


def store_or_vector_clean(pipe, task):
    fts = pipe.store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (task.file_id,)
    ).fetchone()[0]
    vecs = pipe.vectors.count()
    return fts == 0 and vecs == 0


def test_retry_failed_recovers(pipe, state, tmp_path):
    path = tmp_path / "recover.md"
    path.write_text(MD_BODY, encoding="utf-8")
    pipe.embedder = ExplodingEmbedder()
    assert pipe.process_file(path) == "failed"

    pipe.embedder = FakeEmbedder()
    report = pipe.retry_failed()
    assert report.indexed == 1
    assert state.get_task(str(path)).state == "indexed"
    # 成功后 retries 保留失败历史
    assert state.get_task(str(path)).retries == 1


def test_retry_failed_respects_cap(pipe, state, tmp_path):
    path = tmp_path / "capped.md"
    path.write_text(MD_BODY, encoding="utf-8")
    pipe.embedder = ExplodingEmbedder()
    assert pipe.process_file(path) == "failed"          # retries=1
    report1 = pipe.retry_failed()                        # retries=2（上限 2）
    assert report1.failed == 1
    assert state.get_task(str(path)).retries == 2
    report2 = pipe.retry_failed()                        # 达上限，不再重排
    assert report2.indexed == 0 and report2.failed == 0
    assert state.get_task(str(path)).state == "failed"


def test_retry_failed_cleans_missing_files(pipe, store, state, tmp_path):
    path = tmp_path / "gone.md"
    path.write_text(MD_BODY, encoding="utf-8")
    pipe.embedder = ExplodingEmbedder()
    assert pipe.process_file(path) == "failed"
    os.remove(path)
    report = pipe.retry_failed()
    assert report.deleted == 1
    assert state.get_task(str(path)) is None


def test_pending_listing(pipe, state, tmp_path):
    a = tmp_path / "a.md"
    a.write_text("# A\n\n内容 A。\n", encoding="utf-8")
    assert pipe.pending() == []  # 从未处理过 → 无任务行
    pipe.state.ensure_task(str(a))
    assert [t.file_path for t in pipe.pending()] == [str(a)]


# ------------------------------------------------------- optimize 联动

def test_maybe_optimize_triggered_via_process_all(store, vectors, state):
    params = IngestParams(optimize_every=1, optimize_hours=24.0)
    pipe = IngestPipeline(store, vectors, state,
                          embedder=FakeEmbedder(), params=params)
    root = store.config.vault_root
    root.mkdir(parents=True, exist_ok=True)
    p = root / "opt.md"
    p.write_text("# 优化\n\n触发攒批。\n", encoding="utf-8")
    report = pipe.process_all()
    assert report.indexed == 1
    st = state.get_optimize_state()
    assert st.pending_count == 0
    assert st.last_optimize_at is not None
    # 生命周期归夹具管理（vectors/state 夹具负责 close），不在此重复关闭


# ------------------------------------------------------- 参数与环境

def test_ingest_params_from_env():
    params = IngestParams.from_env(
        {"KV_OPTIMIZE_EVERY": "123", "KV_OPTIMIZE_HOURS": "1.5"}
    )
    assert params.optimize_every == 123
    assert params.optimize_hours == 1.5
    defaults = IngestParams.from_env({})
    assert defaults.optimize_every == 50_000
    assert defaults.optimize_hours == 24.0


def test_pipeline_state_schema_rejects_newer(tmp_path):
    db = tmp_path / "p.sqlite3"
    st = PipelineState(db)
    st.conn.execute("UPDATE pipeline_schema_version SET version = 99")
    st.close()
    with pytest.raises(PipelineStateError):
        PipelineState(db)


def test_task_roundtrip_fields(state, tmp_path):
    task = state.ensure_task("/some/path.md")
    assert task.state == "pending"
    updated = state.set_task_state(
        "/some/path.md", "parsed", content_hash="abc123", file_id=42
    )
    assert updated.state == "parsed"
    assert updated.content_hash == "abc123"
    assert updated.file_id == 42
    assert updated.parsed_at is not None
    assert state.delete_task("/some/path.md") is True
    assert state.delete_task("/some/path.md") is False

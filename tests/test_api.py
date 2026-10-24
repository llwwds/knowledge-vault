"""API 壳测试：起线程打 /health /search /files /optimize（stdlib urllib）。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pytest

from knowledge_vault import api
from knowledge_vault.embedder import FakeEmbedder

from _phase3_helpers import FakeRerankerByLength, build_kb


@pytest.fixture()
def kb(store, make_file) -> dict:
    return build_kb(store, make_file)


@pytest.fixture()
def api_factory(config, monkeypatch):
    """起线程的 API 服务器工厂；用完自动 shutdown。"""

    created: list = []

    def _factory(reranker=None, embedder=None) -> str:
        writable_root = Path(config.state_dir).parent / "writable-notes"
        writable_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("KV_WRITABLE_ROOT", str(writable_root))
        server = api.make_server(
            config, host="127.0.0.1", port=0, reranker=reranker,
            embedder=embedder,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        created.append((server, thread))
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    yield _factory
    for server, thread in created:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def _post(
    url: str, payload: dict | bytes, *, headers: dict | None = None
) -> tuple[int, dict]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def _method(
    url: str, method: str, payload: dict | None = None, *, headers: dict | None = None
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(
        url, data=data, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


class TestHealth:
    def test_health(self, api_factory):
        base = api_factory()
        status, body = _get(f"{base}/health")
        assert status == 200
        assert body["ok"] is True
        assert body["service"] == "knowledge-vault"
        assert body["version"]

    def test_health_trailing_slash_and_query(self, api_factory):
        base = api_factory()
        status, _ = _get(f"{base}/health/")
        assert status == 200

    def test_unknown_path_404(self, api_factory):
        base = api_factory()
        status, body = _get(f"{base}/nope")
        assert status == 404
        assert "error" in body

    def test_wrong_method_405(self, api_factory):
        base = api_factory()
        status, _ = _post(f"{base}/health", {})
        assert status == 405
        status, _ = _get(f"{base}/search")
        assert status == 405


class TestSearchEndpoint:
    def test_search_basic(self, api_factory, kb):
        base = api_factory()
        status, body = _post(f"{base}/search", {"query": "机器学习"})
        assert status == 200
        assert body["query"] == "机器学习"
        items = body["items"]
        assert items, "应至少命中一个 chunk"
        item = items[0]
        for key in ("file_id", "chunk_id", "chunk_seq", "kind", "score", "sources", "text"):
            assert key in item
        assert any("fts" in it["sources"] for it in items)
        assert body["meta"]["reranked"] is False
        # 软删除文件不出现在结果里
        assert all(it["file_id"] != kb["d"] for it in items)

    def test_search_filters_status(self, api_factory, kb):
        base = api_factory()
        status, body = _post(
            f"{base}/search", {"query": "机器学习", "filters": {"status": "now"}}
        )
        assert status == 200
        assert {it["file_id"] for it in body["items"]} <= {kb["a"]}

    def test_search_filters_tags(self, api_factory, kb):
        base = api_factory()
        status, body = _post(
            f"{base}/search", {"query": "机器学习", "filters": {"tags": ["ml"]}}
        )
        assert status == 200
        assert all(it["file_id"] != kb["b"] for it in body["items"])

    def test_search_filters_file_ids(self, api_factory, kb):
        base = api_factory()
        status, body = _post(
            f"{base}/search",
            {"query": "机器学习", "filters": {"file_ids": [kb["c"]]}},
        )
        assert status == 200
        assert {it["file_id"] for it in body["items"]} <= {kb["c"]}

    def test_search_top_j(self, api_factory, kb):
        base = api_factory()
        status, body = _post(f"{base}/search", {"query": "机器学习", "top_j": 2})
        assert status == 200
        assert len(body["items"]) <= 2

    def test_search_with_server_reranker(self, api_factory, kb):
        base = api_factory(reranker=FakeRerankerByLength())
        status, body = _post(f"{base}/search", {"query": "机器学习"})
        assert status == 200
        assert body["meta"]["reranked"] is True

    def test_search_no_rerank_flag(self, api_factory, kb):
        base = api_factory(reranker=FakeRerankerByLength())
        status, body = _post(
            f"{base}/search", {"query": "机器学习", "no_rerank": True}
        )
        assert status == 200
        assert body["meta"]["reranked"] is False

    def test_search_validation_errors(self, api_factory, kb):
        base = api_factory()
        # 非法 JSON
        status, body = _post(f"{base}/search", b"not-json")
        assert status == 400
        # 空 query
        status, _ = _post(f"{base}/search", {"query": "  "})
        assert status == 400
        status, _ = _post(f"{base}/search", {})
        assert status == 400
        # 非法 top_j
        status, _ = _post(f"{base}/search", {"query": "q", "top_j": 0})
        assert status == 400
        status, _ = _post(f"{base}/search", {"query": "q", "top_j": "3"})
        assert status == 400
        # 非法 filters
        status, _ = _post(f"{base}/search", {"query": "q", "filters": {"tags": "ml"}})
        assert status == 400
        status, _ = _post(
            f"{base}/search", {"query": "q", "filters": {"file_ids": ["1"]}}
        )
        assert status == 400

    def test_search_concurrent_requests(self, api_factory, kb):
        # 每请求独立连接的并发读：8 线程 × 4 次
        base = api_factory()
        errors: list[str] = []

        def worker():
            for _ in range(4):
                try:
                    status, body = _post(f"{base}/search", {"query": "机器学习"})
                    if status != 200 or not body["items"]:
                        errors.append(f"status={status}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert errors == []


class TestFilesEndpoint:
    def test_get_file(self, api_factory, kb):
        base = api_factory()
        status, body = _get(f"{base}/files/{kb['a']}")
        assert status == 200
        assert body["file_id"] == kb["a"]
        assert body["file_path"].endswith("机器学习笔记.md")
        assert body["context_tag"] == ["ml", "ai"]
        assert body["status"] == "now"

    def test_get_legacy_alias_is_read_only(self, api_factory, kb, store):
        store.conn.execute(
            """CREATE TABLE legacy_id_alias (
                namespace TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                legacy_file_id TEXT NOT NULL, relative_path TEXT NOT NULL,
                full_sha256 TEXT NOT NULL, new_file_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        store.conn.execute(
            "INSERT INTO legacy_id_alias(namespace,snapshot_id,legacy_file_id,relative_path,full_sha256,new_file_id) "
            "VALUES('test','snapshot','kvf_test_123','notes/example.md','abc',?)",
            (kb["a"],),
        )
        store.conn.commit()
        base = api_factory()
        status, body = _get(f"{base}/files/kvf_test_123")
        assert status == 200
        assert body["file_id"] == kb["a"]
        assert body["legacy_file_ids"] == ["kvf_test_123"]

        status, body = _get(f"{base}/documents/kvf_test_123")
        assert status == 200
        assert body["read_only"] is True
        assert "content" not in body
        write_status, _ = _method(
            f"{base}/documents/kvf_test_123", "PUT", {"content": "must not update"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert write_status == 400

    def test_get_file_deleted_record_returned_as_is(self, api_factory, kb):
        base = api_factory()
        status, body = _get(f"{base}/files/{kb['d']}")
        assert status == 200
        assert body["deleted_at"] is not None

    def test_get_file_missing(self, api_factory, kb):
        base = api_factory()
        status, body = _get(f"{base}/files/999999")
        assert status == 404

    def test_get_file_bad_id(self, api_factory, kb):
        base = api_factory()
        status, _ = _get(f"{base}/files/abc")
        assert status == 400


class TestControlledDocumentCrud:
    def test_add_update_read_and_soft_delete(self, api_factory, config, monkeypatch):
        token = "test-kv-write-token"
        monkeypatch.setenv("KV_WRITE_TOKEN", token)
        base = api_factory(embedder=FakeEmbedder())
        auth = {"Authorization": f"Bearer {token}"}
        status, created = _post(
            f"{base}/documents",
            {"title": "CRUD验收笔记", "content": "# CRUD验收笔记\n\n初始内容：琥珀星图。"},
            headers=auth,
        )
        assert status == 201, created
        file_id = created["file_id"]
        source = Path(created["file_path"])
        assert source.is_file()
        for field in ("file_id:", "title:", "context_tag: []", "created_at:", "modified_at:", "summary: null", "status: library"):
            assert field in created["content"]
        assert f"file_id: {file_id}" in created["content"]

        status, read_back = _get(f"{base}/documents/{file_id}")
        assert status == 200
        assert "琥珀星图" in read_back["content"]

        status, updated = _method(
            f"{base}/documents/{file_id}", "PUT",
            {"content": "# CRUD验收笔记\n\n更新内容：蓝色彗星。"}, headers=auth,
        )
        assert status == 200, updated
        assert updated["file_id"] == file_id
        assert "蓝色彗星" in updated["content"]
        assert source.read_text(encoding="utf-8") == updated["content"]

        status, result = _post(f"{base}/search", {"query": "蓝色彗星"})
        assert status == 200
        assert any(item["file_id"] == file_id for item in result["items"])

        status, deleted = _method(f"{base}/documents/{file_id}", "DELETE", headers=auth)
        assert status == 200
        assert deleted["soft_deleted"] is True
        assert deleted["deleted_at"] is not None
        assert deleted["source_exists"] is True
        assert source.is_file(), "软删除必须保留笔记源文件"
        status, result = _post(f"{base}/search", {"query": "蓝色彗星"})
        assert status == 200
        assert all(item["file_id"] != file_id for item in result["items"])

    def test_mutations_require_bearer_token(self, api_factory, monkeypatch):
        monkeypatch.setenv("KV_WRITE_TOKEN", "test-kv-write-token")
        base = api_factory(embedder=FakeEmbedder())
        status, body = _post(
            f"{base}/documents", {"title": "无凭据", "content": "不得创建"}
        )
        assert status == 401, body

    def test_legacy_document_cannot_be_modified(self, api_factory, kb, monkeypatch):
        token = "test-kv-write-token"
        monkeypatch.setenv("KV_WRITE_TOKEN", token)
        base = api_factory(embedder=FakeEmbedder())
        legacy_id = kb["a"]
        status, legacy = _get(f"{base}/files/{legacy_id}")
        assert status == 200
        source = Path(legacy["file_path"])
        original = source.read_bytes()
        status, body = _method(
            f"{base}/documents/{legacy_id}", "PUT",
            {"content": "# 禁止修改 legacy 文件"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert status == 400, body
        assert source.read_bytes() == original

    def test_failed_update_restores_source_and_reindexes_old_content(
        self, api_factory, monkeypatch
    ):
        token = "test-kv-write-token"
        monkeypatch.setenv("KV_WRITE_TOKEN", token)
        base = api_factory(embedder=FakeEmbedder())
        auth = {"Authorization": f"Bearer {token}"}
        original = "# 回滚验收笔记\n\n原始内容：银色山谷。"
        status, created = _post(
            f"{base}/documents",
            {"title": "回滚验收笔记", "content": original},
            headers=auth,
        )
        assert status == 201, created
        file_id = created["file_id"]
        source = Path(created["file_path"])
        original_note = created["content"]

        process_file = api.IngestPipeline.process_file
        calls = 0

        def fail_once(pipeline, path):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected index failure")
            return process_file(pipeline, path)

        monkeypatch.setattr(api.IngestPipeline, "process_file", fail_once)
        status, body = _method(
            f"{base}/documents/{file_id}", "PUT",
            {"content": "# 回滚验收笔记\n\n新内容：琥珀海湾。"}, headers=auth,
        )
        assert status == 503, body
        assert body["source_restored"] is True
        assert body["rollback_ingest"] == "indexed"
        assert source.read_text(encoding="utf-8") == original_note
        status, read_back = _get(f"{base}/documents/{file_id}")
        assert status == 200
        assert read_back["content"] == original_note


class TestOptimizeEndpoint:
    def test_optimize(self, api_factory, kb):
        base = api_factory()
        status, body = _post(f"{base}/optimize", {})
        assert status == 200
        assert body["fts_optimized"] is True
        # tmp 状态目录没有向量库 → 跳过
        assert body["vector"] is None

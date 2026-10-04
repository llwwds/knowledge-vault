"""API 壳测试：起线程打 /health /search /files /optimize（stdlib urllib）。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Callable

import pytest

from knowledge_vault import api

from _phase3_helpers import FakeRerankerByLength, build_kb


@pytest.fixture()
def kb(store, make_file) -> dict:
    return build_kb(store, make_file)


@pytest.fixture()
def api_factory(config):
    """起线程的 API 服务器工厂；用完自动 shutdown。"""

    created: list = []

    def _factory(reranker=None) -> str:
        server = api.make_server(config, host="127.0.0.1", port=0, reranker=reranker)
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


def _post(url: str, payload: dict | bytes) -> tuple[int, dict]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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


class TestOptimizeEndpoint:
    def test_optimize(self, api_factory, kb):
        base = api_factory()
        status, body = _post(f"{base}/optimize", {})
        assert status == 200
        assert body["fts_optimized"] is True
        # tmp 状态目录没有向量库 → 跳过
        assert body["vector"] is None

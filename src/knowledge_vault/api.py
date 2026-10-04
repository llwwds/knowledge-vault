"""HTTP API 壳（stdlib ``http.server``，无第三方依赖）。

路由（读 API + optimize 触发；写入只经摄入管线，本壳不提供写入端点）：

- ``GET  /health``          → 存活 / 版本
- ``POST /search``          → 召回。body::

        {"query": "...", "top_j": 10,
         "filters": {"status": "library", "tags": ["ml"], "file_ids": [1, 2]},
         "no_rerank": false}

  ``top_j`` / ``filters`` / ``no_rerank`` 均可省略；``no_rerank=true`` 时即使
  服务端配置了 reranker 也跳过重排。
- ``GET  /files/{file_id}`` → 登记记录原样返回（含软删除记录，``deleted_at``
  非空即软删除；调用方自行判断）。
- ``POST /optimize``        → 触发 FTS/SQLite/zvec 索引优化，返回摘要。

并发模型：``ThreadingHTTPServer`` + **每请求独立 SQLite 连接**（WAL 允许多读
并发，连接不跨线程复用）；写操作（optimize）由进程内互斥锁串行化——单线程化
写保护。端口取 ``KV_API_PORT``（默认 8770）。
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zvec
from zvec import HnswQueryParam

from . import __version__
from .config import VaultConfig, load_config
from .embedder import BGEM3Embedder
from .protocols import Reranker
from .registry import get_file
from .retrieve import RetrievalOutput, ZvecVectorIndex, retrieve
from .rerank import FlagRerankerImpl
from .store import Store

__all__ = [
    "DEFAULT_API_PORT",
    "make_server",
    "serve",
    "open_vector_path",
    "run_optimize",
    "collect_stats",
]


def open_vector_path(config: VaultConfig) -> tuple[ZvecVectorIndex | None, "zvec.Collection | None"]:
    """向量路自动发现：``state/zvec_chunks`` 存在时打开原生 collection。

    摄入管线（HNSW m=16/efc=200，spike① 定稿）建的 collection 用
    ``HnswQueryParam(ef=300)`` 查询；目录不存在（纯 FTS 库）返回 (None, None)，
    召回自动降级为全文+图。返回的 collection 句柄由调用方负责 ``close()``
    （HTTP/CLI 均按每请求开关使用，与每请求独立 SQLite 连接同构）。
    """
    zvec_dir = config.state_dir / "zvec_chunks"
    if not (zvec_dir / "meta.json").is_file():
        return None, None
    coll = zvec.open(str(zvec_dir))
    return ZvecVectorIndex(coll, query_param=HnswQueryParam(ef=300)), coll

DEFAULT_API_PORT = 8770
ENV_API_PORT = "KV_API_PORT"

#: optimize 等写操作的进程内互斥锁（单线程化写保护）
_WRITE_LOCK = threading.Lock()


# ------------------------------------------------------------------ 存储访问


def _open_store(config: VaultConfig) -> Store:
    """每请求打开独立连接。

    init=True：init_db 幂等（CREATE TABLE IF NOT EXISTS + schema_version
    校验/迁移），空库自动就绪；库已初始化时每请求开销仅一遍幂等 DDL 与
    版本检查，单机 headless 场景可接受。
    """
    return Store(config=config, init=True)


# ------------------------------------------------------------------ 优化/统计


def run_optimize(config: VaultConfig, *, include_vector: bool = True) -> dict:
    """触发索引优化：FTS 合并 + SQLite 优化/checkpoint + 可选 zvec optimize。

    向量库目录（``KV_STATE_DIR/zvec_chunks``）由摄入管线创建；不存在时跳过。
    返回摘要 dict（JSON 可序列化）。调用方负责写保护（API 侧有锁，CLI 单发）。
    """
    summary: dict = {"fts_optimized": False, "vector": None}
    store = _open_store(config)
    try:
        conn = store.conn
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        conn.execute("PRAGMA optimize")
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        summary["fts_optimized"] = True
        summary["sqlite"] = {
            "wal_checkpoint": {
                "busy": int(checkpoint[0]) if checkpoint else None,
                "wal_frames": int(checkpoint[1]) if checkpoint else None,
                "checkpointed": int(checkpoint[2]) if checkpoint else None,
            }
        }
    finally:
        store.close()

    vec_dir = Path(config.state_dir) / "zvec_chunks"
    if include_vector and vec_dir.is_dir():
        import zvec

        coll = zvec.open(str(vec_dir))
        try:
            try:
                coll.optimize()
                optimized = True
            except Exception as exc:  # zvec 行为差异：optimize 失败不阻塞返回
                optimized = False
                summary["vector_error"] = str(exc)
            entry: dict = {"path": str(vec_dir), "optimized": optimized}
            try:
                stats = coll.stats()
                entry["stats"] = stats if isinstance(stats, dict) else str(stats)
            except Exception as exc:
                summary.setdefault("vector_error", str(exc))
            summary["vector"] = entry
        finally:
            coll.close()
    return summary


def collect_stats(config: VaultConfig) -> dict:
    """收集只读运行统计（documents/chunks/edges/FTS/向量库目录）。"""
    store = _open_store(config)
    try:
        conn = store.conn

        def one(sql: str, params: tuple = ()) -> int:
            return int(conn.execute(sql, params).fetchone()[0])

        docs = {
            "total": one("SELECT COUNT(*) FROM documents"),
            "active": one("SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL"),
            "deleted": one(
                "SELECT COUNT(*) FROM documents WHERE deleted_at IS NOT NULL"
            ),
            "by_status": {
                row["status"]: int(row["n"])
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM documents "
                    "WHERE deleted_at IS NULL GROUP BY status"
                )
            },
            "is_large": one(
                "SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL AND is_large = 1"
            ),
        }
        chunks = {
            "total": one("SELECT COUNT(*) FROM chunks"),
            "by_kind": {
                row["kind"]: int(row["n"])
                for row in conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind"
                )
            },
        }
        edges_total = one("SELECT COUNT(*) FROM edges")
        db_path = Path(config.db_path)
        db_size = db_path.stat().st_size if db_path.exists() else 0
    finally:
        store.close()

    vec_dir = Path(config.state_dir) / "zvec_chunks"
    return {
        "version": __version__,
        "state_dir": str(config.state_dir),
        "db_size_bytes": db_size,
        "documents": docs,
        "chunks": chunks,
        "edges": edges_total,
        "vector": {"present": vec_dir.is_dir(), "path": str(vec_dir)},
    }


# ------------------------------------------------------------------ HTTP 壳


#: 各路径允许的 HTTP 方法（其余方法 → 405）
_ALLOWED_METHODS = {
    "/health": {"GET"},
    "/search": {"POST"},
    "/optimize": {"POST"},
}


def _method_allowed(path: str, method: str) -> bool:
    allowed = _ALLOWED_METHODS.get(path)
    if allowed is None and path.startswith("/files/"):
        allowed = {"GET"}
    return allowed is None or method in allowed


class _ApiHandler(BaseHTTPRequestHandler):
    """请求处理器；``config`` / ``reranker`` 由 :func:`make_server` 注入类属性。"""

    config: VaultConfig
    reranker: Reranker | None

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # 静默默认 stderr 访问日志

    # ---- 响应工具

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("请求体为空，需要 JSON 对象")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    # ---- 路由

    def do_GET(self) -> None:  # noqa: N802 (http.server 命名约定)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if not _method_allowed(path, "GET"):
                self._send_json(405, {"error": f"{path} 不支持 GET"})
            elif path == "/health":
                self._send_json(
                    200, {"ok": True, "service": "knowledge-vault", "version": __version__}
                )
            elif path.startswith("/files/"):
                self._handle_get_file(path)
            else:
                self._send_json(404, {"error": f"未知路径: {path}"})
        except Exception as exc:  # 防御：任何 handler 异常都回 500 而非断连
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if not _method_allowed(path, "POST"):
                self._send_json(405, {"error": f"{path} 不支持 POST"})
            elif path == "/search":
                self._handle_search()
            elif path == "/optimize":
                with _WRITE_LOCK:
                    self._send_json(200, run_optimize(self.config))
            else:
                self._send_json(404, {"error": f"未知路径: {path}"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_get_file(self, path: str) -> None:
        raw = path[len("/files/"):]
        try:
            file_id = int(raw)
        except ValueError:
            self._send_json(400, {"error": f"file_id 必须是整数，收到 {raw!r}"})
            return
        store = _open_store(self.config)
        try:
            doc = get_file(store.conn, file_id)
        finally:
            store.close()
        if doc is None:
            self._send_json(404, {"error": f"file_id 不存在: {file_id}"})
        else:
            self._send_json(200, doc.as_dict())

    def _handle_search(self) -> None:
        body = self._read_json_body()
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            self._send_json(400, {"error": "query 必须是非空字符串"})
            return
        top_j = body.get("top_j", 10)
        if not isinstance(top_j, int) or isinstance(top_j, bool) or top_j <= 0:
            self._send_json(400, {"error": "top_j 必须是正整数"})
            return
        filters = body.get("filters") or {}
        if not isinstance(filters, dict):
            self._send_json(400, {"error": "filters 必须是对象"})
            return
        status = filters.get("status")
        tags = filters.get("tags")
        file_ids = filters.get("file_ids")
        if tags is not None and (
            not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
        ):
            self._send_json(400, {"error": "filters.tags 必须是字符串数组"})
            return
        if file_ids is not None and (
            not isinstance(file_ids, list)
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in file_ids)
        ):
            self._send_json(400, {"error": "filters.file_ids 必须是整数数组"})
            return
        no_rerank = bool(body.get("no_rerank", False))
        reranker = None if no_rerank else self.reranker

        vector_path, coll = open_vector_path(self.config)
        store = _open_store(self.config)
        try:
            output: RetrievalOutput = retrieve(
                store,
                query,
                embedder=self.embedder,
                vector_path=vector_path,
                reranker=reranker,
                status=status,
                context_tag=tags,
                file_ids=file_ids,
                top_j=top_j,
            )
        finally:
            store.close()
            if coll is not None:
                coll.close()
        self._send_json(200, output.as_dict())


def make_server(
    config: VaultConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    reranker: Reranker | None = None,
    embedder: "BGEM3Embedder | None" = None,
) -> ThreadingHTTPServer:
    """构造已绑定（未 serve）的 API 服务器。

    ``port=0`` 由系统分配临时端口（测试用）；生产用 :func:`serve`。
    ``reranker`` / ``embedder`` 缺省为生产实现（``FlagRerankerImpl`` /
    ``BGEM3Embedder``，均惰性加载，构造不触发模型加载；测试注入替身即可）。
    向量路每请求经 :func:`open_vector_path` 自动发现。
    """
    handler = type("BoundApiHandler", (_ApiHandler,), {})
    handler.config = config
    handler.reranker = reranker
    handler.embedder = embedder if embedder is not None else BGEM3Embedder()
    return ThreadingHTTPServer((host, port), handler)


def serve(
    config: VaultConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    reranker: Reranker | None = None,
) -> None:
    """阻塞式启动 API 服务（``kv-serve`` 入口）。

    ``port=None`` 时取 ``KV_API_PORT``（默认 8770）；``reranker`` 缺省为
    ``FlagRerankerImpl()``（惰性加载，首次查询才加载模型）。
    """
    config = config or load_config()
    if port is None:
        port = int(os.environ.get(ENV_API_PORT, DEFAULT_API_PORT))
    server = make_server(
        config, host=host, port=port, reranker=reranker or FlagRerankerImpl()
    )
    host_bound, port_bound = server.server_address[:2]
    print(f"knowledge-vault api listening on {host_bound}:{port_bound}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

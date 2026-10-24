"""HTTP API 壳（stdlib ``http.server``，无第三方依赖）。

路由（只读查询 + 受控写 API；所有新增/修改均经摄入管线）：

- ``GET  /health``          → 存活 / 版本
- ``POST /search``          → 召回。body::

        {"query": "...", "top_j": 10,
         "filters": {"status": "library", "tags": ["ml"], "file_ids": [1, 2]},
         "no_rerank": false}

  ``top_j`` / ``filters`` / ``no_rerank`` 均可省略；``no_rerank=true`` 时即使
  服务端配置了 reranker 也跳过重排。
- ``GET  /files/{file_id}`` / ``GET /files/{legacy_id}`` → 登记记录原样返回；
  legacy string ID 只解析到精确 path+SHA256 审计生成的 integer alias（含软删除记录，``deleted_at``
  非空即软删除；调用方自行判断）。
- ``GET  /files?limit=20``  → 最近登记列表（file_id 倒序，含软删除记录，
  带 ``deleted_at`` 供调用方判断；``limit`` 默认 20，上限 500）。
- ``POST /documents``       → 在 ``KV_WRITABLE_ROOT`` 新建笔记并经摄入管线登记/索引；
  body 为 ``{"title":"...", "content":"..."}``。
- ``GET  /documents/{id}``  → 读取 API 管理笔记的登记字段与内容；历史 ID
  仅返回只读登记数据，不授予旧笔记写权限。
- ``PUT  /documents/{id}``  → 更新受控笔记正文并重建索引；body 为
  ``{"content":"..."}``。
- ``DELETE /documents/{id}`` → 调用摄入管线 ``handle_deleted`` 软删除；保留源文件。
- ``POST /optimize``        → 触发 FTS/SQLite/zvec 索引优化，返回摘要。
- ``GET  /stats``           → 只读运行统计（:func:`collect_stats` 原样 JSON）。
- ``GET  /ui``              → 单页 web 看板（``?theme=<name>`` 指定皮肤，
  未知名字回退默认皮肤 ``xai-dark``；皮肤系统见 :mod:`knowledge_vault.webui`）。
- ``GET  /ui/theme.css``    → 皮肤 CSS（``?name=<name>``；未知名字回退默认）。

并发模型：``ThreadingHTTPServer`` + **每请求独立 SQLite 连接**（WAL 允许多读
并发，连接不跨线程复用）；写操作（optimize）由进程内互斥锁串行化——单线程化
写保护。端口取 ``KV_API_PORT``（默认 8770）。
"""

from __future__ import annotations

import json
import hmac
import os
import re
import threading
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zvec
from zvec import HnswQueryParam

from . import __version__, webui
from .config import VaultConfig, load_config
from .embedder import BGEM3Embedder, OllamaEmbedder
from .protocols import Reranker
from .registry import get_file
from .retrieve import RetrievalOutput, ZvecVectorIndex, retrieve
from .rerank import FlagRerankerImpl
from .store import Store
from .pipeline import IngestPipeline, PipelineState
from .vectorstore import ZvecStore

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
ENV_EMBEDDER = "KV_EMBEDDER"
ENV_FTS_ONLY = "KV_FTS_ONLY"

#: optimize 等写操作的进程内互斥锁（单线程化写保护）
_WRITE_LOCK = threading.Lock()


class _NoVectorIndex:
    """No-op adapter for FTS-only mode; it must never create fake vectors."""

    def attach_optimize_state(self, _state) -> None:
        return None

    def delete_file(self, _file_id: int) -> None:
        return None

    def upsert_chunks(self, _file_id: int, _seqs, _vectors, _kinds) -> None:
        return None

    def optimize_if_needed(self, _state, *, every: int, hours: float) -> bool:
        return False

    def optimize(self) -> None:
        return None

    def close(self) -> None:
        return None


def _fts_only_enabled() -> bool:
    return os.environ.get(ENV_FTS_ONLY, "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _frontmatter(title: str, file_id: int | str, timestamp: str) -> str:
    """Render the seven required Obsidian metadata fields in their stable order."""
    return (
        "---\n"
        f"file_id: {file_id}\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        "context_tag: []\n"
        f"created_at: {json.dumps(timestamp)}\n"
        f"modified_at: {json.dumps(timestamp)}\n"
        "summary: null\n"
        "status: library\n"
        "---\n"
    )


def _update_note_body_and_modified_at(raw: bytes, file_id: int, body: str) -> bytes:
    """Preserve all frontmatter except modified_at, then replace only Markdown body."""
    text = raw.decode("utf-8")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n?", text, re.DOTALL)
    if match is None:
        raise ValueError("受控笔记缺少规范 frontmatter，拒绝覆盖")
    header = match.group(1)
    id_match = re.search(r"(?m)^file_id:\s*(\d+)\s*$", header)
    modified_match = re.search(r"(?m)^modified_at:\s*.*$", header)
    if id_match is None or int(id_match.group(1)) != file_id:
        raise ValueError("笔记 frontmatter 的 file_id 与登记记录不一致")
    if modified_match is None:
        raise ValueError("笔记 frontmatter 缺少 modified_at，拒绝覆盖")
    updated_header = (
        header[: modified_match.start()]
        + f"modified_at: {json.dumps(_utc_now())}"
        + header[modified_match.end() :]
    )
    rendered = f"---\n{updated_header}\n---\n{body}"
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered.encode("utf-8")


# ------------------------------------------------------------------ 存储访问


def _open_store(config: VaultConfig) -> Store:
    """每请求打开独立连接。

    init=True：init_db 幂等（CREATE TABLE IF NOT EXISTS + schema_version
    校验/迁移），空库自动就绪；库已初始化时每请求开销仅一遍幂等 DDL 与
    版本检查，单机 headless 场景可接受。
    """
    return Store(config=config, init=True)


def _legacy_aliases(conn, file_ids: list[int]) -> dict[int, list[str]]:
    """Return read-only historical IDs for runtime integer IDs, if alias state exists."""
    if not file_ids:
        return {}
    aliases: dict[int, list[str]] = {}
    placeholders = ",".join("?" for _ in file_ids)
    try:
        rows = conn.execute(
            "SELECT new_file_id, legacy_file_id FROM legacy_id_alias "
            f"WHERE new_file_id IN ({placeholders}) ORDER BY legacy_file_id",
            file_ids,
        )
    except Exception as exc:
        # The alias table is created only by the resumable legacy restore script.
        # Fresh deployments remain fully usable before that restore starts.
        if "no such table" in str(exc).lower():
            return {}
        raise
    for row in rows:
        aliases.setdefault(int(row["new_file_id"]), []).append(str(row["legacy_file_id"]))
    return aliases


def _resolve_legacy_alias(conn, legacy_id: str) -> int | None:
    try:
        row = conn.execute(
            "SELECT new_file_id FROM legacy_id_alias WHERE legacy_file_id=? "
            "ORDER BY namespace,snapshot_id LIMIT 1",
            (legacy_id,),
        ).fetchone()
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    return int(row[0]) if row else None


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


#: ``GET /files``（列表）单次返回的登记记录数上限
_FILES_LIST_MAX = 500

#: 各路径允许的 HTTP 方法（其余方法 → 405）
_ALLOWED_METHODS = {
    "/health": {"GET"},
    "/search": {"POST"},
    "/optimize": {"POST"},
    "/stats": {"GET"},
    "/ui": {"GET"},
    "/ui/theme.css": {"GET"},
    "/files": {"GET"},
    "/documents": {"POST"},
}


def _method_allowed(path: str, method: str) -> bool:
    allowed = _ALLOWED_METHODS.get(path)
    if allowed is None and path.startswith("/files/"):
        allowed = {"GET"}
    if allowed is None and path.startswith("/documents/"):
        allowed = {"GET", "PUT", "DELETE"}
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

    def _send_text(self, status: int, text: str, content_type: str) -> None:
        """发送 UTF-8 文本响应（``GET /ui`` 的 HTML 与 ``/ui/theme.css`` 用）。"""
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict[str, list[str]]:
        """解析 URL 查询串为 ``{key: [values]}``（无查询参数时空 dict）。"""
        return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("请求体为空，需要 JSON 对象")
        if length > 2_100_000:
            raise ValueError("JSON 请求体不能超过 2.1MB")
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
            elif path == "/ui":
                requested_theme = (self._query().get("theme") or [None])[0]
                self._send_text(
                    200,
                    webui.render_index(webui.resolve_theme(requested_theme)),
                    "text/html; charset=utf-8",
                )
            elif path == "/ui/theme.css":
                name = (self._query().get("name") or [None])[0]
                self._send_text(
                    200, webui.render_theme_css(name), "text/css; charset=utf-8"
                )
            elif path == "/stats":
                self._send_json(200, collect_stats(self.config))
            elif path.startswith("/files/"):
                self._handle_get_file(path)
            elif path == "/files":
                self._handle_list_files()
            elif path.startswith("/documents/"):
                self._handle_get_document(path)
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
            elif path == "/documents":
                if not self._authorized_write():
                    return
                self._handle_create_document()
            else:
                self._send_json(404, {"error": f"未知路径: {path}"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_PUT(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if not path.startswith("/documents/"):
                self._send_json(404, {"error": f"未知路径: {path}"})
                return
            if not self._authorized_write():
                return
            if not _method_allowed(path, "PUT"):
                self._send_json(405, {"error": f"{path} 不支持 PUT"})
            else:
                self._handle_update_document(path)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if not path.startswith("/documents/"):
                self._send_json(404, {"error": f"未知路径: {path}"})
                return
            if not self._authorized_write():
                return
            if not _method_allowed(path, "DELETE"):
                self._send_json(405, {"error": f"{path} 不支持 DELETE"})
            else:
                self._handle_delete_document(path)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _writable_root(self) -> Path:
        raw = os.environ.get("KV_WRITABLE_ROOT")
        if not raw:
            raise ValueError("写入 API 未配置 KV_WRITABLE_ROOT")
        root = Path(raw).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _authorized_write(self) -> bool:
        expected = os.environ.get("KV_WRITE_TOKEN", "")
        scheme, _, supplied = self.headers.get("Authorization", "").partition(" ")
        if not expected or scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
            self._send_json(401, {"error": "需要有效的 Bearer 写入凭据"})
            return False
        return True

    def _controlled_path(self, raw_path: str) -> Path:
        """拒绝 symlink/越界；HTTP 写入只允许独立的可控笔记目录。"""
        root = self._writable_root()
        candidate = Path(raw_path)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("该记录不属于 API 管理的可写笔记目录") from exc
        if resolved == root or candidate.is_symlink():
            raise ValueError("不允许操作目录或符号链接")
        return resolved

    def _read_document_body(self, *, require_title: bool) -> tuple[dict, str, str]:
        body = self._read_json_body()
        title = body.get("title")
        content = body.get("content")
        if require_title and (not isinstance(title, str) or not title.strip()):
            raise ValueError("title 必须是非空字符串")
        if not require_title and title is not None:
            raise ValueError("更新只接受 content；标题由登记记录管理")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须是非空字符串")
        if len(content.encode("utf-8")) > 2_000_000:
            raise ValueError("单条笔记正文不能超过 2MB")
        safe_title = " ".join(title.split()).replace("/", "-").replace("\\", "-") if title else ""
        if len(safe_title) > 120:
            raise ValueError("title 不能超过 120 个字符")
        return body, safe_title, content

    def _new_pipeline(self):
        """构造本次写操作的 pipeline；调用方需在 finally 中关闭全部句柄。"""
        store = _open_store(self.config)
        state = PipelineState(self.config.state_dir / "pipeline.sqlite3")
        if _fts_only_enabled():
            vectors = _NoVectorIndex()
            pipeline_embedder = None
        else:
            vec_dir = self.config.state_dir / "zvec_chunks"
            if (vec_dir / "meta.json").is_file():
                vectors = ZvecStore.open(vec_dir, optimize_state=state)
            else:
                vectors = ZvecStore.create(vec_dir, optimize_state=state)
            pipeline_embedder = self.embedder
        return IngestPipeline(store, vectors, state, embedder=pipeline_embedder), store, vectors, state

    @staticmethod
    def _close_pipeline(handles) -> None:
        _, store, vectors, state = handles
        vectors.close()
        state.close()
        store.close()

    def _managed_doc(self, file_id: int):
        store = _open_store(self.config)
        try:
            doc = get_file(store.conn, file_id)
        finally:
            store.close()
        if doc is None:
            return None, None
        path = self._controlled_path(doc.file_path)
        return doc, path

    def _handle_create_document(self) -> None:
        _, title, content = self._read_document_body(require_title=True)
        root = self._writable_root()
        path = self._controlled_path(str(root / f"{uuid.uuid4().hex}-{title}.md"))
        created_at = _utc_now()
        with _WRITE_LOCK:
            # Register first to obtain the integer runtime ID, then write a
            # complete required Obsidian frontmatter and ingest that final file.
            path.write_text(_frontmatter(title, "pending", created_at) + content, encoding="utf-8")
            os.chmod(path, 0o600)
            store = _open_store(self.config)
            try:
                doc = store.register_file(
                    path,
                    title=title,
                    context_tag=[],
                    status="library",
                    created_at=created_at,
                )
            finally:
                store.close()
            path.write_text(_frontmatter(title, doc.file_id, created_at) + content, encoding="utf-8")
            os.chmod(path, 0o600)
            handles = self._new_pipeline()
            try:
                result = handles[0].process_file(path)
            finally:
                self._close_pipeline(handles)
        if result not in {"indexed", "skipped"}:
            self._send_json(503, {"error": f"摄入失败: {result}", "file_id": doc.file_id})
            return
        current_doc, _ = self._managed_doc(doc.file_id)
        self._send_json(201, {**current_doc.as_dict(), "content": path.read_text(encoding="utf-8"), "ingest": result})

    def _parse_document_id(self, path: str) -> int:
        raw = path[len("/documents/"):]
        try:
            file_id = int(raw)
        except ValueError as exc:
            raise ValueError(f"file_id 必须是整数，收到 {raw!r}") from exc
        if file_id <= 0:
            raise ValueError("file_id 必须是正整数")
        return file_id

    def _handle_get_document(self, path: str) -> None:
        raw_id = urllib.parse.unquote(path[len("/documents/"):])
        if not raw_id.isdecimal():
            store = _open_store(self.config)
            try:
                file_id = _resolve_legacy_alias(store.conn, raw_id)
                if file_id is None:
                    self._send_json(404, {"error": "legacy file_id 没有可用的精确别名映射"})
                    return
                doc = get_file(store.conn, file_id)
                aliases = _legacy_aliases(store.conn, [file_id]).get(file_id, [])
            finally:
                store.close()
            if doc is None:
                self._send_json(404, {"error": "legacy alias 指向的登记记录不存在"})
                return
            self._send_json(200, {**doc.as_dict(), "legacy_file_ids": aliases, "read_only": True})
            return
        file_id = self._parse_document_id(path)
        doc, note_path = self._managed_doc(file_id)
        if doc is None:
            self._send_json(404, {"error": f"file_id 不存在: {file_id}"})
            return
        content = note_path.read_text(encoding="utf-8") if note_path.exists() else None
        self._send_json(200, {**doc.as_dict(), "content": content, "read_only": False})

    def _handle_update_document(self, path: str) -> None:
        file_id = self._parse_document_id(path)
        _, _, content = self._read_document_body(require_title=False)
        doc, note_path = self._managed_doc(file_id)
        if doc is None:
            self._send_json(404, {"error": f"file_id 不存在: {file_id}"})
            return
        if doc.deleted_at is not None:
            self._send_json(409, {"error": "该记录已软删除"})
            return
        if not note_path.is_file():
            self._send_json(409, {"error": "受控笔记源文件缺失，拒绝重建或覆盖"})
            return
        old_content = note_path.read_bytes()
        try:
            updated_content = _update_note_body_and_modified_at(old_content, file_id, content)
        except ValueError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        ingest_error = None
        result = "failed"
        with _WRITE_LOCK:
            tmp = note_path.with_name(f".{note_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_bytes(updated_content)
                os.chmod(tmp, 0o600)
                os.replace(tmp, note_path)
                try:
                    handles = self._new_pipeline()
                    try:
                        result = handles[0].process_file(note_path)
                    finally:
                        self._close_pipeline(handles)
                except Exception as exc:  # 恢复原文并重建原索引
                    ingest_error = type(exc).__name__
            finally:
                if tmp.exists():
                    tmp.unlink()
        if result not in {"indexed", "skipped"}:
            rollback_error = None
            try:
                rollback_tmp = note_path.with_name(f".{note_path.name}.{uuid.uuid4().hex}.rollback")
                rollback_tmp.write_bytes(old_content)
                os.chmod(rollback_tmp, 0o600)
                os.replace(rollback_tmp, note_path)
                with _WRITE_LOCK:
                    handles = self._new_pipeline()
                    try:
                        rollback_result = handles[0].process_file(note_path)
                    finally:
                        self._close_pipeline(handles)
            except Exception as exc:  # noqa: BLE001
                rollback_error = type(exc).__name__
                rollback_result = "failed"
            self._send_json(503, {
                "error": f"摄入失败: {result}",
                "file_id": file_id,
                "source_restored": rollback_error is None,
                "rollback_ingest": rollback_result,
                "failure_type": ingest_error,
                "rollback_failure_type": rollback_error,
            })
            return
        updated, _ = self._managed_doc(file_id)
        self._send_json(200, {**updated.as_dict(), "content": note_path.read_text(encoding="utf-8"), "ingest": result})

    def _handle_delete_document(self, path: str) -> None:
        file_id = self._parse_document_id(path)
        doc, note_path = self._managed_doc(file_id)
        if doc is None:
            self._send_json(404, {"error": f"file_id 不存在: {file_id}"})
            return
        if doc.deleted_at is not None:
            self._send_json(200, {**doc.as_dict(), "soft_deleted": True, "source_exists": note_path.exists()})
            return
        with _WRITE_LOCK:
            handles = self._new_pipeline()
            try:
                result = handles[0].handle_deleted(note_path)
            finally:
                self._close_pipeline(handles)
        updated, _ = self._managed_doc(file_id)
        self._send_json(
            200,
            {**updated.as_dict(), "soft_deleted": result == "deleted", "source_exists": note_path.exists()},
        )

    def _handle_get_file(self, path: str) -> None:
        raw = urllib.parse.unquote(path[len("/files/"):])
        store = _open_store(self.config)
        try:
            if raw.isdecimal():
                file_id = int(raw)
            else:
                file_id = _resolve_legacy_alias(store.conn, raw)
                if file_id is None:
                    self._send_json(404, {"error": "legacy file_id 没有可用的精确别名映射"})
                    return
            doc = get_file(store.conn, file_id)
            aliases = _legacy_aliases(store.conn, [file_id]).get(file_id, [])
        finally:
            store.close()
        if doc is None:
            self._send_json(404, {"error": f"登记记录不存在: {raw}"})
        else:
            self._send_json(200, {**doc.as_dict(), "legacy_file_ids": aliases})

    def _handle_list_files(self) -> None:
        """``GET /files?limit=N``：最近登记列表（file_id 倒序，含软删除记录）。"""
        raw = (self._query().get("limit") or ["20"])[0]
        try:
            limit = int(raw)
        except ValueError:
            self._send_json(400, {"error": f"limit 必须是整数，收到 {raw!r}"})
            return
        if limit <= 0:
            self._send_json(400, {"error": "limit 必须是正整数"})
            return
        limit = min(limit, _FILES_LIST_MAX)
        store = _open_store(self.config)
        try:
            docs = list(store.iterate_files())  # 含软删除，deleted_at 供调用方判断
        finally:
            store.close()
        docs.reverse()  # iterate_files 按 file_id 升序 → 倒序即最近登记在前
        files = [
            {
                "file_id": doc.file_id,
                "title": doc.title,
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "size_bytes": doc.size_bytes,
                "status": doc.status,
                "mtime": doc.mtime,
                "deleted_at": doc.deleted_at,
            }
            for doc in docs[:limit]
        ]
        self._send_json(200, {"count": len(files), "files": files})

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
            payload = output.as_dict()
            aliases = _legacy_aliases(store.conn, [item.file_id for item in output.items])
            for item in payload["items"]:
                item["legacy_file_ids"] = aliases.get(item["file_id"], [])
        finally:
            store.close()
            if coll is not None:
                coll.close()
        self._send_json(200, payload)


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
    if embedder is not None:
        handler.embedder = embedder
    elif _fts_only_enabled():
        handler.embedder = None
    elif os.environ.get(ENV_EMBEDDER, "").lower() == "ollama":
        handler.embedder = OllamaEmbedder(
            model_name=os.environ.get("KV_OLLAMA_EMBED_MODEL", "bge-m3"),
            api_base=os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
        )
    else:
        handler.embedder = BGEM3Embedder()
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

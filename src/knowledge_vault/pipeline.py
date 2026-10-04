"""摄入管线：pipeline.db 状态机 + 切块/embedding/双写编排。

组成（设计定稿）：
- **pipeline.db**（独立 SQLite，``KV_STATE_DIR`` 下，默认 ``pipeline.sqlite3``）：
  ``tasks`` 表状态机 ``pending → parsed → embedded → indexed``（任意步失败
  → ``failed``，带 ``retries`` 计数、``error`` 与每状态时间戳）；
  ``optimize_state`` 表持久化 zvec optimize 攒批阈值状态（见 vectorstore）。
- **断点续传**：任务已 ``indexed`` 且文件当前 ``content_hash`` 未变 → 跳过；
  其余情况一律"先清旧（chunks FTS + 向量）再重写"，天然幂等。
- **文件变更语义**：
  - 修改（content_hash 变化）＝ 删旧 chunks + 重嵌重插；file_id 不变，
    登记行的 content_hash/size_bytes/mtime/updated_at 由本模块刷新
    （见 :meth:`IngestPipeline._refresh_registered` 的下沉备注）；
  - 删除 ＝ 登记 soft_delete + 删 chunks + 删向量 + 清 task 行；
  - 移动 ＝ ``Store.move_file``（**产生新 file_id**）+ 向量与 FTS 按新
    file_id 全量重建。
- **50MB 阈值**：文件本体 > 50MB 不切块不进向量库，只登记（+ summary 非空
  则 summary 记录）。注意与登记层 ``is_large``（registry 自带 100MB 默认阈值）
  是两个独立口径：前者管切块，后者是登记字段。
- **kind=summary**：登记行 ``summary`` 非空的文件（含非 md 小文件与超大文件）
  追加一条 seq=0 的 summary 记录（FTS 走 ``Store.add_summary_chunk``，
  向量 kind='summary'）。summary 真源在登记行（由用户/API 填写），本管线
  不解析 frontmatter、不自动生成。
- 切块范围（映射规则 Q8）：``.md/.markdown`` 按标题层级；``.txt`` 段落聚合；
  其他扩展名只登记不切块。
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunker import (
    MARKDOWN_EXTENSIONS,
    TEXT_EXTENSIONS,
    ChunkerParams,
    chunk_markdown,
    chunk_plain,
)
from .embedder import Embedder
from .registry import Document, UnknownFileIdError, compute_content_hash, utc_iso
from .store import Store
from .vectorstore import (
    DEFAULT_OPTIMIZE_EVERY,
    DEFAULT_OPTIMIZE_HOURS,
    OptimizeState,
    ZvecStore,
)

__all__ = [
    "IngestPipeline",
    "IngestParams",
    "PipelineState",
    "ProcessReport",
    "Task",
    "LARGE_FILE_THRESHOLD_BYTES",
    "TASK_STATES",
    "ENV_OPTIMIZE_EVERY",
    "ENV_OPTIMIZE_HOURS",
]

#: 管线级 optimize 阈值环境变量（KV_OPTIMIZE_EVERY 条 / KV_OPTIMIZE_HOURS 小时）
ENV_OPTIMIZE_EVERY = "KV_OPTIMIZE_EVERY"
ENV_OPTIMIZE_HOURS = "KV_OPTIMIZE_HOURS"

#: 任务状态机（pending → parsed → embedded → indexed；failed 旁路）
TASK_STATES = ("pending", "parsed", "embedded", "indexed", "failed")

#: 50MB 阈值：超过则文件本体不切块不进向量库（只登记 + summary 非空则 summary）
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024

#: pipeline.db 默认文件名 / zvec collection 默认目录名（均在 KV_STATE_DIR 下）
DEFAULT_PIPELINE_DB_FILENAME = "pipeline.sqlite3"
DEFAULT_ZVEC_DIRNAME = "zvec_chunks"

_DEFAULT_MAX_RETRIES = 3

# pipeline.db 独立 schema 版本（与登记层 schema_version 互不相干）
_PIPELINE_SCHEMA_VERSION = 1

_UNSET = object()


# ------------------------------------------------------------------- 状态机

@dataclass(frozen=True)
class Task:
    """pipeline.db ``tasks`` 表一行。"""

    file_path: str
    state: str
    retries: int
    error: str | None
    content_hash: str | None
    file_id: int | None
    created_at: str
    updated_at: str
    pending_at: str | None
    parsed_at: str | None
    embedded_at: str | None
    indexed_at: str | None
    failed_at: str | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


_TASK_COLUMNS = (
    "file_path", "state", "retries", "error", "content_hash", "file_id",
    "created_at", "updated_at", "pending_at", "parsed_at", "embedded_at",
    "indexed_at", "failed_at",
)

_STATE_TS_COLUMN = {
    "pending": "pending_at",
    "parsed": "parsed_at",
    "embedded": "embedded_at",
    "indexed": "indexed_at",
    "failed": "failed_at",
}

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS tasks (
    file_path    TEXT PRIMARY KEY,
    state        TEXT NOT NULL DEFAULT 'pending'
                 CHECK (state IN ('pending', 'parsed', 'embedded', 'indexed', 'failed')),
    retries      INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    content_hash TEXT,
    file_id      INTEGER,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    pending_at   TEXT,
    parsed_at    TEXT,
    embedded_at  TEXT,
    indexed_at   TEXT,
    failed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);

CREATE TABLE IF NOT EXISTS optimize_state (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    pending_count    INTEGER NOT NULL DEFAULT 0,
    oldest_pending_at REAL,
    last_optimize_at REAL
);
INSERT OR IGNORE INTO optimize_state(id, pending_count) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS pipeline_schema_version (version INTEGER NOT NULL);
"""


class PipelineStateError(RuntimeError):
    """pipeline.db 的 schema 版本比当前代码新，拒绝打开。"""


class PipelineState:
    """pipeline.db 句柄：任务状态机 + optimize 攒批状态（持久化层，无业务逻辑）。

    同时实现 :class:`knowledge_vault.vectorstore.OptimizeStateStore` 协议，
    可直接注入 :class:`ZvecStore`。
    """

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA_SQL)
        row = self.conn.execute(
            "SELECT version FROM pipeline_schema_version"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO pipeline_schema_version(version) VALUES (?)",
                (_PIPELINE_SCHEMA_VERSION,),
            )
        elif int(row[0]) > _PIPELINE_SCHEMA_VERSION:
            raise PipelineStateError(
                f"pipeline.db schema v{row[0]} 比代码 v{_PIPELINE_SCHEMA_VERSION} 新"
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PipelineState":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------- tasks

    @staticmethod
    def _to_task(row: sqlite3.Row) -> Task:
        return Task(
            file_path=row["file_path"],
            state=row["state"],
            retries=int(row["retries"]),
            error=row["error"],
            content_hash=row["content_hash"],
            file_id=row["file_id"] if row["file_id"] is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            pending_at=row["pending_at"],
            parsed_at=row["parsed_at"],
            embedded_at=row["embedded_at"],
            indexed_at=row["indexed_at"],
            failed_at=row["failed_at"],
        )

    def ensure_task(self, file_path: str | Path) -> Task:
        """任务行不存在则建（state=pending）；存在则原样返回。"""
        path_str = str(file_path)
        now = utc_iso()
        self.conn.execute(
            "INSERT OR IGNORE INTO tasks(file_path, state, created_at, updated_at, pending_at) "
            "VALUES (?, 'pending', ?, ?, ?)",
            (path_str, now, now, now),
        )
        return self.get_task(path_str)  # type: ignore[return-value]

    def get_task(self, file_path: str | Path) -> Task | None:
        row = self.conn.execute(
            f"SELECT {', '.join(_TASK_COLUMNS)} FROM tasks WHERE file_path = ?",
            (str(file_path),),
        ).fetchone()
        return self._to_task(row) if row is not None else None

    def set_task_state(
        self,
        file_path: str | Path,
        state: str,
        *,
        error: str | None | object = _UNSET,
        content_hash: str | None | object = _UNSET,
        file_id: int | None | object = _UNSET,
        retries: int | object = _UNSET,
    ) -> Task:
        """推进任务状态（写对应状态时间戳；缺省字段保持不变）。

        ``error``/``content_hash``/``file_id``/``retries`` 传 :data:`_UNSET`
        （模块私有哨兵）表示"不改"；显式 ``None`` 表示"清空/置空"。
        """
        if state not in TASK_STATES:
            raise ValueError(f"state 必须是 {TASK_STATES} 之一，收到 {state!r}")
        path_str = str(file_path)
        self.ensure_task(path_str)
        now = utc_iso()
        assignments = ["state = ?", "updated_at = ?", f"{_STATE_TS_COLUMN[state]} = ?"]
        params: list[object] = [state, now, now]
        if error is not _UNSET:
            assignments.append("error = ?")
            params.append(error)
        if content_hash is not _UNSET:
            assignments.append("content_hash = ?")
            params.append(content_hash)
        if file_id is not _UNSET:
            assignments.append("file_id = ?")
            params.append(file_id)
        if retries is not _UNSET:
            assignments.append("retries = ?")
            params.append(int(retries))  # type: ignore[arg-type]
        params.append(path_str)
        self.conn.execute(
            f"UPDATE tasks SET {', '.join(assignments)} WHERE file_path = ?", params
        )
        return self.get_task(path_str)  # type: ignore[return-value]

    def mark_failed(
        self, file_path: str | Path, *, error: str, retries: int | None = None
    ) -> Task:
        """置 failed（retries 缺省在当前值上 +1）。"""
        path_str = str(file_path)
        if retries is None:
            task = self.get_task(path_str)
            retries = (task.retries if task else 0) + 1
        return self.set_task_state(
            path_str, "failed", error=error, retries=retries
        )

    def increment_retries(self, file_path: str | Path) -> Task:
        task = self.get_task(file_path)
        return self.set_task_state(
            file_path, task.state if task else "pending",
            retries=(task.retries if task else 0) + 1,
        )

    def delete_task(self, file_path: str | Path) -> bool:
        cur = self.conn.execute(
            "DELETE FROM tasks WHERE file_path = ?", (str(file_path),)
        )
        return cur.rowcount > 0

    def list_tasks(self, *, state: str | None = None) -> list[Task]:
        sql = f"SELECT {', '.join(_TASK_COLUMNS)} FROM tasks"
        params: list[object] = []
        if state is not None:
            if state not in TASK_STATES:
                raise ValueError(f"state 必须是 {TASK_STATES} 之一，收到 {state!r}")
            sql += " WHERE state = ?"
            params.append(state)
        sql += " ORDER BY file_path"
        return [self._to_task(row) for row in self.conn.execute(sql, params)]

    # ------------------------------------------- optimize 攒批状态（协议实现）

    def get_optimize_state(self) -> OptimizeState:
        row = self.conn.execute(
            "SELECT pending_count, oldest_pending_at, last_optimize_at "
            "FROM optimize_state WHERE id = 1"
        ).fetchone()
        if row is None:  # pragma: no cover - init 时 INSERT OR IGNORE 保证存在
            return OptimizeState(pending_count=0, oldest_pending_at=None,
                                 last_optimize_at=None)
        return OptimizeState(
            pending_count=int(row["pending_count"]),
            oldest_pending_at=row["oldest_pending_at"],
            last_optimize_at=row["last_optimize_at"],
        )

    def add_pending_optimize(self, n: int, *, now: float | None = None) -> None:
        if n <= 0:
            return
        now = time.time() if now is None else now
        self.conn.execute(
            "UPDATE optimize_state SET "
            "pending_count = pending_count + ?, "
            "oldest_pending_at = CASE "
            "  WHEN pending_count = 0 OR oldest_pending_at IS NULL THEN ? "
            "  ELSE oldest_pending_at END "
            "WHERE id = 1",
            (int(n), now),
        )

    def mark_optimized(self, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.conn.execute(
            "UPDATE optimize_state SET pending_count = 0, oldest_pending_at = NULL, "
            "last_optimize_at = ? WHERE id = 1",
            (now,),
        )


# --------------------------------------------------------------------- 参数

@dataclass(frozen=True)
class IngestParams:
    """摄入管线参数。

    - ``optimize_every`` / ``optimize_hours``：optimize 攒批阈值，
      默认取 ``KV_OPTIMIZE_EVERY``（50000）/ ``KV_OPTIMIZE_HOURS``（24h）。
    - ``large_file_threshold``：50MB 阈值（设计定稿；与登记层 is_large 口径独立）。
    - ``max_retries``：失败自动重试上限（超过不再自动重排，人工裁决）。
    - ``chunker``：切块参数（默认值靠后网格实验调参）。
    """

    large_file_threshold: int = LARGE_FILE_THRESHOLD_BYTES
    max_retries: int = _DEFAULT_MAX_RETRIES
    optimize_every: int = DEFAULT_OPTIMIZE_EVERY
    optimize_hours: float = DEFAULT_OPTIMIZE_HOURS
    chunker: ChunkerParams = field(default_factory=ChunkerParams)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "IngestParams":
        env = os.environ if env is None else env
        every = int(env.get(ENV_OPTIMIZE_EVERY) or DEFAULT_OPTIMIZE_EVERY)
        hours = float(env.get(ENV_OPTIMIZE_HOURS) or DEFAULT_OPTIMIZE_HOURS)
        return cls(optimize_every=every, optimize_hours=hours)


@dataclass
class ProcessReport:
    """process_all / retry_failed 的汇总。"""

    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    deleted: int = 0
    noop: int = 0
    failed_paths: list[str] = field(default_factory=list)

    def add(self, result: str, file_path: str) -> None:
        if result == "indexed":
            self.indexed += 1
        elif result == "skipped":
            self.skipped += 1
        elif result == "failed":
            self.failed += 1
            self.failed_paths.append(file_path)
        elif result == "deleted":
            self.deleted += 1
        else:
            self.noop += 1

    def as_dict(self) -> dict:
        return {
            "indexed": self.indexed, "skipped": self.skipped,
            "failed": self.failed, "deleted": self.deleted, "noop": self.noop,
            "failed_paths": list(self.failed_paths),
        }


# --------------------------------------------------------------------- 管线

class IngestPipeline:
    """切块 → embedding → FTS/向量双写的摄入编排器。

    协作对象（生命周期归调用方管理）：
    - :class:`~knowledge_vault.store.Store`：登记 + chunks/FTS；
    - :class:`~knowledge_vault.vectorstore.ZvecStore`：向量；
    - :class:`PipelineState`：pipeline.db 状态机；
    - :class:`~knowledge_vault.embedder.Embedder`：向量计算。
    """

    def __init__(
        self,
        store: Store,
        vectors: ZvecStore,
        state: PipelineState,
        *,
        embedder: Embedder,
        params: IngestParams | None = None,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.state = state
        self.embedder = embedder
        self.params = params or IngestParams.from_env()
        # 组合根职责：把 pipeline.db 绑定为向量库 optimize 攒批状态的唯一真源，
        # 保证"写入记账"（ZvecStore._record_pending）与"阈值检查"（maybe_optimize）
        # 读的是同一份状态，否则 pending 永远对不上、optimize 永不触发。
        self.vectors.attach_optimize_state(state)

    # ------------------------------------------------------ 单文件入口

    def process_file(self, file_path: str | Path) -> str:
        """处理单个文件，返回终态：``indexed`` / ``skipped`` / ``failed`` /
        ``deleted``（文件已不存在且已登记）/ ``noop``（不存在且未登记）。

        任意步骤异常都落 ``failed``（retries+1），不向上抛。
        """
        path = Path(file_path)
        path_str = str(path)
        if not path.exists():
            return self.handle_deleted(path)

        try:
            task = self.state.ensure_task(path_str)
            content_hash = compute_content_hash(path)
            # 断点续传：同 content_hash 已 indexed 的跳过
            if (
                task.state == "indexed"
                and task.content_hash == content_hash
            ):
                return "skipped"

            stat = path.stat()
            self.state.set_task_state(
                path_str, "pending", content_hash=content_hash, error=None
            )

            # ---- 登记层对齐（新文件登记；修改则刷新登记元数据）
            doc = self._get_active_doc(path_str)
            if doc is None:
                doc = self.store.register_file(path)
            elif doc.content_hash != content_hash:
                doc = self._refresh_registered(doc, path, content_hash, stat.st_size)

            file_id = doc.file_id

            # ---- 幂等清理：先清旧（FTS chunks + 向量），再重写
            self.store.delete_chunks_for_file(file_id)
            self.vectors.delete_file(file_id)

            # ---- 切块（parsed）
            chunks = self._chunk_file(path, stat.st_size)
            self.state.set_task_state(
                path_str, "parsed", content_hash=content_hash, file_id=file_id
            )

            # ---- embedding（embedded）
            texts: list[str] = []
            seqs: list[int] = []
            kinds: list[str] = []
            if doc.summary:
                texts.append(doc.summary)
                seqs.append(0)
                kinds.append("summary")
            for chunk in chunks:
                texts.append(chunk.text)
                seqs.append(chunk.seq)
                kinds.append("chunk")
            if texts:
                vecs = self.embedder.encode(texts)
            else:
                vecs = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self.state.set_task_state(path_str, "embedded")

            # ---- 双写（indexed）：先 FTS 后向量（两侧以 chunk_seq 对齐）。
            # 向量侧必须用 upsert：上面的 delete_by_filter 会留下墓碑
            # （id 在下次 optimize 前不可 insert，见 ZvecStore.delete_file），
            # upsert 既覆盖存活 id 也覆盖墓碑 id，重写路径因此幂等。
            if chunks:
                self.store.add_chunks(
                    file_id, [c.text for c in chunks], start_seq=1
                )
            if doc.summary:
                self.store.add_summary_chunk(file_id, doc.summary)
            if seqs:
                self.vectors.upsert_chunks(file_id, seqs, vecs, kinds)
            self.state.set_task_state(
                path_str, "indexed", error=None,
                content_hash=content_hash, file_id=file_id,
            )
            self.maybe_optimize()
            return "indexed"
        except Exception as exc:  # noqa: BLE001 - 管线吞异常落 failed 状态
            self.state.mark_failed(path_str, error=f"{type(exc).__name__}: {exc}")
            return "failed"

    def handle_deleted(self, file_path: str | Path) -> str:
        """文件消失语义：soft_delete + 删 chunks + 删向量 + 清 task 行。

        返回 ``deleted``（确有登记被软删）或 ``noop``（登记层无此活跃文件）。
        """
        path_str = str(file_path)
        doc = self._get_active_doc(path_str)
        result = "noop"
        if doc is not None:
            self.store.soft_delete(doc.file_id)
            self.store.delete_chunks_for_file(doc.file_id)
            self.vectors.delete_file(doc.file_id)
            result = "deleted"
        self.state.delete_task(path_str)
        return result

    def handle_moved(self, old_path: str | Path, new_path: str | Path) -> int | None:
        """移动/重命名语义：move_file（新 file_id）+ 旧 id 向量/FTS 清理 +
        新路径全量重建。返回新 file_id（旧路径无登记时返回 None，新路径按
        新文件走 :meth:`process_file`）。"""
        old_str, new_str = str(old_path), str(new_path)
        doc = self._get_active_doc(old_str)
        if doc is None:
            self.state.delete_task(old_str)
            self.process_file(new_path)
            new_doc = self._get_active_doc(new_str)
            return new_doc.file_id if new_doc is not None else None
        try:
            new_doc = self.store.move_file(doc.file_id, new_path)
        except UnknownFileIdError:  # pragma: no cover - doc 刚查到，防御
            self.process_file(new_path)
            return None
        # 向量与 FTS 按旧 file_id 清理（move 产生新 file_id，需按新 id 重建）
        self.store.delete_chunks_for_file(doc.file_id)
        self.vectors.delete_file(doc.file_id)
        self.state.delete_task(old_str)
        self.process_file(new_path)
        return new_doc.file_id

    # ------------------------------------------------------ 批量入口

    def process_all(
        self,
        *,
        vault_root: str | Path | None = None,
        limit: int | None = None,
        sweep_deleted: bool = True,
    ) -> ProcessReport:
        """扫描 vault 根目录全量处理（幂等，可反复跑）。

        - 目录下全部文件逐个 :meth:`process_file`（未变的靠断点续传跳过）；
        - ``sweep_deleted=True`` 时把登记层活跃但磁盘已消失的文件软删；
        - 隐藏目录/隐藏文件与 ``__pycache__`` 不进管线（与 watcher 口径一致）。
        """
        root = Path(vault_root) if vault_root is not None else self.store.config.vault_root
        report = ProcessReport()
        if root.exists():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(
                    d for d in dirnames
                    if not d.startswith(".") and d != "__pycache__"
                )
                for name in sorted(filenames):
                    if name.startswith("."):
                        continue
                    if limit is not None and report.indexed >= limit:
                        break
                    path = Path(dirpath) / name
                    report.add(self.process_file(path), str(path))
        if sweep_deleted:
            for doc in list(self.store.iterate_files(include_deleted=False)):
                if not Path(doc.file_path).exists():
                    result = self.handle_deleted(doc.file_path)
                    report.add(result, doc.file_path)
        self.maybe_optimize()
        return report

    def retry_failed(self, *, limit: int | None = None) -> ProcessReport:
        """重排 failed 任务（retries 未达上限的）并逐个重处理。"""
        report = ProcessReport()
        for task in self.state.list_tasks(state="failed"):
            if task.retries >= self.params.max_retries:
                continue
            if limit is not None and report.indexed >= limit:
                break
            if not Path(task.file_path).exists():
                report.add(self.handle_deleted(task.file_path), task.file_path)
                continue
            report.add(self.process_file(task.file_path), task.file_path)
        return report

    def pending(self) -> list[Task]:
        """当前处于 pending 状态的任务（快照）。"""
        return self.state.list_tasks(state="pending")

    # ------------------------------------------------------ 运维入口

    def maybe_optimize(self) -> bool:
        """按攒批阈值触发 optimize（见 :meth:`ZvecStore.optimize_if_needed`）。"""
        return self.vectors.optimize_if_needed(
            self.state,
            every=self.params.optimize_every,
            hours=self.params.optimize_hours,
        )

    def optimize_now(self) -> None:
        """手动触发 flush + optimize（低频运维动作），并清零攒批状态。"""
        self.vectors.optimize()

    # ------------------------------------------------------ 内部工具

    def _get_active_doc(self, file_path: str) -> Document | None:
        """按路径查登记层活跃记录（registry 未暴露按路径查询，这里直查
        documents 表；语义与 registry 的 partial UNIQUE INDEX 一致）。"""
        row = self.store.conn.execute(
            "SELECT file_id FROM documents WHERE file_path = ? AND deleted_at IS NULL",
            (file_path,),
        ).fetchone()
        if row is None:
            return None
        return self.store.get_file(int(row["file_id"]))

    def _refresh_registered(
        self, doc: Document, path: Path, content_hash: str, size_bytes: int
    ) -> Document:
        """文件被修改后刷新登记行（content_hash/size_bytes/mtime/updated_at）。

        注意：这是对 documents 表的最小直写，语义上属 registry 职责
        （registry.py 现无 update API；候选下沉见交付报告的最小 diff 建议）。
        is_large 维持登记时取值（登记层自有阈值口径，不由管线改写）。
        """
        mtime = utc_iso(path.stat().st_mtime)
        self.store.conn.execute(
            "UPDATE documents SET content_hash = ?, size_bytes = ?, mtime = ?, "
            "updated_at = ? WHERE file_id = ?",
            (content_hash, int(size_bytes), mtime, utc_iso(), doc.file_id),
        )
        refreshed = self.store.get_file(doc.file_id)
        assert refreshed is not None  # 刚 UPDATE 的行必然存在
        return refreshed

    def _chunk_file(self, path: Path, size_bytes: int):
        """按扩展名与 50MB 阈值决定切块方式；不切块的文件返回空列表。"""
        if size_bytes > self.params.large_file_threshold:
            return []  # >50MB：本体不切块不进向量库（summary 由调用方另行处理）
        ext = path.suffix.lower()
        if ext in MARKDOWN_EXTENSIONS:
            return chunk_markdown(_read_text(path), self.params.chunker)
        if ext in TEXT_EXTENSIONS:
            return chunk_plain(_read_text(path), self.params.chunker)
        return []  # 非文本类：只登记不切块（映射规则 Q8）


def _read_text(path: Path) -> str:
    """读文本（UTF-8，非法字节替换而非失败——真源区允许个别脏文件）。"""
    return path.read_text(encoding="utf-8", errors="replace")

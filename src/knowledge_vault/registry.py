"""documents 登记层：登记 / 软删除 / 移动 / 查询。

语义要点（设计规范）：
- file_id 与物理文件 1:1（同内容不同路径 = 两个 id）；同一 ``file_path`` 的活跃
  记录唯一（partial UNIQUE INDEX），重复登记抛 :class:`DuplicateFileError`。
- 软删除 = 仅置 ``deleted_at``；登记层不隐藏软删除记录（get/iterate 一视同仁，
  过滤是召回层的事）。
- 移动/重命名 = 旧记录置 ``deleted_at`` + 新记录分配新 file_id（内容 hash 不变），
  两者在同一事务内完成。
- ``title`` 默认取文件名去扩展名；``created_at`` 默认取 mtime（ISO8601）；
  ``summary`` 可空且禁止硬写占位内容。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from .file_id import DEFAULT_SEED, next_file_id

STATUSES = ("now", "library")

#: is_large 的阶段1默认判定阈值（超大文件仅登记元数据；可按次覆盖）
LARGE_FILE_THRESHOLD_BYTES = 100 * 1024 * 1024

_HASH_BLOCK = 1024 * 1024

_COLUMNS = (
    "file_id", "file_path", "file_type", "size_bytes", "is_large", "title",
    "context_tag", "summary", "status", "content_hash",
    "created_at", "mtime", "registered_at", "updated_at", "deleted_at",
)


class DuplicateFileError(Exception):
    """同 file_path 的活跃记录已存在。"""


class UnknownFileIdError(Exception):
    """file_id 不存在。"""


class RegistryValueError(ValueError):
    """登记参数非法（status 枚举、context_tag 形态、软删除记录不可移动等）。"""


@dataclass(frozen=True)
class Document:
    """documents 表一行（context_tag 已还原为 list[str]）。"""

    file_id: int
    file_path: str
    file_type: str
    size_bytes: int
    is_large: bool
    title: str
    context_tag: list[str]
    summary: str | None
    status: str
    content_hash: str
    created_at: str
    mtime: str
    registered_at: str
    updated_at: str
    deleted_at: str | None

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 工具函数

def utc_iso(ts: float | None = None) -> str:
    """Unix 时间戳 / 当前时刻 → UTC ISO8601 字符串（秒精度，``Z`` 后缀）。"""
    dt = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts is not None
        else datetime.now(timezone.utc)
    )
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def infer_file_type(path: str | Path) -> str:
    """由扩展名推断 file_type（小写、去点）；无扩展名 → ``unknown``。"""
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def derive_title(path: str | Path) -> str:
    """默认标题规则：文件名去扩展名。"""
    return Path(path).stem


def compute_content_hash(path: str | Path) -> str:
    """流式计算文件 sha256，取前 12 位十六进制。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def _to_document(row: sqlite3.Row) -> Document:
    context_tag = json.loads(row["context_tag"]) if row["context_tag"] else []
    return Document(
        file_id=int(row["file_id"]),
        file_path=row["file_path"],
        file_type=row["file_type"],
        size_bytes=int(row["size_bytes"]),
        is_large=bool(row["is_large"]),
        title=row["title"],
        context_tag=context_tag,
        summary=row["summary"],
        status=row["status"],
        content_hash=row["content_hash"],
        created_at=row["created_at"],
        mtime=row["mtime"],
        registered_at=row["registered_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
    )


def _get_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM documents WHERE file_id = ?", (file_id,)
    ).fetchone()


# ------------------------------------------------------------------- 写入侧

def register_file(
    conn: sqlite3.Connection,
    file_path: str | Path,
    *,
    id_seed: int = DEFAULT_SEED,
    content_hash: str | None = None,
    size_bytes: int | None = None,
    title: str | None = None,
    context_tag: Sequence[str] | None = None,
    summary: str | None = None,
    status: str = "library",
    is_large: bool | None = None,
    is_large_threshold: int = LARGE_FILE_THRESHOLD_BYTES,
    mtime: str | None = None,
    created_at: str | None = None,
    registered_at: str | None = None,
) -> Document:
    """登记一个文件，返回新建记录。

    默认从文件本体取值：``size_bytes``/``mtime`` 来自 stat，``content_hash`` 为
    sha256 前 12 位。文件不存在时必须显式给出 ``content_hash``/``size_bytes``/``mtime``
    （为后续"仅登记元数据"的管线留口）。``title`` 缺省取文件名去扩展名，
    ``created_at`` 缺省取 mtime。

    去重语义：同 ``file_path`` 已有活跃记录时抛 :class:`DuplicateFileError`
    （先查后插 + 唯一索引兜底）。失败的分配不消耗路径，但可能跳号（id 永不回收）。
    """
    if status not in STATUSES:
        raise RegistryValueError(f"status 必须是 {STATUSES} 之一，收到 {status!r}")
    if summary is not None and not isinstance(summary, str):
        raise RegistryValueError("summary 必须是 str 或 None，禁止硬写占位内容")
    tag_list = [str(tag) for tag in context_tag] if context_tag else []

    path_str = str(file_path)
    path_obj = Path(file_path)
    try:
        stat = path_obj.stat()
    except OSError:
        stat = None

    if size_bytes is None:
        if stat is None:
            raise FileNotFoundError(f"文件不存在且未显式提供 size_bytes: {path_str}")
        size_bytes = stat.st_size
    if mtime is None:
        if stat is None:
            raise FileNotFoundError(f"文件不存在且未显式提供 mtime: {path_str}")
        mtime = utc_iso(stat.st_mtime)
    if created_at is None:
        created_at = mtime  # 规则：created_at 用 mtime
    if content_hash is None:
        if stat is None:
            raise FileNotFoundError(f"文件不存在且未显式提供 content_hash: {path_str}")
        content_hash = compute_content_hash(path_obj)
    if title is None:
        title = derive_title(path_obj)
    if is_large is None:
        is_large = size_bytes >= is_large_threshold

    now = registered_at or utc_iso()

    conn.execute("BEGIN IMMEDIATE")
    try:
        dup = conn.execute(
            "SELECT 1 FROM documents WHERE file_path = ? AND deleted_at IS NULL",
            (path_str,),
        ).fetchone()
        if dup:
            raise DuplicateFileError(f"同路径活跃记录已存在: {path_str}")
        file_id = next_file_id(conn, seed=id_seed)
        conn.execute(
            f"INSERT INTO documents ({', '.join(_COLUMNS)}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id, path_str, infer_file_type(path_obj), int(size_bytes),
                int(bool(is_large)), title, json.dumps(tag_list, ensure_ascii=False),
                summary, status, content_hash, created_at, mtime, now, now, None,
            ),
        )
        conn.execute("COMMIT")
    except sqlite3.IntegrityError as exc:  # 并发兜底：唯一索引命中
        conn.execute("ROLLBACK")
        raise DuplicateFileError(f"同路径活跃记录已存在: {path_str}") from exc
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return get_file(conn, file_id)  # type: ignore[return-value]


def soft_delete(
    conn: sqlite3.Connection, file_id: int, *, deleted_at: str | None = None
) -> Document:
    """软删除：仅置 ``deleted_at``（不改动其他任何字段），幂等。

    chunks/edges 不在此处清理——登记层不过滤，过滤与清理是召回层/管线的事。
    """
    row = _get_row(conn, file_id)
    if row is None:
        raise UnknownFileIdError(f"file_id 不存在: {file_id}")
    if row["deleted_at"] is None:
        conn.execute(
            "UPDATE documents SET deleted_at = ? WHERE file_id = ?",
            (deleted_at or utc_iso(), file_id),
        )
    return get_file(conn, file_id)  # type: ignore[return-value]


def move_file(
    conn: sqlite3.Connection,
    file_id: int,
    new_path: str | Path,
    *,
    id_seed: int = DEFAULT_SEED,
    title: str | None = None,
    registered_at: str | None = None,
    deleted_at: str | None = None,
) -> Document:
    """移动/重命名：旧记录置 ``deleted_at`` + 新记录分配新 file_id（单事务）。

    内容侧字段（content_hash / size_bytes / is_large / context_tag / summary /
    status / created_at / mtime）原样继承；``title`` 与 ``file_type`` 按新路径
    重新推断（可显式覆盖 title）。
    """
    old = get_file(conn, file_id)
    if old is None:
        raise UnknownFileIdError(f"file_id 不存在: {file_id}")
    if old.deleted_at is not None:
        raise RegistryValueError(f"file_id {file_id} 已是软删除记录，不可移动")

    new_path_str = str(new_path)
    now = registered_at or utc_iso()
    when = deleted_at or now
    new_title = title if title is not None else derive_title(new_path)

    conn.execute("BEGIN IMMEDIATE")
    try:
        dup = conn.execute(
            "SELECT 1 FROM documents WHERE file_path = ? AND deleted_at IS NULL",
            (new_path_str,),
        ).fetchone()
        if dup:
            raise DuplicateFileError(f"目标路径已有活跃记录: {new_path_str}")
        conn.execute(
            "UPDATE documents SET deleted_at = ? WHERE file_id = ?", (when, file_id)
        )
        new_id = next_file_id(conn, seed=id_seed)
        conn.execute(
            f"INSERT INTO documents ({', '.join(_COLUMNS)}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id, new_path_str, infer_file_type(new_path), old.size_bytes,
                int(old.is_large), new_title,
                json.dumps(old.context_tag, ensure_ascii=False),
                old.summary, old.status, old.content_hash,
                old.created_at, old.mtime, now, now, None,
            ),
        )
        conn.execute("COMMIT")
    except sqlite3.IntegrityError as exc:
        conn.execute("ROLLBACK")
        raise DuplicateFileError(f"目标路径已有活跃记录: {new_path_str}") from exc
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return get_file(conn, new_id)  # type: ignore[return-value]


# ------------------------------------------------------------------- 读取侧

def get_file(conn: sqlite3.Connection, file_id: int) -> Document | None:
    """按 id 取单条记录（含软删除记录）；不存在返回 None。"""
    row = _get_row(conn, file_id)
    return _to_document(row) if row is not None else None


def iterate_files(
    conn: sqlite3.Connection,
    *,
    include_deleted: bool = True,
    status: str | None = None,
) -> Iterator[Document]:
    """遍历登记记录，按 file_id 升序。

    登记层默认不过滤（含软删除记录）；``include_deleted=False`` / ``status``
    是给调用方的显式开关，不是登记层的隐式行为。
    """
    sql = f"SELECT {', '.join(_COLUMNS)} FROM documents"
    conditions: list[str] = []
    params: list[object] = []
    if not include_deleted:
        conditions.append("deleted_at IS NULL")
    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY file_id"
    for row in conn.execute(sql, params):
        yield _to_document(row)

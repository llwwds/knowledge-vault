"""连接管理与建库/迁移（schema_version）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from .config import VaultConfig, load_config
from .graph import add_edge, expand, remove_edges_by_src
from .registry import (
    get_file,
    iterate_files,
    move_file,
    register_file,
    soft_delete,
)
from .textindex import (
    add_chunk,
    add_chunks,
    add_summary_chunk,
    delete_chunks_for_file,
    search,
    snippet,
)

#: 当前 schema 版本；迁移注册表按"目标版本"登记迁移函数，v1 为初始版本。
SCHEMA_VERSION = 1
MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {}

_SCHEMA_SQL = Path(__file__).with_name("schema.sql")


class SchemaVersionError(RuntimeError):
    """库文件的 schema 版本比当前代码新，无法打开（防止旧代码写坏新库）。"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """打开 SQLite 连接并设置运行 PRAGMA。

    - ``journal_mode=WAL``（持久化在库文件，重复设置无害）
    - ``foreign_keys=ON``（chunks.file_id → documents.file_id）
    - ``synchronous=NORMAL``（WAL 推荐档位）
    - ``row_factory=sqlite3.Row``（各模块按列名取值）

    连接为 autocommit（``isolation_level=None``）；多步写入由各函数显式
    ``BEGIN IMMEDIATE`` / ``COMMIT`` 包住。
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表（幂等）并校验/推进 schema_version。

    - 空库：写入当前 ``SCHEMA_VERSION``。
    - 版本低于当前：按 ``MIGRATIONS`` 逐版本迁移后推进。
    - 版本高于当前：抛 :class:`SchemaVersionError`。
    """
    conn.executescript(_SCHEMA_SQL.read_text(encoding="utf-8"))
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        return
    version = int(row[0])
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"数据库 schema v{version} 比代码 v{SCHEMA_VERSION} 新，拒绝打开"
        )
    for target in range(version + 1, SCHEMA_VERSION + 1):
        migrate = MIGRATIONS.get(target)
        if migrate is not None:
            migrate(conn)
        conn.execute("UPDATE schema_version SET version = ?", (target,))


class Store:
    """config + 连接 + 各模块公共入口的薄门面。

    职责仅是把 :class:`~knowledge_vault.config.VaultConfig`（file_id 种子、
    userdict 路径）接到各模块函数上；也可以只用 ``store.connect`` + 各模块
    函数自行组装。
    """

    def __init__(
        self,
        config: VaultConfig | None = None,
        *,
        db_path: str | Path | None = None,
        init: bool = True,
    ) -> None:
        self.config = config or load_config()
        self._db_path = Path(db_path) if db_path is not None else self.config.db_path
        self.conn = connect(self._db_path)
        try:
            if init:
                init_db(self.conn)
        except BaseException:
            self.conn.close()
            raise
        self._tokenizer = None

    # ---- 基础设施

    @property
    def tokenizer(self):
        """与 config 绑定的分词器（懒构造，首次访问才加载 userdict）。"""
        if self._tokenizer is None:
            from .textindex import Tokenizer

            self._tokenizer = Tokenizer(userdict_path=self.config.userdict_path)
        return self._tokenizer

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- 登记层（registry.py）

    def register_file(self, file_path, **kwargs):
        return register_file(self.conn, file_path, id_seed=self.config.file_id_seed, **kwargs)

    def soft_delete(self, file_id, **kwargs):
        return soft_delete(self.conn, file_id, **kwargs)

    def move_file(self, file_id, new_path, **kwargs):
        return move_file(self.conn, file_id, new_path, id_seed=self.config.file_id_seed, **kwargs)

    def get_file(self, file_id):
        return get_file(self.conn, file_id)

    def iterate_files(self, **kwargs):
        return iterate_files(self.conn, **kwargs)

    # ---- 全文索引（textindex.py）

    def add_chunk(self, file_id, text, **kwargs):
        return add_chunk(self.conn, self.tokenizer, file_id, text, **kwargs)

    def add_chunks(self, file_id, texts, **kwargs):
        return add_chunks(self.conn, self.tokenizer, file_id, texts, **kwargs)

    def add_summary_chunk(self, file_id, text):
        return add_summary_chunk(self.conn, self.tokenizer, file_id, text)

    def delete_chunks_for_file(self, file_id):
        return delete_chunks_for_file(self.conn, self.tokenizer, file_id)

    def search(self, query, **kwargs):
        return search(self.conn, self.tokenizer, query, **kwargs)

    def snippet(self, query, **kwargs):
        return snippet(self.conn, self.tokenizer, query, **kwargs)

    # ---- 图（graph.py）

    def add_edge(self, src, dst, edge_type):
        return add_edge(self.conn, src, dst, edge_type)

    def remove_edges_by_src(self, src, edge_type=None):
        return remove_edges_by_src(self.conn, src, edge_type)

    def expand(self, file_id, **kwargs):
        return expand(self.conn, file_id, **kwargs)

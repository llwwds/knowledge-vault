"""单调递增 file_id 分配器。

语义（设计规范）：
- 程序生成、整数、单调递增、**永不回收**；计数器落在 ``id_counters`` 表。
- 起始值可由 ``KV_FILE_ID_SEED``（经 :class:`~knowledge_vault.config.VaultConfig.file_id_seed`）
  注入：首次初始化时计数器置为 ``seed - 1``，首个分配值即 ``seed``（默认从 1 开始）。
- seed 只会抬高计数器、从不回退（``MAX`` 语义），为快照批量导入预留迁移能力。
- 分配由两条原子语句完成（upsert 抬底 + ``UPDATE ... RETURNING`` 自增），
  不自管事务，可安全嵌套在调用方的事务里；autocommit 下也天然并发安全。
  需要 SQLite >= 3.35（Python 3.12 自带的 sqlite3 均满足）。
"""

from __future__ import annotations

import sqlite3

COUNTER_NAME = "file_id"
DEFAULT_SEED = 1


def next_file_id(conn: sqlite3.Connection, *, seed: int = DEFAULT_SEED) -> int:
    """分配下一个 file_id。

    若计数器当前值落后于 ``seed - 1``（例如新库注入了更大的 seed），先跳到
    ``seed - 1`` 再自增；否则直接在当前值上 ``+1``。
    """
    seed = int(seed)
    conn.execute(
        "INSERT INTO id_counters(name, value) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET value = MAX(value, excluded.value)",
        (COUNTER_NAME, max(seed - 1, 0)),
    )
    row = conn.execute(
        "UPDATE id_counters SET value = value + 1 WHERE name = ? RETURNING value",
        (COUNTER_NAME,),
    ).fetchone()
    if row is None:  # pragma: no cover - upsert 刚写入必然存在
        raise RuntimeError("file_id counter row missing after upsert")
    return int(row[0])


def current_file_id(conn: sqlite3.Connection) -> int:
    """当前已分配到的最大 file_id（未初始化时返回 0）。"""
    row = conn.execute(
        "SELECT value FROM id_counters WHERE name = ?", (COUNTER_NAME,)
    ).fetchone()
    return int(row[0]) if row else 0

"""store.py：建库、PRAGMA、schema_version、幂等初始化、重开持久化。"""

from __future__ import annotations

import pytest

from knowledge_vault import SCHEMA_VERSION, SchemaVersionError, Store, connect, init_db

EXPECTED_OBJECTS = {"documents", "chunks", "edges", "id_counters", "schema_version"}


class TestConnectAndInit:
    def test_creates_db_with_all_tables(self, config, conn, store):
        assert config.db_path.exists()
        names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        assert EXPECTED_OBJECTS <= names
        # FTS5 虚表存在且可查
        count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert count == 0

    def test_wal_and_foreign_keys(self, conn):
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_schema_version_recorded(self, conn):
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION == 1

    def test_init_db_idempotent(self, conn):
        init_db(conn)
        init_db(conn)
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0

    def test_wal_sidecar_files_created(self, config, store):
        # WAL 模式下写一条数据，-wal 文件出现
        store.conn.execute("CREATE TABLE IF NOT EXISTS _wal_probe(x)")
        store.conn.execute("INSERT INTO _wal_probe VALUES (1)")
        assert (config.db_path.parent / "kv.sqlite3-wal").exists()


class TestSchemaVersionGuard:
    def test_future_version_rejected(self, config):
        probe = connect(config.db_path)
        init_db(probe)
        probe.execute("UPDATE schema_version SET version = 99")
        probe.close()
        with pytest.raises(SchemaVersionError):
            Store(config=config)

    def test_older_version_accepted_and_current(self, config):
        # v1 是初始版本：模拟"旧库"= 当前版本，打开后仍是当前版本
        probe = connect(config.db_path)
        init_db(probe)
        probe.execute("UPDATE schema_version SET version = 1")
        probe.close()
        with Store(config=config) as store:
            row = store.conn.execute("SELECT version FROM schema_version").fetchone()
            assert row[0] == SCHEMA_VERSION


class TestReopen:
    def test_data_persists_across_reopen(self, config, make_file):
        f = make_file("persist.md", "内容")
        with Store(config=config) as first:
            doc = first.register_file(f)
        with Store(config=config) as second:
            got = second.get_file(doc.file_id)
            assert got is not None
            assert got.file_path == str(f)
            assert got.content_hash == doc.content_hash

    def test_store_context_manager_closes(self, config):
        with Store(config=config) as store:
            assert store.conn is not None
        # 关闭后连接不可再用
        with pytest.raises(Exception):
            store.conn.execute("SELECT 1")


class TestStoreFacade:
    def test_facade_smoke(self, store, make_file):
        f1 = make_file("n1.md", "占位一")
        f2 = make_file("n2.md", "占位二")
        d1 = store.register_file(f1)
        d2 = store.register_file(f2)
        store.add_chunks(d1.file_id, ["zvec 向量检索入门"])
        store.add_chunks(d2.file_id, ["递归 CTE 图扩展"])
        assert store.add_edge(d1.file_id, d2.file_id, "link") is True

        hits = store.search("向量")
        assert [h.file_id for h in hits] == [d1.file_id]
        result = store.expand(d1.file_id, max_hops=1)
        assert result.nodes == [(d2.file_id, 1)]

        assert store.snippet("向量")  # 返回非空片段列表

    def test_tokenizer_lazy_and_shared(self, store):
        tok = store.tokenizer
        assert tok is store.tokenizer  # 同一实例
        toks = tok.tokenize("知识图谱入门")
        assert toks
        assert all(any(ch.isalnum() for ch in t) for t in toks)

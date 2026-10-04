"""端到端：环境变量注入 → Store → 登记 → 切块 → 检索 → 图 → 生命周期 → 重开。

全程合成数据 + tmp_path，不触碰真实 state 与 vault。
"""

from __future__ import annotations

from knowledge_vault import Store, current_file_id, load_config


def test_end_to_end_with_env_injection(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    vault_root = tmp_path / "vault"
    vault_root.mkdir()

    monkeypatch.setenv("KV_STATE_DIR", str(state_dir))
    monkeypatch.setenv("KV_VAULT_ROOT", str(vault_root))
    monkeypatch.setenv("KV_FILE_ID_SEED", "10")
    monkeypatch.delenv("KV_USERDICT", raising=False)      # 走包内默认词典

    cfg = load_config()
    assert cfg.file_id_seed == 10
    assert cfg.db_path == state_dir / "kv.sqlite3"

    f1 = vault_root / "vector-notes.md"
    f1.write_text("向量检索笔记", encoding="utf-8")
    f2 = vault_root / "graph-notes.md"
    f2.write_text("图扩展笔记", encoding="utf-8")

    with Store() as kv:                                    # 配置全部来自 os.environ
        d1 = kv.register_file(f1)
        d2 = kv.register_file(f2)
        assert (d1.file_id, d2.file_id) == (10, 11)        # 种子注入生效

        kv.add_chunks(d1.file_id, ["zvec 负责向量检索"])
        kv.add_chunks(d2.file_id, ["递归 CTE 做图扩展"])
        kv.add_edge(d1.file_id, d2.file_id, "link")

        assert [h.file_id for h in kv.search("向量")] == [d1.file_id]
        assert kv.expand(d1.file_id, max_hops=1).nodes == [(d2.file_id, 1)]

        # 移动：旧记录软删 + 新 id；chunks 仍挂在旧记录下且依然可检索
        moved = kv.move_file(d1.file_id, vault_root / "vector-notes-v2.md")
        assert moved.file_id == 12
        assert {h.file_id for h in kv.search("向量")} == {d1.file_id}

        # 软删除：登记层不过滤，检索侧依然可见（过滤是召回层的事）
        kv.soft_delete(d2.file_id)
        assert kv.get_file(d2.file_id).deleted_at is not None
        assert kv.search("CTE") != []

        assert current_file_id(kv.conn) == 12

    # 重开：数据持久化、计数器接续
    with Store() as kv2:
        reopened = kv2.get_file(12)
        assert reopened.file_path.endswith("vector-notes-v2.md")
        assert reopened.deleted_at is None
        f3 = vault_root / "new-file.md"
        f3.write_text("新文件", encoding="utf-8")
        assert kv2.register_file(f3).file_id == 13
        assert current_file_id(kv2.conn) == 13


def test_registry_and_index_survive_move_semantics(store, make_file, tmp_path):
    """move 后旧 file_id 的 chunks 不动：索引层与登记层解耦，重建是管线的事。"""
    src = make_file("old.md", "占位")
    doc = store.register_file(src)
    store.add_chunks(doc.file_id, ["connect_mcp 的使用说明"])
    store.add_edge(doc.file_id, doc.file_id + 100, "link")  # 指向尚未登记的节点也允许

    moved = store.move_file(doc.file_id, tmp_path / "new.md")

    # 旧记录：软删、chunks 仍在、检索仍可达
    old = store.get_file(doc.file_id)
    assert old.deleted_at is not None
    assert [h.chunk_id for h in store.search("connect_mcp", file_id=doc.file_id)]
    # 新记录：无 chunks（管线尚未重新摄入）
    assert store.search("connect_mcp", file_id=moved.file_id) == []

"""file_id 分配器：单调、永不回收、种子注入。"""

from __future__ import annotations

from dataclasses import replace

from knowledge_vault import Store, current_file_id, next_file_id


class TestAllocator:
    def test_first_id_is_one_then_monotonic(self, conn):
        assert current_file_id(conn) == 0
        assert [next_file_id(conn) for _ in range(5)] == [1, 2, 3, 4, 5]
        assert current_file_id(conn) == 5

    def test_ids_never_recycled_across_soft_delete(self, store, make_file):
        f = make_file("doc.md", "x")
        d1 = store.register_file(f)
        store.soft_delete(d1.file_id)
        d2 = store.register_file(f)  # 路径释放后可重登记，但 id 继续前进
        assert d2.file_id == d1.file_id + 1
        assert current_file_id(store.conn) == d2.file_id

    def test_counter_persists_across_reopen(self, config):
        with Store(config=config) as first:
            next_file_id(first.conn)
            next_file_id(first.conn)
        with Store(config=config) as second:
            assert current_file_id(second.conn) == 2
            assert next_file_id(second.conn) == 3


class TestSeed:
    def test_seed_raises_starting_point(self, conn):
        assert next_file_id(conn, seed=1000) == 1000
        assert next_file_id(conn) == 1001

    def test_seed_never_lowers_counter(self, conn):
        next_file_id(conn)  # 1
        next_file_id(conn)  # 2
        # 同值/更小 seed 不回退
        assert next_file_id(conn, seed=1) == 3
        assert next_file_id(conn, seed=1) == 4
        # 更大 seed 抬高起点（快照导入预留能力）
        assert next_file_id(conn, seed=5000) == 5000
        assert next_file_id(conn) == 5001

    def test_seed_via_config(self, config):
        seeded = replace(config, file_id_seed=77)
        with Store(config=seeded) as store:
            f = store.register_file(
                _touch(config.state_dir.parent / "seeded.md")
            )
            assert f.file_id == 77


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("seed", encoding="utf-8")
    return path

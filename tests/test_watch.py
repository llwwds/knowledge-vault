"""watcher → 摄入管线接线测试（真实 FTS-only 栈 + tmp vault 合成文件）。

覆盖：create/modify/delete/move 四类变更真的落到登记层与 FTS；受控写入目录
（``KV_WRITABLE_ROOT``）整棵子树不触发管线；``kv-watch --once`` 单轮对账冒烟；
``KV_WATCH=1`` 内嵌守护线程能摄入并能随服务关闭。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from knowledge_vault import api
from knowledge_vault.cli import main
from knowledge_vault.store import Store
from knowledge_vault.watcher import WatcherService


@dataclass
class Recorder:
    """鸭子类型 fake pipeline：只记录分发调用（不触真实索引）。"""

    processed: list[str] = field(default_factory=list)

    def process_file(self, path):
        self.processed.append(str(path))
        return "indexed"

    def handle_deleted(self, path):
        return "deleted"

    def handle_moved(self, old, new):
        return 1


@pytest.fixture()
def fts_only(monkeypatch):
    """FTS-only 部署口径：不开向量库、不调 embedding（无需真实模型）。"""
    monkeypatch.setenv("KV_FTS_ONLY", "1")
    monkeypatch.delenv("KV_WRITABLE_ROOT", raising=False)


@pytest.fixture()
def vault(config) -> Path:
    config.vault_root.mkdir(parents=True, exist_ok=True)
    return config.vault_root


@pytest.fixture()
def reader(config):
    """只读断言用连接（与 watcher 自建栈各自独立连接，WAL 允许）。"""
    store = Store(config=config)
    yield store
    store.close()


@pytest.fixture()
def make_service(config, fts_only):
    """构造 WatcherService（间隔压到 50ms，便于守护线程测试）；用完统一关闭。"""
    created: list[WatcherService] = []

    def _make(**kwargs) -> WatcherService:
        service = WatcherService(config, interval=0.05, **kwargs)
        created.append(service)
        return service

    yield _make
    for service in created:
        service.close()


# ------------------------------------------------------- 四类变更真的入库


def test_created_file_becomes_searchable(make_service, vault, reader):
    service = make_service()
    note = vault / "note.md"
    note.write_text("# 笔记\n\n首版正文 token_alpha。\n", encoding="utf-8")

    assert service.poll_once() == 1

    hits = reader.search("token_alpha")
    assert hits, "新建文件应经 watcher 自动入库"
    doc = reader.get_file(hits[0].file_id)
    assert Path(doc.file_path) == note
    assert doc.deleted_at is None


def test_modified_file_reindexes_same_file_id(make_service, vault, reader):
    service = make_service()
    note = vault / "grow.md"
    note.write_text("# 笔记\n\n旧正文 token_alpha。\n", encoding="utf-8")
    service.poll_once()
    file_id = reader.search("token_alpha")[0].file_id

    note.write_text("# 笔记\n\n新正文 token_beta，这一版明显更长一些。\n", encoding="utf-8")
    assert service.poll_once() == 1

    hits = reader.search("token_beta")
    assert hits and hits[0].file_id == file_id, "修改不换 file_id"
    assert reader.search("token_alpha") == [], "旧 chunk 必须被清掉"


def test_deleted_file_is_soft_deleted_and_unsearchable(make_service, vault, reader):
    service = make_service()
    note = vault / "doomed.md"
    note.write_text("# 待删\n\n消失正文 token_gone。\n", encoding="utf-8")
    service.poll_once()
    file_id = reader.search("token_gone")[0].file_id

    note.unlink()
    assert service.poll_once() == 1

    assert reader.search("token_gone") == []
    assert reader.get_file(file_id).deleted_at is not None


def test_moved_file_reindexed_under_new_path(make_service, vault, reader):
    service = make_service()
    src = vault / "old.md"
    src.write_text("# 移动\n\n移动正文 token_move。\n", encoding="utf-8")
    service.poll_once()
    old_file_id = reader.search("token_move")[0].file_id

    dst = vault / "new.md"
    os.replace(src, dst)  # 同盘 rename：size+mtime 不变 → 被识别为 moved
    assert service.poll_once() == 1

    hits = reader.search("token_move")
    assert hits, "移动后新路径应重新入库"
    new_doc = reader.get_file(hits[0].file_id)
    assert Path(new_doc.file_path) == dst
    assert new_doc.file_id != old_file_id  # move 产生新 file_id
    assert reader.get_file(old_file_id).deleted_at is not None


# ------------------------------------------------------- 受控目录排除


def test_controlled_dir_never_reaches_pipeline(make_service, vault, monkeypatch):
    controlled = vault / "00 收集箱" / "000 暂未处理" / "KnowledgeVault API"
    controlled.mkdir(parents=True)
    monkeypatch.setenv("KV_WRITABLE_ROOT", str(controlled))
    recorder = Recorder()
    service = make_service(pipeline=recorder)

    (controlled / "api.md").write_text("# API\n\n受控写入。\n", encoding="utf-8")
    normal = vault / "normal.md"
    normal.write_text("# 普通\n\n普通写入。\n", encoding="utf-8")

    assert service.poll_once() == 1, "受控目录内的新建不得产生事件"
    assert recorder.processed == [str(normal)]


def test_controlled_dir_modification_and_deletion_are_ignored(
    make_service, vault, monkeypatch
):
    controlled = vault / "收集箱" / "KnowledgeVault API"
    controlled.mkdir(parents=True)
    note = controlled / "api.md"
    note.write_text("# API\n\n第一版。\n", encoding="utf-8")
    monkeypatch.setenv("KV_WRITABLE_ROOT", str(controlled))
    recorder = Recorder()
    service = make_service(pipeline=recorder)
    assert service.watcher.scan() == {}, "受控目录子树不应进入快照"

    note.write_text("# API\n\n第二版内容更长。\n", encoding="utf-8")
    note.unlink()
    assert service.poll_once() == 0
    assert recorder.processed == []


def test_controlled_dir_exclusion_survives_late_creation(make_service, vault, monkeypatch):
    """受控目录晚于 watcher 创建（首写才建）时也要排除。"""
    controlled = vault / "late" / "KnowledgeVault API"
    monkeypatch.setenv("KV_WRITABLE_ROOT", str(controlled))
    recorder = Recorder()
    service = make_service(pipeline=recorder)

    controlled.mkdir(parents=True)
    (controlled / "api.md").write_text("受控写入\n", encoding="utf-8")
    assert service.poll_once() == 0
    assert recorder.processed == []


# ------------------------------------------------------- kv-watch --once


def test_kv_watch_once_ingests_then_sweeps_deleted(config, vault, reader, fts_only, capsys):
    note = vault / "once.md"
    note.write_text("# 一次性\n\n冒烟正文 token_gamma。\n", encoding="utf-8")

    assert main(["kv-watch", "--once"], config=config) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["once"] is True and summary["events"] >= 1

    hits = reader.search("token_gamma")
    assert hits, "启动前已存在的文件也要靠 --once 对账入库"
    file_id = hits[0].file_id

    # 幂等：再对账两次不产生重复 chunk
    count_sql = "SELECT COUNT(*) FROM chunks WHERE file_id = ?"
    before = reader.conn.execute(count_sql, (file_id,)).fetchone()[0]
    assert main(["kv-watch", "--once"], config=config) == 0
    assert reader.conn.execute(count_sql, (file_id,)).fetchone()[0] == before

    note.unlink()
    assert main(["kv-watch", "--once"], config=config) == 0
    assert reader.search("token_gamma") == [], "消失的文件要被抓到并清索引"
    assert reader.get_file(file_id).deleted_at is not None


def test_kv_watch_interval_defaults_and_env(config, vault, fts_only, monkeypatch):
    service = WatcherService(config)
    try:
        assert service.interval == 30.0  # 兜底默认
    finally:
        service.close()

    monkeypatch.setenv("KV_WATCH_INTERVAL", "2.5")
    service = WatcherService(config)
    try:
        assert service.interval == 2.5
    finally:
        service.close()

    monkeypatch.setenv("KV_WATCH_INTERVAL", "not-a-number")
    service = WatcherService(config)
    try:
        assert service.interval == 30.0  # 非法值回落默认
    finally:
        service.close()


def test_kv_watch_rejects_non_positive_interval(config, vault, fts_only):
    with pytest.raises(ValueError):
        WatcherService(config, interval=0)


# ------------------------------------------------------- serve 内嵌 watcher


def test_embedded_watcher_thread_ingests_and_stops(config, vault, reader, fts_only, monkeypatch):
    monkeypatch.setenv("KV_WATCH_INTERVAL", "0.05")
    service = api._start_watcher(config)
    assert service is not None

    note = vault / "live.md"
    note.write_text("# 实时\n\n守护线程正文 token_delta。\n", encoding="utf-8")

    deadline = time.monotonic() + 10
    hits: list = []
    while time.monotonic() < deadline and not hits:
        hits = reader.search("token_delta")
        if not hits:
            time.sleep(0.05)
    assert hits, "内嵌监听线程应把新文件实时摄入"

    service.stop()
    # 线程停止后不再摄入：新写的文件不应再出现在索引里
    later = vault / "after-stop.md"
    later.write_text("# 停止后\n\n停止后正文 token_omega。\n", encoding="utf-8")
    time.sleep(0.3)
    assert reader.search("token_omega") == []


def test_serve_starts_watcher_only_when_kv_watch_enabled(config, fts_only, monkeypatch):
    """``KV_WATCH=1`` 才起内嵌监听（serve 本体用假 server 顶替，不真占端口）。"""

    class FakeServer:
        server_address = ("127.0.0.1", 0)

        def serve_forever(self) -> None:
            return

        def server_close(self) -> None:
            return

    started: list = []
    monkeypatch.setattr(api, "make_server", lambda *a, **kw: FakeServer())
    monkeypatch.setattr(api, "_start_watcher", lambda cfg: started.append(cfg))

    monkeypatch.delenv("KV_WATCH", raising=False)
    api.serve(config, port=0)
    assert started == []

    monkeypatch.setenv("KV_WATCH", "1")
    api.serve(config, port=0)
    assert started == [config]

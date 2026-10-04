"""watcher 测试：PollingWatcher 扫描/差分/移动配对/分发（合成文件 + tmp_path）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from knowledge_vault.watcher import (
    CREATED,
    DELETED,
    MODIFIED,
    MOVED,
    ChangeEvent,
    PollingWatcher,
)


def _stamp_ns(path: Path, ns: int) -> None:
    """显式设置 mtime_ns，避免测试里两次写文件落在同一时间戳上。"""
    os.utime(path, ns=(ns, ns))


@dataclass
class Recorder:
    """鸭子类型 fake pipeline：记录分发调用。"""

    processed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    moved: list[tuple[str, str]] = field(default_factory=list)

    def process_file(self, path):
        self.processed.append(str(path))
        return "indexed"

    def handle_deleted(self, path):
        self.deleted.append(str(path))
        return "deleted"

    def handle_moved(self, old, new):
        self.moved.append((str(old), str(new)))
        return 1


# ---------------------------------------------------------------- 扫描

def test_scan_finds_files_excludes_hidden_and_dirs(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("h", encoding="utf-8")
    obsidian = tmp_path / ".obsidian"
    obsidian.mkdir()
    (obsidian / "c.md").write_text("c", encoding="utf-8")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "d.md").write_text("d", encoding="utf-8")

    watcher = PollingWatcher(tmp_path)
    keys = set(watcher.scan())
    assert keys == {"a.md", os.path.join("sub", "b.md")}


def test_scan_missing_root_returns_empty(tmp_path):
    watcher = PollingWatcher(tmp_path / "not_yet")
    assert watcher.scan() == {}
    # 目录随后出现，下一轮能扫到
    (tmp_path / "not_yet").mkdir()
    (tmp_path / "not_yet" / "x.md").write_text("x", encoding="utf-8")
    assert set(watcher.scan()) == {"x.md"}


def test_custom_exclude_dirs(tmp_path):
    (tmp_path / "keep.md").write_text("k", encoding="utf-8")
    skip = tmp_path / "skipme"
    skip.mkdir()
    (skip / "drop.md").write_text("d", encoding="utf-8")
    watcher = PollingWatcher(tmp_path, exclude_dirs={"skipme"})
    assert set(watcher.scan()) == {"keep.md"}


def test_exclude_hidden_can_be_disabled(tmp_path):
    (tmp_path / ".dot.md").write_text("x", encoding="utf-8")
    watcher = PollingWatcher(tmp_path, exclude_hidden=False)
    assert set(watcher.scan()) == {".dot.md"}


def test_negative_interval_rejected(tmp_path):
    with pytest.raises(ValueError):
        PollingWatcher(tmp_path, interval=-1)


# ---------------------------------------------------------------- 差分

def test_poll_detects_created_modified_deleted(tmp_path):
    watcher = PollingWatcher(tmp_path)  # 基线：空
    assert watcher.poll_once() == []

    a = tmp_path / "a.md"
    a.write_text("first", encoding="utf-8")
    events = watcher.poll_once()
    assert [(e.kind, e.path) for e in events] == [(CREATED, str(a))]

    # 修改：内容与 mtime 都变
    a.write_text("second with more length", encoding="utf-8")
    _stamp_ns(a, 1_500_000_000_000_000_000)
    events = watcher.poll_once()
    assert [(e.kind, e.path) for e in events] == [(MODIFIED, str(a))]

    os.remove(a)
    events = watcher.poll_once()
    assert [(e.kind, e.path) for e in events] == [(DELETED, str(a))]


def test_poll_detects_move_by_size_mtime_pair(tmp_path):
    watcher = PollingWatcher(tmp_path)
    a = tmp_path / "origin.md"
    a.write_text("移动内容", encoding="utf-8")
    assert watcher.poll_once()  # 基线吃掉 created

    b = tmp_path / "target.md"
    os.rename(a, b)  # 同文件系统 rename 保留 size 与 mtime
    events = watcher.poll_once()
    assert len(events) == 1
    event = events[0]
    assert event.kind == MOVED
    assert event.old_path == str(a)
    assert event.path == str(b)


def test_poll_move_falls_back_to_delete_plus_create(tmp_path):
    """size/mtime 不一致（跨盘复制-删除场景）→ 拆成删除 + 新建，仍可处理。"""
    watcher = PollingWatcher(tmp_path)
    a = tmp_path / "src.md"
    a.write_text("内容", encoding="utf-8")
    watcher.poll_once()

    b = tmp_path / "dst.md"
    b.write_text("内容", encoding="utf-8")
    _stamp_ns(b, 1_600_000_000_000_000_000)  # 新路径 mtime 不同
    os.remove(a)
    events = watcher.poll_once()
    kinds = sorted(e.kind for e in events)
    assert kinds == [CREATED, DELETED]


def test_poll_modified_requires_stamp_change(tmp_path):
    """同路径同 size 同 mtime（无实际变化）→ 不产生事件。"""
    watcher = PollingWatcher(tmp_path)
    a = tmp_path / "same.md"
    a.write_text("stable", encoding="utf-8")
    watcher.poll_once()
    # 什么都不改
    assert watcher.poll_once() == []


# ---------------------------------------------------------------- 分发

def test_dispatch_routes_to_pipeline_handlers(tmp_path):
    recorder = Recorder()
    watcher = PollingWatcher(tmp_path, pipeline=recorder)
    a = tmp_path / "a.md"
    a.write_text("内容", encoding="utf-8")
    watcher.dispatch(watcher.poll_once())
    assert recorder.processed == [str(a)]

    b = tmp_path / "b.md"
    os.rename(a, b)
    watcher.dispatch(watcher.poll_once())
    assert recorder.moved == [(str(a), str(b))]

    os.remove(b)
    watcher.dispatch(watcher.poll_once())
    assert recorder.deleted == [str(b)]


def test_on_event_callback_observes_all(tmp_path):
    seen: list[ChangeEvent] = []
    watcher = PollingWatcher(tmp_path, on_event=seen.append)
    a = tmp_path / "a.md"
    a.write_text("x", encoding="utf-8")
    watcher.dispatch(watcher.poll_once())
    assert [e.kind for e in seen] == [CREATED]


def test_dispatch_without_pipeline_or_callback_is_safe(tmp_path):
    watcher = PollingWatcher(tmp_path)
    a = tmp_path / "a.md"
    a.write_text("x", encoding="utf-8")
    events = watcher.poll_once()
    watcher.dispatch(events)  # 无 sink：不抛错
    assert events


# ---------------------------------------------------------------- 循环

def test_run_max_polls_drives_loop(tmp_path):
    watcher = PollingWatcher(tmp_path, interval=0)
    a = tmp_path / "a.md"
    a.write_text("loop", encoding="utf-8")
    polls = watcher.run(max_polls=3)
    assert polls == 3


def test_run_processes_files_across_polls(tmp_path):
    recorder = Recorder()
    watcher = PollingWatcher(tmp_path, interval=0, pipeline=recorder)
    a = tmp_path / "a.md"
    a.write_text("x", encoding="utf-8")
    watcher.run(max_polls=1)
    assert recorder.processed == [str(a)]

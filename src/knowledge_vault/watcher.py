"""vault 文件监听：标准库轮询实现（零新依赖）。

设计定稿：
- **PollingWatcher**：固定间隔（默认 5s）``os.walk`` 扫描 ``KV_VAULT_ROOT``，
  与上一轮快照对比得出 创建/修改/删除/移动 四类变更 → 交给摄入管线。
  **不引入第三方依赖**；watchdog 留作后续可选升级（接口形态不变，仅换实现）。
- 变更判据：rel 路径集合差分；修改 = 路径仍在但 ``(size, mtime_ns)`` 变化；
  移动 = 同一轮里"消失"与"新增"存在 ``(size, mtime_ns)`` 完全一致的配对
  （同文件系统内 move 保留 mtime；跨盘复制-删除会被拆成 删除+新建 两事件，
  结果仍然正确，只是多一次重建）。
- 隐藏目录/隐藏文件与 ``__pycache__`` 默认不进管线（与 pipeline.process_all
  口径一致）；排除集可配置。
- 事件分发：``on_event`` 回调（观察者，总是先调）→ ``pipeline`` 的
  ``process_file`` / ``handle_deleted`` / ``handle_moved``（鸭子类型，
  不强制 import，便于测试注入 fake）。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PollingWatcher", "ChangeEvent", "DEFAULT_EXCLUDE_DIRS", "FileStamp"]

#: 事件类别
CREATED = "created"
MODIFIED = "modified"
DELETED = "deleted"
MOVED = "moved"

#: 默认排除目录（与 pipeline.process_all 的隐藏目录口径一致）
DEFAULT_EXCLUDE_DIRS = frozenset(
    {".git", ".obsidian", ".trash", "__pycache__", ".venv", "node_modules"}
)

#: 快照条目：(size_bytes, mtime_ns)
FileStamp = tuple[int, int]


@dataclass(frozen=True)
class ChangeEvent:
    """一次轮询得出的一个文件变更。

    ``path`` 为绝对路径；``old_path`` 仅 moved 事件有值（移动前路径）。
    """

    kind: str
    path: str
    old_path: str | None = None


class PollingWatcher:
    """轮询式 vault 监听器（标准库实现）。

    用法::

        watcher = PollingWatcher(vault_root, pipeline=pipeline)
        watcher.run()                # 阻塞轮询；Ctrl-C 退出
        events = watcher.poll_once() # 或手动单步轮询
    """

    def __init__(
        self,
        vault_root: str | Path,
        *,
        interval: float = 5.0,
        exclude_hidden: bool = True,
        exclude_dirs: frozenset[str] | set[str] | None = None,
        pipeline: object | None = None,
        on_event: Callable[[ChangeEvent], None] | None = None,
    ) -> None:
        if interval < 0:
            raise ValueError(f"interval 不能为负，收到 {interval}")
        self.vault_root = Path(vault_root)
        self.interval = float(interval)
        self.exclude_hidden = bool(exclude_hidden)
        self.exclude_dirs = (
            DEFAULT_EXCLUDE_DIRS if exclude_dirs is None else set(exclude_dirs)
        )
        self.pipeline = pipeline
        self.on_event = on_event
        # 构造时取基线快照：构造之后到首轮 poll 之间的变化才会成事件
        self._last: dict[str, FileStamp] = self.scan()

    # ------------------------------------------------------------- 扫描

    def scan(self) -> dict[str, FileStamp]:
        """当前快照：rel 路径 → (size, mtime_ns)（排除规则在此生效）。"""
        root = self.vault_root
        if not root.exists():
            return {}
        snapshot: dict[str, FileStamp] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            if self.exclude_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
            for name in filenames:
                if self.exclude_hidden and name.startswith("."):
                    continue
                full = Path(dirpath) / name
                try:
                    stat = full.stat()
                except OSError:  # 轮询间隙文件消失：跳过，下一轮自然按删除处理
                    continue
                snapshot[str(full.relative_to(root))] = (
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
        return snapshot

    # ------------------------------------------------------------- 差分

    def poll_once(self) -> list[ChangeEvent]:
        """与上一轮快照对比一次，返回变更事件列表（并推进基线）。"""
        current = self.scan()
        last = self._last
        self._last = current

        events: list[ChangeEvent] = []
        last_keys = set(last)
        current_keys = set(current)

        created_keys = current_keys - last_keys
        deleted_keys = last_keys - current_keys
        modified_keys = {
            key
            for key in last_keys & current_keys
            if last[key] != current[key]
        }

        # 移动配对启发式：消失与新增中 (size, mtime_ns) 完全一致的项
        stamps_to_created: dict[FileStamp, list[str]] = {}
        for key in created_keys:
            stamps_to_created.setdefault(current[key], []).append(key)
        moved_pairs: list[tuple[str, str]] = []
        for old_key in sorted(deleted_keys):
            candidates = stamps_to_created.get(last[old_key])
            if candidates:
                new_key = candidates.pop(0)
                created_keys.discard(new_key)
                deleted_keys.discard(old_key)
                moved_pairs.append((old_key, new_key))

        for old_key, new_key in sorted(moved_pairs):
            events.append(
                ChangeEvent(
                    kind=MOVED,
                    path=self._abs(new_key),
                    old_path=self._abs(old_key),
                )
            )
        for key in sorted(created_keys):
            events.append(ChangeEvent(kind=CREATED, path=self._abs(key)))
        for key in sorted(modified_keys):
            events.append(ChangeEvent(kind=MODIFIED, path=self._abs(key)))
        for key in sorted(deleted_keys):
            events.append(ChangeEvent(kind=DELETED, path=self._abs(key)))
        return events

    # ------------------------------------------------------------- 分发

    def dispatch(self, events: list[ChangeEvent]) -> None:
        """把事件交给 on_event 回调与管线（鸭子类型，均可缺省）。"""
        for event in events:
            if self.on_event is not None:
                self.on_event(event)
            if self.pipeline is None:
                continue
            if event.kind in (CREATED, MODIFIED):
                self.pipeline.process_file(event.path)
            elif event.kind == DELETED:
                self.pipeline.handle_deleted(event.path)
            elif event.kind == MOVED:
                self.pipeline.handle_moved(event.old_path, event.path)

    def run(self, *, max_polls: int | None = None) -> int:
        """阻塞轮询循环（``Ctrl-C`` 退出）；``max_polls`` 供测试/单步驱动。

        返回实际轮询轮数。
        """
        polls = 0
        while max_polls is None or polls < max_polls:
            events = self.poll_once()
            self.dispatch(events)
            polls += 1
            if max_polls is None or polls < max_polls:
                time.sleep(self.interval)
        return polls

    # ------------------------------------------------------------- 内部

    def _abs(self, rel_key: str) -> str:
        return str(self.vault_root / rel_key)

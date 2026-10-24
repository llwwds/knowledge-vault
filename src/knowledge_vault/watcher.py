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
- **受控写入目录整棵子树排除**（``exclude_roots``）：``KV_WRITABLE_ROOT`` 在
  部署上是 vault 内某子目录的挂载点，其写入已由认证 API 自带摄入管线；
  重复摄入会竞态并产生重复 chunk。路径前缀在绑定挂载下不可靠，因此按
  ``(st_dev, st_ino)`` 目录身份排除。
- 事件分发：``on_event`` 回调（观察者，总是先调）→ ``pipeline`` 的
  ``process_file`` / ``handle_deleted`` / ``handle_moved``（鸭子类型，
  不强制 import，便于测试注入 fake）。
- **WatcherService**：watcher + 摄入栈的装配与容错轮询循环（``kv-watch``
  与 ``kv-serve`` 内嵌监听共用），单轮异常只记日志、绝不退出循环。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import VaultConfig, writable_root
from .pipeline import open_pipeline

__all__ = [
    "PollingWatcher",
    "WatcherService",
    "ChangeEvent",
    "DEFAULT_EXCLUDE_DIRS",
    "DEFAULT_WATCH_INTERVAL",
    "ENV_WATCH_INTERVAL",
    "FileStamp",
    "resolve_interval",
]

#: 事件类别
CREATED = "created"
MODIFIED = "modified"
DELETED = "deleted"
MOVED = "moved"

#: 默认排除目录（与 pipeline.process_all 的隐藏目录口径一致）
DEFAULT_EXCLUDE_DIRS = frozenset(
    {".git", ".obsidian", ".trash", "__pycache__", ".venv", "node_modules"}
)

#: 轮询间隔（秒）：环境变量与默认值
ENV_WATCH_INTERVAL = "KV_WATCH_INTERVAL"
DEFAULT_WATCH_INTERVAL = 30.0

logger = logging.getLogger(__name__)

#: 快照条目：(size_bytes, mtime_ns)
FileStamp = tuple[int, int]


def _dir_identity(path: str | Path) -> tuple[int, int] | None:
    """目录身份 ``(st_dev, st_ino)``；不存在或无权限返回 ``None``。

    用 inode 身份而非路径前缀配对，是因为容器里受控目录是 vault 子目录的
    绑定挂载（``/managed-notes`` 与 ``/obsidian-vault/...`` 路径不同、inode 相同）。
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino)


def resolve_interval(interval: float | None = None) -> float:
    """轮询间隔：显式参数 > ``KV_WATCH_INTERVAL`` > 30s；非法值告警回落默认。"""
    if interval is not None:
        value = float(interval)
    else:
        value = DEFAULT_WATCH_INTERVAL
        raw = os.environ.get(ENV_WATCH_INTERVAL)
        if raw:
            try:
                value = float(raw)
            except ValueError:
                logger.warning(
                    "%s=%r 不是数字，回落 %gs", ENV_WATCH_INTERVAL, raw, DEFAULT_WATCH_INTERVAL
                )
    if value <= 0:
        raise ValueError(f"轮询间隔必须为正数，收到 {value}")
    return value


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
        exclude_roots: Iterable[str | Path] | None = None,
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
        # 整棵子树排除的目录（受控写入目录）；按 inode 身份比对，见 _excluded_ids
        self.exclude_roots = tuple(Path(p).expanduser() for p in (exclude_roots or ()))
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
        excluded = self._excluded_ids()
        snapshot: dict[str, FileStamp] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            if self.exclude_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
            if excluded:
                dirnames[:] = [
                    d for d in dirnames
                    if _dir_identity(os.path.join(dirpath, d)) not in excluded
                ]
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

    def _excluded_ids(self) -> frozenset[tuple[int, int]]:
        """本轮要排除的子树目录身份集合（空集 ＝ 不排除）。

        每轮重算而不是构造时缓存：受控目录可能晚于 watcher 创建（首写才建），
        且目录身份不变，每轮一次 ``stat`` 的开销可忽略。
        受控目录就是 vault 根本身时视为配置错误，不排除任何东西（否则监听停摆）。
        """
        if not self.exclude_roots:
            return frozenset()
        vault_id = _dir_identity(self.vault_root)
        return frozenset(
            identity
            for identity in (_dir_identity(root) for root in self.exclude_roots)
            if identity is not None and identity != vault_id
        )

    # ------------------------------------------------------------- 差分

    def reset_baseline(self) -> None:
        """清空上一轮快照：下一轮把所有现存文件视为新建（单轮全量对账用）。"""
        self._last = {}

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


# ----------------------------------------------------------------- 服务装配


class WatcherService:
    """watcher + 摄入栈的装配与容错轮询（``kv-watch`` 与 ``kv-serve`` 共用）。

    - 自动排除 ``KV_WRITABLE_ROOT`` 整棵子树：受控目录物理上在 vault 内部，
      其写入已由认证 API 自带摄入管线，重复摄入会竞态并产生重复 chunk。
    - 单轮 poll / 单条事件处理的异常都只记日志，循环（含 serve 守护线程）不退出。
    - ``pipeline`` 缺省由 :func:`~knowledge_vault.pipeline.open_pipeline` 自建，
      句柄在所属线程收尾；注入替身时不动其生命周期。
    - **基线即时、管线惰性**：变更基线必须在构造时拍下（之后的变化才算事件），
      而登记库 / pipeline.db 的 SQLite 连接有线程亲和性，自建栈只能在真正跑
      轮询的线程里装（见 :meth:`_bind_pipeline`）。
    """

    def __init__(
        self,
        config: VaultConfig,
        *,
        interval: float | None = None,
        pipeline: object | None = None,
    ) -> None:
        self.config = config
        self.interval = resolve_interval(interval)
        self._handles = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        controlled = writable_root()
        # 注入替身时管线就位；否则先留空，等跑轮询的线程里再 _bind_pipeline
        self.watcher = PollingWatcher(
            config.vault_root,
            interval=self.interval,
            exclude_roots=[controlled] if controlled is not None else [],
            pipeline=pipeline,
        )

    # ------------------------------------------------------------- 单轮

    def _bind_pipeline(self) -> bool:
        """在跑轮询的线程里装配摄入栈（幂等）；失败记日志并等下回重试。"""
        if self.watcher.pipeline is not None:
            return True
        try:
            self._handles = open_pipeline(self.config)
        except Exception:  # noqa: BLE001 - 装配失败不得杀死轮询循环
            logger.exception("摄入栈装配失败，下一轮重试")
            return False
        self.watcher.pipeline = self._handles.pipeline
        return True

    def poll_once(self) -> int:
        """跑一轮并分发事件，返回本轮事件数；任何异常都不外抛。"""
        if not self._bind_pipeline():
            return 0
        try:
            events = self.watcher.poll_once()
        except Exception:  # noqa: BLE001 - 单轮扫描异常不得杀死轮询循环
            logger.exception("vault 轮询失败，下一轮重试")
            return 0
        for event in events:
            try:
                self.watcher.dispatch([event])
            except Exception:  # noqa: BLE001 - 单个文件失败不阻塞其余事件
                logger.exception("变更处理失败：%s %s", event.kind, event.path)
        return len(events)

    def run(self, *, once: bool = False) -> int:
        """阻塞轮询循环，返回累计处理的事件数（:meth:`stop` 置位后结束）。

        ``once=True``（``kv-watch --once``）＝一轮全量对账：先清空基线让现存
        文件全部按新建过一遍（未变的靠 ``content_hash`` 跳过），再清扫登记层
        活跃但磁盘已消失的文件；受控写入目录照样排除。
        """
        stop = self._stop_event
        try:
            if once:
                self.watcher.reset_baseline()
            total = 0
            rounds = 0
            while not stop.is_set():
                started = time.monotonic()
                changed = self.poll_once()
                elapsed = time.monotonic() - started
                total += changed
                rounds += 1
                if changed:
                    logger.info(
                        "vault 轮询 #%d：%d 个变更，耗时 %.3fs", rounds, changed, elapsed
                    )
                else:
                    logger.debug("vault 轮询 #%d：无变更，耗时 %.3fs", rounds, elapsed)
                if once:
                    total += self.sweep_deleted()
                    break
                stop.wait(self.interval)  # stop 置位时立即返回，关机不必等满一轮
            return total
        finally:
            # 句柄必须在装配它的线程里关闭（SQLite 连接不可跨线程）
            self._close_handles()

    def sweep_deleted(self) -> int:
        """对账清扫：登记层活跃但磁盘已消失的文件 → ``handle_deleted``。

        只在自建摄入栈（持有登记库句柄）时生效，注入替身时返回 0。
        """
        store = self._handles.store if self._handles is not None else None
        if store is None:
            return 0
        vanished = [
            doc.file_path
            for doc in store.iterate_files(include_deleted=False)
            if not Path(doc.file_path).exists()
        ]
        swept = 0
        for path in vanished:
            try:
                self.watcher.dispatch([ChangeEvent(kind=DELETED, path=path)])
            except Exception:  # noqa: BLE001 - 单个文件失败不影响其余清扫
                logger.exception("清扫已消失文件失败：%s", path)
                continue
            swept += 1
        if swept:
            logger.info("vault 对账清扫：%d 个已消失文件已软删", swept)
        return swept

    # ------------------------------------------------------------- 后台线程

    def start(self) -> threading.Thread:
        """起守护线程跑轮询（``kv-serve`` 内嵌监听）；异常已在 :meth:`poll_once` 兜住。"""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.run, name="kv-vault-watch", daemon=True
            )
            self._thread.start()
        return self._thread

    def stop(self, *, timeout: float = 10.0) -> None:
        """通知后台线程退出并等它收尾（句柄由该线程的 ``run`` 收尾关闭）。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "vault 监听线程未在 %.1fs 内退出（句柄随进程退出回收）", timeout
                )
                return
            self._thread = None
        self.close()

    def close(self) -> None:
        """关闭自建的摄入栈句柄（幂等；注入 pipeline 时为空操作）。

        仅当句柄由调用线程装配（未走过 :meth:`start`）时才真正关闭；守护线程
        自己已在 ``run`` 的 ``finally`` 里关过，此处天然幂等。
        """
        if self._closed:
            return
        self._closed = True
        self._close_handles()

    def _close_handles(self) -> None:
        handles, self._handles = self._handles, None
        if handles is not None:
            handles.close()

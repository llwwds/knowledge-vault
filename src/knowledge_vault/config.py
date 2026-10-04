"""路径与运行参数配置。

所有外部注入点走环境变量，代码内只保留默认值：

- ``KV_STATE_DIR``     登记层 SQLite / 后续向量库等运行时产物的根目录
                       （默认 ``~/llwwds_application/knowledge-vault/state``，expanduser）
- ``KV_VAULT_ROOT``    原文件区根目录（默认 ``~/Documents/knowledge-vault_file``）
- ``KV_USERDICT``      jieba 用户词典路径（默认取包内打包的 ``data/userdict.txt``；
                       置为空字符串表示不加载任何用户词典）
- ``KV_FILE_ID_SEED``  file_id 分配器起始种子（默认 1；快照批量导入时可抬高起点）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping

ENV_STATE_DIR = "KV_STATE_DIR"
ENV_VAULT_ROOT = "KV_VAULT_ROOT"
ENV_USERDICT = "KV_USERDICT"
ENV_FILE_ID_SEED = "KV_FILE_ID_SEED"

DEFAULT_STATE_DIR = "~/llwwds_application/knowledge-vault/state"
DEFAULT_VAULT_ROOT = "~/Documents/knowledge-vault_file"
DEFAULT_FILE_ID_SEED = 1


def packaged_userdict_path() -> Path:
    """包内打包的 jieba 用户词典的真实文件系统路径。

    索引侧与查询侧必须加载同一份词典文件（见 docs/spike-5-jieba-vocab.md），
    生产路径默认即本文件；如需替换，用 ``KV_USERDICT`` 指向同形态的词典。
    """
    with resources.as_file(
        resources.files("knowledge_vault").joinpath("data", "userdict.txt")
    ) as path:
        return path


@dataclass(frozen=True)
class VaultConfig:
    """一次运行的全部注入配置。"""

    state_dir: Path
    vault_root: Path
    userdict_path: Path | None
    file_id_seed: int = DEFAULT_FILE_ID_SEED

    @property
    def db_path(self) -> Path:
        """登记层 SQLite 数据库文件路径（位于 state_dir 下）。"""
        return self.state_dir / "kv.sqlite3"


def load_config(env: Mapping[str, str] | None = None) -> VaultConfig:
    """从环境映射构造配置；``env=None`` 时读 ``os.environ``。

    - 三个路径类变量都做 expanduser（默认值带 ``~``）。
    - ``KV_USERDICT`` 为空字符串 → ``userdict_path=None``（显式禁用用户词典）。
    - ``KV_FILE_ID_SEED`` 必须是整数，非法值直接抛 ``ValueError``。
    """
    env = os.environ if env is None else env

    state_dir = Path(env.get(ENV_STATE_DIR, DEFAULT_STATE_DIR)).expanduser()
    vault_root = Path(env.get(ENV_VAULT_ROOT, DEFAULT_VAULT_ROOT)).expanduser()

    userdict_raw = env.get(ENV_USERDICT)
    if userdict_raw is None:
        userdict_path: Path | None = packaged_userdict_path()
    elif userdict_raw == "":
        userdict_path = None
    else:
        userdict_path = Path(userdict_raw).expanduser()

    seed_raw = env.get(ENV_FILE_ID_SEED)
    file_id_seed = int(seed_raw) if seed_raw else DEFAULT_FILE_ID_SEED

    return VaultConfig(
        state_dir=state_dir,
        vault_root=vault_root,
        userdict_path=userdict_path,
        file_id_seed=file_id_seed,
    )

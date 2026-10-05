"""路径与运行参数配置。

所有外部注入点走环境变量，代码内只保留默认值：

- ``KV_STATE_DIR``     登记层 SQLite / 后续向量库等运行时产物的根目录
                       （默认 ``~/llwwds_application/knowledge-vault/state``，expanduser）
- ``KV_VAULT_ROOT``    原文件真源根目录（默认 ``~/Documents/obsidian_file``，
                       即用户 Obsidian 仓库原位置——单真源，系统对其零写入）
- ``KV_EXCLUDE_DIRS``  用户级排除目录（相对 vault_root 的目录路径，逗号分隔，
                       如 ``99 废纸篓,02 项目/某子目录``）；与各使用方的内置
                       业务排除（隐藏目录/.git 等）叠加生效
- ``KV_USERDICT``      jieba 用户词典路径（默认取包内打包的 ``data/userdict.txt``；
                       置为空字符串表示不加载任何用户词典）
- ``KV_FILE_ID_SEED``  file_id 分配器起始种子（默认 1；批量导入时可抬高起点）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Mapping

ENV_STATE_DIR = "KV_STATE_DIR"
ENV_VAULT_ROOT = "KV_VAULT_ROOT"
ENV_EXCLUDE_DIRS = "KV_EXCLUDE_DIRS"
ENV_USERDICT = "KV_USERDICT"
ENV_FILE_ID_SEED = "KV_FILE_ID_SEED"

DEFAULT_STATE_DIR = "~/llwwds_application/knowledge-vault/state"
DEFAULT_VAULT_ROOT = "~/Documents/obsidian_file"
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
    exclude_dirs: tuple[str, ...] = ()

    @property
    def db_path(self) -> Path:
        """登记层 SQLite 数据库文件路径（位于 state_dir 下）。"""
        return self.state_dir / "kv.sqlite3"


def load_config(env: Mapping[str, str] | None = None) -> VaultConfig:
    """从环境映射构造配置；``env=None`` 时读 ``os.environ``。

    - 路径类变量都做 expanduser（默认值带 ``~``）。
    - ``KV_EXCLUDE_DIRS`` 逗号分隔的相对目录路径，剥空白；空段忽略。
    - ``KV_USERDICT`` 为空字符串 → ``userdict_path=None``（显式禁用用户词典）。
    - ``KV_FILE_ID_SEED`` 必须是整数，非法值直接抛 ``ValueError``。
    """
    env = os.environ if env is None else env

    state_dir = Path(env.get(ENV_STATE_DIR, DEFAULT_STATE_DIR)).expanduser()
    vault_root = Path(env.get(ENV_VAULT_ROOT, DEFAULT_VAULT_ROOT)).expanduser()

    exclude_raw = env.get(ENV_EXCLUDE_DIRS, "")
    exclude_dirs = tuple(
        part.strip().strip("/")
        for part in exclude_raw.split(",")
        if part.strip()
    )

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
        exclude_dirs=exclude_dirs,
    )

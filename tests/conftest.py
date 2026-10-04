"""pytest 夹具：全部使用 tmp_path 隔离与合成数据，不触碰真实 state/vault。"""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_vault import Store, VaultConfig, packaged_userdict_path


@pytest.fixture()
def config(tmp_path: Path) -> VaultConfig:
    """标准测试配置：state/vault 都在 tmp_path 下；userdict 用包内打包词典。"""
    return VaultConfig(
        state_dir=tmp_path / "state",
        vault_root=tmp_path / "vault",
        userdict_path=packaged_userdict_path(),
        file_id_seed=1,
    )


@pytest.fixture()
def store(config: VaultConfig):
    s = Store(config=config)
    yield s
    s.close()


@pytest.fixture()
def conn(store: Store):
    return store.conn


@pytest.fixture()
def tokenizer(store: Store):
    return store.tokenizer


@pytest.fixture()
def make_file(tmp_path: Path):
    """在 tmp_path 下造文本文件的工厂（合成数据，绝不指向真实 vault）。"""

    def _make(name: str, content: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    return _make

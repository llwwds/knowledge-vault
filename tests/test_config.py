"""配置注入：KV_* 环境变量、默认值、expanduser、包内 userdict。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from knowledge_vault import Store
from knowledge_vault.config import (
    DEFAULT_FILE_ID_SEED,
    ENV_FILE_ID_SEED,
    ENV_STATE_DIR,
    ENV_USERDICT,
    ENV_VAULT_ROOT,
    load_config,
    packaged_userdict_path,
)


class TestDefaults:
    def test_paths_expanduser(self):
        cfg = load_config(env={})
        assert cfg.state_dir == Path(
            "~/llwwds_application/knowledge-vault/state"
        ).expanduser()
        assert cfg.vault_root == Path("~/Documents/knowledge-vault_file").expanduser()
        assert "~" not in str(cfg.state_dir)
        assert "~" not in str(cfg.vault_root)

    def test_default_seed(self):
        assert load_config(env={}).file_id_seed == DEFAULT_FILE_ID_SEED == 1

    def test_default_userdict_is_packaged_file(self):
        packaged = packaged_userdict_path()
        assert packaged.is_file()
        content = packaged.read_text(encoding="utf-8")
        # 词典随包打包、非空、含核心栈词（userdict_v0 的拷贝）
        assert "zvec 100000 n" in content
        assert "bge-m3 100000 n" in content
        assert load_config(env={}).userdict_path == packaged

    def test_db_path_under_state_dir(self):
        cfg = load_config(env={ENV_STATE_DIR: "/tmp/any-state"})
        assert cfg.db_path == Path("/tmp/any-state/kv.sqlite3")


class TestEnvInjection:
    def test_all_vars(self):
        env = {
            ENV_STATE_DIR: "/tmp/kv-state",
            ENV_VAULT_ROOT: "/tmp/kv-vault",
            ENV_USERDICT: "/tmp/my-userdict.txt",
            ENV_FILE_ID_SEED: "1000",
        }
        cfg = load_config(env=env)
        assert cfg.state_dir == Path("/tmp/kv-state")
        assert cfg.vault_root == Path("/tmp/kv-vault")
        assert cfg.userdict_path == Path("/tmp/my-userdict.txt")
        assert cfg.file_id_seed == 1000

    def test_paths_expanduser_from_env(self):
        cfg = load_config(env={ENV_STATE_DIR: "~/kv-state-x"})
        assert cfg.state_dir == Path("~/kv-state-x").expanduser()

    def test_userdict_empty_string_disables(self):
        assert load_config(env={ENV_USERDICT: ""}).userdict_path is None

    def test_seed_unset_keeps_default(self):
        assert load_config(env={}).file_id_seed == 1

    def test_invalid_seed_raises(self):
        with pytest.raises(ValueError):
            load_config(env={ENV_FILE_ID_SEED: "not-a-number"})

    def test_reads_os_environ_by_default(self, monkeypatch):
        monkeypatch.setenv(ENV_STATE_DIR, "/tmp/env-read-state")
        monkeypatch.setenv(ENV_FILE_ID_SEED, "7")
        cfg = load_config()
        assert cfg.state_dir == Path("/tmp/env-read-state")
        assert cfg.file_id_seed == 7

    def test_env_seed_flows_into_store(self, tmp_path):
        cfg = load_config(
            env={ENV_STATE_DIR: str(tmp_path / "state"), ENV_FILE_ID_SEED: "500"}
        )
        with Store(config=cfg) as store:
            doc = store.register_file(
                _touch(tmp_path / "seeded.md"),
            )
        assert doc.file_id == 500


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("seed", encoding="utf-8")
    return path


class TestVaultConfig:
    def test_replace_keeps_frozen_dataclass(self, tmp_path):
        cfg = load_config(env={ENV_STATE_DIR: str(tmp_path)})
        seeded = replace(cfg, file_id_seed=42)
        assert seeded.file_id_seed == 42
        assert cfg.file_id_seed == 1

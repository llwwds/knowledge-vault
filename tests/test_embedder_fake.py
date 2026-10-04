"""embedder 测试：FakeEmbedder 全覆盖 + BGEM3Embedder 懒加载契约；真实模型只留
1 个 slow 冒烟（默认跳过，见模块尾部的 skip 条件说明）。"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from knowledge_vault.embedder import (
    BGE_M3_DIM,
    DEFAULT_EMBED_MODEL,
    BGEM3Embedder,
    Embedder,
    FakeEmbedder,
    bge_m3_cache_available,
    l2_normalize,
)


# ------------------------------------------------------------ 归一化工具

def test_l2_normalize_rows_unit():
    vecs = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(vecs)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_l2_normalize_zero_row_stays_zero():
    vecs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    out = l2_normalize(vecs)
    assert np.allclose(out[0], 0.0)
    assert not np.isnan(out).any()


def test_l2_normalize_rejects_non_2d():
    with pytest.raises(ValueError):
        l2_normalize(np.ones(4, dtype=np.float32))


# ---------------------------------------------------------- FakeEmbedder

def test_fake_embedder_shape_dtype_and_norm():
    emb = FakeEmbedder()
    out = emb.encode(["文本一", "text two", "文本三"])
    assert out.shape == (3, BGE_M3_DIM)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_fake_embedder_deterministic_across_instances():
    a = FakeEmbedder().encode(["同一文本"])
    b = FakeEmbedder().encode(["同一文本"])
    assert np.array_equal(a, b)


def test_fake_embedder_seed_changes_vectors():
    a = FakeEmbedder(seed="s1").encode(["文本"])
    b = FakeEmbedder(seed="s2").encode(["文本"])
    assert not np.array_equal(a, b)


def test_fake_embedder_distinct_texts_differ():
    emb = FakeEmbedder(dim=64)
    a, b = emb.encode(["Alpha 文本", "Beta 文本"])
    assert not np.allclose(a, b)
    # 归一化后余弦（点积）远离 1（哈希敏感：稍有差异即近乎正交）
    assert float(a @ b) < 0.5


def test_fake_embedder_batch_equals_single():
    emb = FakeEmbedder(dim=32)
    batch = emb.encode(["甲", "乙", "丙"])
    single = np.stack([emb.encode([t])[0] for t in ["甲", "乙", "丙"]])
    assert np.allclose(batch, single)


def test_fake_embedder_empty_input():
    out = FakeEmbedder(dim=16).encode([])
    assert out.shape == (0, 16)
    assert out.dtype == np.float32


def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_embedder_custom_dim():
    assert FakeEmbedder(dim=8).encode(["x"]).shape == (1, 8)


# --------------------------------------------------------- BGEM3Embedder

def test_bgem3_default_calibration_from_spike():
    """macOS spike-6 口径：use_fp16 显式 False 由实现保证；batch=8 / max_length=1024。"""
    emb = BGEM3Embedder()
    assert emb.model_name == DEFAULT_EMBED_MODEL
    assert emb.batch_size == 8
    assert emb.max_length == 1024
    assert emb.threads == 8
    assert emb.dim == BGE_M3_DIM


def test_bgem3_is_lazy_construction_imports_no_torch():
    """构造不触发 torch/FlagEmbedding import（首次 encode 才加载）。"""
    BGEM3Embedder()
    assert "torch" not in sys.modules
    assert "FlagEmbedding" not in sys.modules


def test_bgem3_empty_encode_no_model_load():
    """空列表短路，不加载模型。"""
    out = BGEM3Embedder().encode([])
    assert out.shape == (0, BGE_M3_DIM)


# ------------------------------------------------------------- 模型缓存

def test_bge_m3_cache_available_reads_hf_home(tmp_path, monkeypatch):
    hub = tmp_path / "hub" / "models--BAAI--bge-m3"
    hub.mkdir(parents=True)
    assert bge_m3_cache_available({"HF_HOME": str(tmp_path)})
    assert not bge_m3_cache_available({"HF_HOME": str(tmp_path / "other")})


def test_bge_m3_cache_available_default_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/nonexistent-kv-test")
    assert bge_m3_cache_available() is False


# --------------------------------------------- 真实模型 slow 冒烟（默认跳过）
#
# 说明：真实模型在容器内加载数分钟，为不拖慢常规测试套件，除「缓存缺失
# 跳过」外还需显式 opt-in：设 KV_SLOW_EMBED=1 才执行（pytest -m slow 选中）。


@pytest.mark.slow
@pytest.mark.skipif(
    not bge_m3_cache_available(),
    reason="bge-m3 HF 缓存缺失（HF_HOME/hub/models--BAAI--bge-m3 不存在）",
)
def test_bgem3_real_model_smoke():
    """真实 bge-m3 冒烟：输出 (1, 1024) float32 且已归一化。

    缓存在位的前提下仍默认跳过（未设 KV_SLOW_EMBED=1），避免常规套件变慢。
    """
    import os

    if os.environ.get("KV_SLOW_EMBED") != "1":
        pytest.skip("未设 KV_SLOW_EMBED=1，真实模型冒烟默认跳过")
    emb = BGEM3Embedder()
    out = emb.encode(["知识库切块的 embedding 冒烟测试。"])
    assert out.shape == (1, BGE_M3_DIM)
    assert out.dtype == np.float32
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-5

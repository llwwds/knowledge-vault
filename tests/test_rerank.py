"""rerank 模块测试：FakeReranker 确定性 + FlagRerankerImpl 惰性加载契约。

FlagRerankerImpl 的真实加载用 monkeypatch 的假 FlagEmbedding 模块验证（不碰
网络/模型）；真实模型只有 1 个 @pytest.mark.slow 冒烟，缓存缺失即跳过。
"""

from __future__ import annotations

import os
import sys
import threading
import types

import pytest

from knowledge_vault.protocols import Reranker
from knowledge_vault.rerank import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RERANKER_MODEL,
    FakeReranker,
    FlagRerankerImpl,
)


# ----------------------------------------------------------------- 缓存探测


def reranker_cache_available(env: dict[str, str] | None = None) -> bool:
    """BAAI/bge-reranker-v2-m3 的 HF 缓存是否已就绪（slow 冒烟的 skipif 判据）。

    语义与阶段2 ``bge_m3_cache_available`` 一致：检查 ``HF_HOME``（或默认
    ``~/.cache/huggingface``）下 ``hub/models--BAAI--bge-reranker-v2-m3`` 是否存在。
    """
    env = os.environ if env is None else env
    hf_home = env.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.isdir(
        os.path.join(hf_home, "hub", "models--BAAI--bge-reranker-v2-m3")
    )


def test_cache_probe_reads_hf_home(tmp_path):
    (tmp_path / "hub" / "models--BAAI--bge-reranker-v2-m3").mkdir(parents=True)
    assert reranker_cache_available({"HF_HOME": str(tmp_path)})
    assert not reranker_cache_available({"HF_HOME": str(tmp_path / "other")})


def test_cache_probe_default_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/nonexistent-kv-test")
    assert reranker_cache_available() is False


# --------------------------------------------------------------- FakeReranker


class TestFakeReranker:
    def test_protocol_conformance(self):
        assert isinstance(FakeReranker(), Reranker)

    def test_scores_by_char_overlap(self):
        rr = FakeReranker()
        scores = rr.score("机器学习", ["机器学习是人工智能的分支", "今天天气很好"])
        assert scores[0] > scores[1]
        assert all(isinstance(s, float) for s in scores)

    def test_offset_shifts_scores(self):
        assert FakeReranker(offset=2.5).score("q", ["q"])[0] == 3.5

    def test_calls_recorded(self):
        rr = FakeReranker()
        rr.score("查询", ["文本一", "文本二"])
        assert rr.calls == [("查询", ("文本一", "文本二"))]

    def test_empty_texts(self):
        assert FakeReranker().score("q", []) == []


# ----------------------------------------------------- FlagRerankerImpl（假模块）


class _FakeFlagReranker:
    """替身 FlagReranker：记录构造参数，compute_score 返回预置分数。"""

    instantiations: list[dict] = []
    next_scores: list[float] = [0.5, 0.9]

    def __init__(self, model_name, use_fp16=False, batch_size=8, max_length=None, **kw):
        _FakeFlagReranker.instantiations.append(
            {
                "model_name": model_name,
                "use_fp16": use_fp16,
                "batch_size": batch_size,
                "max_length": max_length,
            }
        )

    def compute_score(self, pairs, batch_size=None):
        assert all(len(pair) == 2 for pair in pairs), "必须成对 (query, text)"
        return list(_FakeFlagReranker.next_scores)


@pytest.fixture()
def fake_flagembedding(monkeypatch):
    """把假 FlagEmbedding 塞进 sys.modules，并重置构造计数。"""
    _FakeFlagReranker.instantiations = []
    module = types.ModuleType("FlagEmbedding")
    module.FlagReranker = _FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)
    return _FakeFlagReranker


class TestFlagRerankerImplLazyLoad:
    def test_lazy_not_loaded_on_construction(self):
        impl = FlagRerankerImpl()
        assert impl.loaded is False
        assert impl.model_name == DEFAULT_RERANKER_MODEL

    def test_empty_texts_never_loads(self, fake_flagembedding):
        impl = FlagRerankerImpl()
        assert impl.score("q", []) == []
        assert impl.loaded is False
        assert fake_flagembedding.instantiations == []

    def test_score_loads_once_and_pairs_query_text(self, fake_flagembedding):
        impl = FlagRerankerImpl()
        scores = impl.score("查询", ["甲", "乙"])
        assert impl.loaded is True
        assert fake_flagembedding.instantiations == [
            {
                "model_name": DEFAULT_RERANKER_MODEL,
                "use_fp16": False,  # 库默认路径即 fp32
                "batch_size": DEFAULT_BATCH_SIZE,
                "max_length": 8192,
            }
        ]
        assert scores == [0.5, 0.9]
        # 再次调用不重复加载
        impl.score("查询", ["丙"])
        assert len(fake_flagembedding.instantiations) == 1

    def test_single_text_scalar_coerced_to_list(self, fake_flagembedding):
        fake_flagembedding.next_scores = [0.7]
        impl = FlagRerankerImpl()
        assert impl.score("q", ["唯一文本"]) == [0.7]

    def test_concurrent_first_calls_load_once(self, fake_flagembedding):
        impl = FlagRerankerImpl()
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            impl.score("q", ["文本"])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(fake_flagembedding.instantiations) == 1

    def test_custom_model_name(self, fake_flagembedding):
        impl = FlagRerankerImpl(model_name="BAAI/other", batch_size=4, max_length=512)
        impl.score("q", ["t"])
        assert fake_flagembedding.instantiations[0]["model_name"] == "BAAI/other"
        assert fake_flagembedding.instantiations[0]["batch_size"] == 4
        assert fake_flagembedding.instantiations[0]["max_length"] == 512


# --------------------------------------------- 真实模型 slow 冒烟（默认跳过）
#
# 与阶段2 embedder 的约定一致：缓存缺失跳过；缓存在位也需显式
# KV_SLOW_RERANK=1 opt-in（pytest -m slow 选中），避免拖慢常规套件。


@pytest.mark.slow
@pytest.mark.skipif(
    not reranker_cache_available(),
    reason="bge-reranker-v2-m3 HF 缓存缺失（HF_HOME/hub/models--BAAI--bge-reranker-v2-m3 不存在）",
)
def test_flag_reranker_real_model_smoke():
    """真实 bge-reranker-v2-m3 冒烟：相关文本得分应高于无关文本。"""
    if os.environ.get("KV_SLOW_RERANK") != "1":
        pytest.skip("未设 KV_SLOW_RERANK=1，真实模型冒烟默认跳过")
    impl = FlagRerankerImpl()
    scores = impl.score(
        "机器学习的基本概念",
        ["机器学习是人工智能的一个分支，通过数据训练模型。", "今天午饭吃了红烧肉和青菜。"],
    )
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    assert scores[0] > scores[1]

"""rerank：交叉编码器重排（BAAI/bge-reranker-v2-m3）。

- :class:`FlagRerankerImpl`：真实实现，满足 :class:`~knowledge_vault.protocols.Reranker`
  协议（鸭子类型，不 import Protocol 也能注入）。**惰性加载**：首次 ``score()``
  才 import FlagEmbedding 并加载模型；库默认路径即 fp32（``use_fp16=False``），
  成对 ``(query, text)`` 打分，batch=8。推理用进程内互斥锁串行化（CPU 场景
  并发无益，且避免底层 torch 并发问题）。
- :class:`FakeReranker`：确定性测试替身——按 query/text 字符交集大小打分，
  无任何模型依赖；记录调用入参供断言。

分数语义统一为「越大越相关」；``compute_score`` 的原始 logits 单调等价。
"""

from __future__ import annotations

import threading

__all__ = ["FlagRerankerImpl", "FakeReranker"]

#: 默认模型（与 docs/spike-2-reranker-cpu.md 基准一致）
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 8


class FlagRerankerImpl:
    """FlagEmbedding.FlagReranker 的惰性封装。

    参数与 :class:`FlagReranker` 对齐：``model_name``、``use_fp16``（库默认
    路径即 fp32，这里显式默认 False）、``batch_size``。模型实例首次
    :meth:`score` / :meth:`preload` 时才创建。
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        use_fp16: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = 8192,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """模型是否已加载（惰性加载完成前为 False）。"""
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:  # 双重检查：并发首调只加载一次
                    from FlagEmbedding import FlagReranker  # 惰性 import，缩短无 rerank 路径的启动

                    self._model = FlagReranker(
                        self._model_name,
                        use_fp16=self._use_fp16,
                        batch_size=self._batch_size,
                        max_length=self._max_length,
                    )
        return self._model

    def preload(self) -> None:
        """显式预热（可在服务启动期调用，避免首个请求承担加载耗时）。"""
        self._ensure_model()

    def score(self, query: str, texts: list[str]) -> list[float]:
        """成对打分：``score[i]`` 对应 ``texts[i]``，越大越相关。"""
        if not texts:
            return []
        model = self._ensure_model()
        pairs = [[query, text] for text in texts]
        with self._infer_lock:
            scores = model.compute_score(pairs, batch_size=self._batch_size)
        if isinstance(scores, (int, float)):  # 单条时 FlagEmbedding 返回标量
            return [float(scores)]
        return [float(s) for s in scores]


class FakeReranker:
    """确定性测试替身：``score = |set(query) ∩ set(text)| + offset``。

    中文字符级重叠即够测试排序用；``offset`` 可整体平移分数（验证分数值
    透传而非仅序）。所有调用入参记录在 :attr:`calls`。
    """

    def __init__(self, *, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        q = set(query)
        return [float(len(q & set(text))) + self.offset for text in texts]

"""跨阶段冻结的组件协议（阶段2 / 阶段3 的对接点）。

Embedder / Reranker 以 :class:`typing.Protocol` 定义，运行时一律依赖注入：

- 测试注入 fake 实现（无模型依赖）；
- 阶段2 的 BGEM3Embedder 按鸭子类型满足 :class:`Embedder`，
  阶段3 的 ``FlagRerankerImpl`` 按鸭子类型满足 :class:`Reranker`；
- 双方**互不 import 对方实现**，只认本协议。

契约（冻结，勿改签名）：

- :meth:`Embedder.encode` ``→ np.ndarray``，shape ``(N, 1024)``、dtype float32、
  每行 L2 归一化，行序与输入一致；
- :meth:`Reranker.score` ``→ list[float]``，与 ``texts`` 等长，越大越相关。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

#: Embedder 输出向量维度（BGE-M3 dense 出口）
EMBEDDING_DIM = 1024


@runtime_checkable
class Embedder(Protocol):
    """查询 / 文本 → 归一化向量。

    实现要求：返回 ``(N, dim)`` 的 ``np.ndarray``（float32），每行 L2 归一化；
    失败时直接抛异常，由调用方决定是否降级（向量路关闭）。
    """

    def encode(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Reranker(Protocol):
    """查询-文本交叉编码打分（rerank 阶段）。

    实现要求：对 ``(query, text)`` 逐对打分，返回与 ``texts`` 等长的
    ``list[float]``，分数越大越相关。线程安全由实现方自行保证。
    """

    def score(self, query: str, texts: list[str]) -> list[float]: ...

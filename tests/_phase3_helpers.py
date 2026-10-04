"""阶段3 测试共用替身与合成知识库（非测试模块，pytest 不收集）。

全部使用合成数据；不读 testdata、不触真实 vault/state。
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from knowledge_vault.retrieve import VectorHit


# ------------------------------------------------------------------ 替身


class FakeEmbedder:
    """协议替身：查表 → 单位向量（one-hot 即满足 L2 归一化契约）。

    ``table``: 文本 → 维度下标；未登记文本返回零向量（余弦无意义，仅用于
    「可编码」语义）。
    """

    def __init__(self, table: dict[str, int], dim: int = 16) -> None:
        self.table = table
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            idx = self.table.get(text)
            if idx is not None:
                vecs[i, idx % self.dim] = 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vecs / norms).astype(np.float32)


class FakeVectorPath:
    """向量路替身：固定排名，忽略真实向量（排名契约就是它的全部接口）。

    记录每次调用的 ``(topk)`` 与向量范数，供断言 embedder→向量路确实接通。
    """

    def __init__(self, ranking: Sequence[VectorHit]) -> None:
        self.ranking = list(ranking)
        self.calls: list[int] = []

    def search(self, vector: np.ndarray, *, topk: int) -> list[VectorHit]:
        self.calls.append(topk)
        assert vector.ndim == 1, "查询向量应为一维"
        return list(self.ranking[:topk])


class FakeRerankerByLength:
    """按「文本越短分越高」重排的确定性替身（可稳定翻转 FTS 序）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [-float(len(text)) for text in texts]


class FakeRerankerBadArity:
    """返回长度错误的替身（验证 rerank 契约校验）。"""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0] * max(0, len(texts) - 1)


# ------------------------------------------------------------------ 合成知识库


def build_kb(store, make_file: Callable[..., object]) -> dict:
    """合成知识库（5 文件 + 2 边），返回 file_id 与 chunk 文本索引。

    布局：
    - a「机器学习笔记.md」 status=now, tags=[ml, ai]，chunks:
      c1=机器学习是人工智能的一个分支 / c2=深度学习是机器学习的子集
    - b「散步日记.md」 tags=[life]，chunks: c1=今天天气很好适合出去散步
    - c「训练数据.md」 tags=[ml]，chunks: c1=机器学习的模型需要训练数据
    - d「已删除文件.md」 tags=[ml]，chunks: c1=机器学习的删除样本，登记后软删除
    - large「超大文件.bin」 is_large=True, tags=[ml]，summary chunk=机器学习超大文件的摘要
      + 常规 chunk（验证大文件只出 summary）
    边：a→c (link)、a→b (link)、c→b (mention)
    """
    f_a = make_file("机器学习笔记.md", "占位：正文不入索引层判断")
    f_b = make_file("散步日记.md", "占位")
    f_c = make_file("训练数据.md", "占位")
    f_d = make_file("已删除文件.md", "占位")
    f_large = make_file("超大文件.bin", "占位")

    doc_a = store.register_file(f_a, status="now", context_tag=["ml", "ai"])
    doc_b = store.register_file(f_b, context_tag=["life"])
    doc_c = store.register_file(f_c, context_tag=["ml"])
    doc_d = store.register_file(f_d, context_tag=["ml"])
    doc_large = store.register_file(f_large, context_tag=["ml"], is_large=True)

    store.add_chunks(
        doc_a.file_id, ["机器学习是人工智能的一个分支", "深度学习是机器学习的子集"]
    )
    store.add_chunks(doc_b.file_id, ["今天天气很好适合出去散步"])
    store.add_chunks(doc_c.file_id, ["机器学习的模型需要训练数据"])
    store.add_chunks(doc_d.file_id, ["机器学习的删除样本"])
    store.add_chunks(doc_large.file_id, ["机器学习大文件正文不应被召回"])
    store.add_summary_chunk(doc_large.file_id, "机器学习超大文件的摘要")

    store.soft_delete(doc_d.file_id)

    store.add_edge(doc_a.file_id, doc_c.file_id, "link")
    store.add_edge(doc_a.file_id, doc_b.file_id, "link")
    store.add_edge(doc_c.file_id, doc_b.file_id, "mention")

    return {
        "a": doc_a.file_id,
        "b": doc_b.file_id,
        "c": doc_c.file_id,
        "d": doc_d.file_id,
        "large": doc_large.file_id,
    }


def chunk_id(ids: dict, key: str, seq: int) -> str:
    """ids 字典 → ``f{file_id}-c{seq}``。"""
    return f"f{ids[key]}-c{seq}"

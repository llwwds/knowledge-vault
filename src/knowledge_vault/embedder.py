"""embedding 接口：Embedder 协议 + BGEM3Embedder（真实模型）+ FakeEmbedder（测试）。

约定（设计定稿）：
- :class:`Embedder` 协议：``encode(texts: list[str]) -> np.ndarray``，
  返回 ``(N, dim)`` float32 且**每行已 L2 归一化**（零向量守卫：范数 0 保持 0）。
- :class:`BGEM3Embedder`：FlagEmbedding ``BGEM3FlagModel('BAAI/bge-m3',
  use_fp16=False)``（macOS spike-6 结论：库默认 fp16 路径在 CPU 上会就地转
  fp32，属"伪加速"，显式 use_fp16=False 干净加载）；batch=8、max_length=1024、
  ``torch.set_num_threads(8)``——**在首次加载模型时设置而非本模块 import 时**
  （import 时不触碰 torch，测试套件与不需要 embedding 的进程零开销）。
- :class:`FakeEmbedder`：确定性伪随机向量（sha256 派生种子 → 正态分布 → 归一化），
  供测试与离线实验复用；同文本恒同向量、不同文本几乎必然不同向量。
- 换 embedding 模型 = 新建 zvec collection 全量重建（模型名记入 collection
  的 meta.json，见 vectorstore.py）。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

#: 默认嵌入模型（HF id）；写入 vectorstore collection meta.json 的模型名字段。
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

#: bge-m3 dense 输出维度
BGE_M3_DIM = 1024

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "BGEM3Embedder",
    "DEFAULT_EMBED_MODEL",
    "BGE_M3_DIM",
    "l2_normalize",
]


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（float32；零向量保持零，不产生 NaN）。"""
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError(f"期望二维 (N, dim) 数组，收到 shape={vecs.shape}")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return (vecs / safe).astype(np.float32, copy=False)


@runtime_checkable
class Embedder(Protocol):
    """嵌入器协议：``encode`` 返回 (N, dim) float32、逐行 L2 归一化的矩阵。"""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """文本列表 → (N, dim) float32 归一化向量矩阵（顺序与输入一致）。"""
        ...


@dataclass
class FakeEmbedder:
    """确定性伪随机嵌入器（测试/离线实验用，零模型依赖）。

    向量由 ``sha256(f"{seed}:{text}")`` 派生种子生成正态分布再归一化：
    同文本恒同向量；文本稍有差异即得到（几乎必然）不同的向量，
    可稳定构造"语义相近"与"语义无关"的检索测试场景（哈希敏感）。
    """

    dim: int = BGE_M3_DIM
    seed: str = "fake"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{self.seed}:{text}".encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            rows[i] = rng.standard_normal(self.dim, dtype=np.float32)
        return l2_normalize(rows)


@dataclass
class BGEM3Embedder:
    """bge-m3 真实嵌入器（懒加载：首次 encode 才 import torch/FlagEmbedding）。

    - ``use_fp16=False`` 显式（spike-6：CPU 上 fp16 加载会被就地转 fp32，伪峰）；
    - ``torch.set_num_threads(threads)`` 在模型加载时执行（非 import 时）；
    - ``encode`` 返回的 dense 向量库内已归一化，此处仍做防御性 L2 归一化。
    """

    model_name: str = DEFAULT_EMBED_MODEL
    batch_size: int = 8
    max_length: int = 1024
    threads: int = 8
    devices: str = "cpu"
    dim: int = BGE_M3_DIM

    _model: object = None

    def _load(self) -> object:
        if self._model is None:
            import torch  # 延迟到真正需要时（避免拖慢不需要 embedding 的进程）

            torch.set_num_threads(self.threads)
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name, use_fp16=False, devices=self.devices
            )
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        out = model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
        )
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        if vecs.shape[0] != len(texts):
            raise RuntimeError(
                f"encode 返回数量不符：期望 {len(texts)}，收到 {vecs.shape[0]}"
            )
        return l2_normalize(vecs)


def bge_m3_cache_available(env: dict[str, str] | None = None) -> bool:
    """bge-m3 的 HF 缓存是否已就绪（slow 冒烟测试的 skipif 判据）。

    检查 ``HF_HOME``（或默认 ``~/.cache/huggingface``）下
    ``hub/models--BAAI--bge-m3`` 目录是否存在。
    """
    env = os.environ if env is None else env
    hf_home = env.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.isdir(os.path.join(hf_home, "hub", "models--BAAI--bge-m3"))

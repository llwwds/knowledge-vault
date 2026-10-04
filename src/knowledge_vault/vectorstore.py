"""zvec 向量库封装：collection 生命周期 + chunk 向量写入/删除 + optimize 策略。

关键决策（设计定稿与 spike-1 结论）：
- **payload 只存 file_id / chunk_seq / kind**，正文不进向量库（正文真源在
  登记 SQLite 的 chunks 表；两侧用同一 ``chunk_id_for`` 主键格式对齐）。
- doc 主键复用 :func:`knowledge_vault.textindex.chunk_id_for`（``f{fid}-c{seq}``），
  保证 FTS 侧与向量侧可互相对照。
- 向量由 Embedder 保证已 L2 归一化后写入（BYO 向量；COSINE 度量下尺度不变，
  这里不做二次归一化）。
- **optimize 低频攒批**（spike-1：optimize 是重 compaction，绝不能每次插入后调）：
  :meth:`ZvecStore.optimize_if_needed` 按阈值策略触发——新增未 optimize 条数
  ≥ ``every``（默认 50000）**或** 最老未 optimize 写入距今 > ``hours`` 小时
  （默认 24h）；阈值状态持久化在 pipeline.db（由调用方注入
  :class:`OptimizeStateStore` 协议实现，即 pipeline 的 :class:`PipelineState`；
  不注入则仅内存计数，进程重启清零）。
- **换 embedding 模型 = 新建 collection 全量重建**：模型名与维度写入
  collection 目录旁 ``meta.json``；:meth:`ZvecStore.open` 校验维度（可选校验
  模型名），不匹配即拒绝打开。
- collection 目录属索引层，放 ``KV_STATE_DIR`` 下（默认 ``state/zvec_chunks/``）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import zvec
from zvec import (
    CollectionSchema,
    DataType,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    MetricType,
    Query,
    VectorSchema,
)

from .embedder import BGE_M3_DIM, DEFAULT_EMBED_MODEL
from .textindex import chunk_id_for

__all__ = [
    "ZvecStore",
    "VectorHit",
    "OptimizeState",
    "OptimizeStateStore",
    "ZvecStoreError",
]

#: optimize 攒批默认阈值（可用参数覆盖；与 pipeline 的 KV_OPTIMIZE_* 环境变量对齐）
DEFAULT_OPTIMIZE_EVERY = 50_000
DEFAULT_OPTIMIZE_HOURS = 24.0

_DEFAULT_HNSW_M = 16
_DEFAULT_EF_CONSTRUCTION = 200

_META_FILENAME = "meta.json"
_SCHEMA_VERSION = 1


class ZvecStoreError(RuntimeError):
    """zvec collection 状态不满足操作前提（元数据不匹配、未初始化等）。"""


@dataclass(frozen=True)
class VectorHit:
    """一条向量召回。``score`` 为 zvec COSINE 度量的余弦距离
    （= 1 − cosine，**越小越相关**，自召回 ≈ 0；探针实证）。"""

    doc_id: str
    file_id: int
    chunk_seq: int
    kind: str | None
    score: float


@dataclass(frozen=True)
class OptimizeState:
    """optimize 攒批状态（持久化在 pipeline.db，由协议实现方读写）。"""

    pending_count: int
    oldest_pending_at: float | None  # 最老未 optimize 写入的 epoch 秒
    last_optimize_at: float | None


class OptimizeStateStore(Protocol):
    """optimize 阈值状态的持久化协议（pipeline.db 的 PipelineState 实现）。"""

    def get_optimize_state(self) -> OptimizeState: ...

    def add_pending_optimize(self, n: int, *, now: float | None = None) -> None: ...

    def mark_optimized(self, *, now: float | None = None) -> None: ...


def _utc_iso(ts: float | None = None) -> str:
    dt = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts is not None
        else datetime.now(timezone.utc)
    )
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


class ZvecStore:
    """单 collection（chunks）向量库的薄封装。

    生命周期::

        vs = ZvecStore.create(path, model_name="BAAI/bge-m3")
        vs.insert_chunks(file_id, seqs, vecs, kinds)
        vs.delete_file(file_id)
        vs.optimize_if_needed(state=pipeline_state)
        vs.close()
    """

    COLLECTION_NAME = "chunks"
    VEC_FIELD = "embedding"

    def __init__(
        self,
        coll: object,
        path: str | Path,
        meta: dict,
        *,
        optimize_state: OptimizeStateStore | None = None,
    ) -> None:
        self._coll = coll
        self.path = Path(path)
        self._meta = dict(meta)
        self._optimize_state = optimize_state
        # 内存侧攒批计数（与持久化状态双向同步；无持久化时独立使用）
        st = optimize_state.get_optimize_state() if optimize_state is not None else None
        self._pending: int = st.pending_count if st is not None else 0
        self._oldest: float | None = st.oldest_pending_at if st is not None else None

    # ---------------------------------------------------------- 生命周期

    def attach_optimize_state(self, store: OptimizeStateStore) -> None:
        """绑定（或更换）optimize 攒批状态的持久化实现，并以其为准同步内存计数。

        组合根（:class:`~knowledge_vault.pipeline.IngestPipeline`）在构造时调用
        本方法，把 pipeline.db 设为阈值状态的唯一真源；绑定后 ``_record_pending``
        与 ``mark_optimized`` 都直接落到该实现，避免"内存计数在涨、阈值检查读库"
        的两账本漂移。
        """
        self._optimize_state = store
        st = store.get_optimize_state()
        self._pending = st.pending_count
        self._oldest = st.oldest_pending_at

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        dim: int = BGE_M3_DIM,
        model_name: str = DEFAULT_EMBED_MODEL,
        hnsw_m: int = _DEFAULT_HNSW_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        optimize_state: OptimizeStateStore | None = None,
    ) -> "ZvecStore":
        """新建 collection（目录必须不存在）并写入 meta.json。"""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"collection 目录已存在（换模型请全量重建到新目录）: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = CollectionSchema(
            name=cls.COLLECTION_NAME,
            fields=[
                # file_id 是按文件删除/过滤的主路径，加倒排索引（spike-1 结论）
                FieldSchema(
                    "file_id",
                    DataType.INT64,
                    index_param=InvertIndexParam(enable_range_optimization=True),
                ),
                FieldSchema("chunk_seq", DataType.INT32),
                FieldSchema("kind", DataType.STRING),
            ],
            vectors=[
                VectorSchema(
                    cls.VEC_FIELD,
                    data_type=DataType.VECTOR_FP32,
                    dimension=dim,
                    index_param=HnswIndexParam(
                        metric_type=MetricType.COSINE,
                        m=hnsw_m,
                        ef_construction=ef_construction,
                    ),
                )
            ],
        )
        coll = zvec.create_and_open(str(path), schema)
        meta = {
            "collection_name": cls.COLLECTION_NAME,
            "model_name": model_name,
            "dim": int(dim),
            "metric": "cosine",
            "hnsw": {"m": hnsw_m, "ef_construction": ef_construction},
            "schema_version": _SCHEMA_VERSION,
            "created_at": _utc_iso(),
        }
        store = cls(coll, path, meta, optimize_state=optimize_state)
        store._write_meta()
        return store

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_model: str | None = None,
        dim: int | None = None,
        optimize_state: OptimizeStateStore | None = None,
    ) -> "ZvecStore":
        """打开既有 collection；meta 缺失/不匹配（维度、可选模型名）即拒绝。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"collection 目录不存在: {path}")
        meta_path = path / _META_FILENAME
        if not meta_path.exists():
            raise ZvecStoreError(f"meta.json 缺失（非本模块创建的目录）: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        want_dim = dim if dim is not None else int(meta.get("dim", BGE_M3_DIM))
        if int(meta.get("dim", -1)) != want_dim:
            raise ZvecStoreError(
                f"维度不匹配：meta.dim={meta.get('dim')}，期望 {want_dim}"
            )
        if expected_model is not None and meta.get("model_name") != expected_model:
            raise ZvecStoreError(
                f"模型不匹配：meta.model_name={meta.get('model_name')!r}，"
                f"期望 {expected_model!r}（换模型须全量重建）"
            )
        coll = zvec.open(str(path))
        return cls(coll, path, meta, optimize_state=optimize_state)

    def _write_meta(self) -> None:
        (self.path / _META_FILENAME).write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @property
    def meta(self) -> dict:
        """collection 元数据（模型名/维度/HNSW 参数/创建时间）的只读副本。"""
        return dict(self._meta)

    def update_meta(self, **kwargs) -> None:
        """合并写回 meta.json（如记录 optimize 时间等运维字段）。"""
        self._meta.update(kwargs)
        self._write_meta()

    def close(self) -> None:
        """关闭 collection（flush 未落盘写入并释放文件锁）。"""
        self._coll.close()

    def __enter__(self) -> "ZvecStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------ 写入侧

    def insert_chunks(
        self,
        file_id: int,
        chunk_seqs: Sequence[int],
        vectors: np.ndarray,
        kinds: Sequence[str],
    ) -> int:
        """插入一个文件的 chunk 向量（自动按 zvec 单次 ≤1024 上限分批）。

        ``vectors`` 为 (N, dim) float32（由 Embedder 保证归一化）；
        ``kinds`` 与 seqs 一一对应（``chunk`` / ``summary``）。
        返回成功写入条数（任何一条失败即抛错）。
        """
        return self._write(file_id, chunk_seqs, vectors, kinds, upsert=False)

    def upsert_chunks(
        self,
        file_id: int,
        chunk_seqs: Sequence[int],
        vectors: np.ndarray,
        kinds: Sequence[str],
    ) -> int:
        """按主键覆盖写入（语义同 :meth:`insert_chunks`，已存在 id 即更新）。"""
        return self._write(file_id, chunk_seqs, vectors, kinds, upsert=True)

    def _write(
        self,
        file_id: int,
        chunk_seqs: Sequence[int],
        vectors: np.ndarray,
        kinds: Sequence[str],
        *,
        upsert: bool,
    ) -> int:
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(chunk_seqs):
            raise ValueError(
                f"vectors shape {vecs.shape} 与 chunk_seqs 数量 {len(chunk_seqs)} 不符"
            )
        if vecs.shape[0] and vecs.shape[1] != int(self._meta["dim"]):
            raise ValueError(
                f"向量维度 {vecs.shape[1]} 与 collection dim {self._meta['dim']} 不符"
            )
        if len(kinds) != len(chunk_seqs):
            raise ValueError("kinds 与 chunk_seqs 数量不一致")
        if not chunk_seqs:
            return 0

        write = self._coll.upsert if upsert else self._coll.insert
        total = 0
        batch = 1024  # zvec 单次 Insert/Upsert 上限 1024（spike-1 结论）
        for start in range(0, len(chunk_seqs), batch):
            docs = [
                zvec.Doc(
                    id=chunk_id_for(file_id, int(seq)),
                    vectors={self.VEC_FIELD: vecs[start + offset]},
                    fields={
                        "file_id": int(file_id),
                        "chunk_seq": int(seq),
                        "kind": str(kind),
                    },
                )
                for offset, (seq, kind) in enumerate(
                    zip(
                        chunk_seqs[start : start + batch],
                        kinds[start : start + batch],
                    )
                )
            ]
            statuses = write(docs)
            bad = [s for s in statuses if not s.ok()]
            if bad:
                raise ZvecStoreError(
                    f"写入失败 {len(bad)} 条（file_id={file_id}）: {bad[0].message()}"
                )
            total += len(docs)
        self._record_pending(total)
        return total

    def _record_pending(self, n: int, *, now: float | None = None) -> None:
        if n <= 0:
            return
        now = time.time() if now is None else now
        if self._pending == 0:
            self._oldest = now
        self._pending += n
        if self._optimize_state is not None:
            self._optimize_state.add_pending_optimize(n, now=now)

    def delete_file(self, file_id: int) -> int:
        """删除某文件全部向量（按 file_id 过滤删除），返回删除条数。

        删除不产生 optimize 债务（攒批计数只统计新增写入）；compaction
        收益与碎片清理统一交给下一次按阈值的 optimize。

        注意（zvec 实证语义）：``delete_by_filter`` 立即从可检索集合移除 doc
        （count 马上下降），但会留下**墓碑**——在下次 optimize 物理回收之前，
        对同一 doc id 再走 :meth:`insert_chunks` 会报 already exists。
        因此"删旧后按同 id 重写"的场景（文件被修改）必须用
        :meth:`upsert_chunks`（upsert 可覆盖墓碑 id，见 pipeline 的用法）；
        :meth:`insert_chunks` 只用于全新 id。
        """
        before = self.count()
        self._coll.delete_by_filter(f"file_id = {int(file_id)}")
        return max(before - self.count(), 0)

    # ------------------------------------------------------------ 运维侧

    def count(self) -> int:
        """当前 doc 总数。"""
        return int(self._coll.stats.doc_count)

    def flush(self) -> None:
        """强制落盘（不触发 compaction）。"""
        self._coll.flush()

    def optimize(self) -> None:
        """手动 flush + optimize（重 compaction，属低频运维动作），并清零攒批状态。"""
        self._coll.flush()
        self._coll.optimize()
        self._pending = 0
        self._oldest = None
        if self._optimize_state is not None:
            self._optimize_state.mark_optimized()

    def optimize_if_needed(
        self,
        state: OptimizeStateStore | None = None,
        *,
        every: int = DEFAULT_OPTIMIZE_EVERY,
        hours: float = DEFAULT_OPTIMIZE_HOURS,
        now: float | None = None,
    ) -> bool:
        """按阈值策略判断并执行 optimize，返回是否触发。

        触发条件（pending = 自上次 optimize 以来的新增写入条数）：
        - ``pending >= every``（默认 50000）；**或**
        - ``pending > 0`` 且最老未 optimize 写入距今 ≥ ``hours`` 小时（默认 24h）。

        ``state`` 缺省用构造时注入的持久化状态；两者皆无则用内存计数
        （进程重启清零，仅建议测试/临时进程使用）。
        """
        store = state if state is not None else self._optimize_state
        now = time.time() if now is None else now
        if store is not None:
            st = store.get_optimize_state()
            pending, oldest = st.pending_count, st.oldest_pending_at
        else:
            pending, oldest = self._pending, self._oldest
        if pending <= 0:
            return False
        overdue = oldest is not None and (now - oldest) >= hours * 3600.0
        if pending >= every or overdue:
            self.optimize()
            return True
        return False

    # ------------------------------------------------------------ 查询侧

    def search(
        self,
        vector: np.ndarray,
        *,
        topk: int = 10,
        file_id: int | None = None,
        ef: int = 64,
    ) -> list[VectorHit]:
        """向量近邻查询（score = 1 − cosine，越小越相关）。"""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != int(self._meta["dim"]):
            raise ValueError(
                f"查询向量维度 {vec.shape[0]} 与 collection dim {self._meta['dim']} 不符"
            )
        query = Query(
            field_name=self.VEC_FIELD, vector=vec, param=HnswQueryParam(ef=int(ef))
        )
        results = self._coll.query(
            queries=query,
            topk=int(topk),
            filter=(f"file_id = {int(file_id)}" if file_id is not None else None),
        )
        hits: list[VectorHit] = []
        for doc in results:
            hits.append(
                VectorHit(
                    doc_id=str(doc.id),
                    file_id=int(doc.field("file_id")),
                    chunk_seq=int(doc.field("chunk_seq")),
                    kind=doc.field("kind"),
                    score=float(doc.score),
                )
            )
        return hits

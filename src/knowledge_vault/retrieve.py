"""召回管线：登记层过滤 → 全文/向量召回 → RRF 融合 → 图扩展 → rerank → top-j。

固定顺序（设计定稿，勿重排）：

1. **登记层过滤**：按 ``status`` / ``context_tag``（请求标签与文档标签数组
   交集非空即命中）/ ``file_id`` 集合，在 documents 表上圈定「可召回文件集」；
   ``deleted_at`` 非空的文件一律排除——登记层本身不过滤（见 registry 模块
   docstring），过滤发生在召回层。
2. **三路召回**：
   - 路① 全文：``Store.search``（FTS5 bm25，jieba+userdict+span 预处理由
     Tokenizer 承担），按 bm25 升序（相关度降序）即 rank 序；
   - 路② 向量：查询文本 → ``Embedder.encode([query])`` → 向量路句柄 KNN。
     句柄以「已打开的 zvec collection 适配器 / 测试 fake」注入，二者都给出
     才启用，任一缺省即降级关闭（不参与融合）；
   - 路③ 图：**不是独立入口**——融合后 top 候选 file_id 经 ``graph.expand``
     做 1-2 跳扩展，扩展命中文件的「代表 chunk」（summary 优先，否则最小
     seq）以固定低权重并入 RRF 名单；已在名单中的候选获得该权重的加成。
3. **RRF(k=60)**：``score = Σ 1/(k + rank_i)``，各路 rank 独立（从 1 起）。
4. **rerank**：注入 :class:`~knowledge_vault.protocols.Reranker`，对融合后前
   ``rerank_candidates`` 个候选取 chunks 表原文成对打分，重排分数即最终分数；
   未注入时最终分数 = 融合分。
5. **top-j**：按最终分数降序截断（分页）。

过滤补充语义：``is_large`` 文件只有 ``kind='summary'`` 的 chunk 可召回。
文本一律按 chunk_id 回 chunks 表取（不回真源现切）；chunks 表中已不存在的
候选（陈旧索引项）直接丢弃。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np
import zvec

from .protocols import Embedder, Reranker
from .store import Store
from .textindex import chunk_id_for

if TYPE_CHECKING:  # 仅供类型检查；zvec 运行时已真实导入（pyproject 依赖）
    from zvec import Collection, HnswQueryParam

__all__ = [
    "VectorHit",
    "VectorPath",
    "ZvecVectorIndex",
    "RetrievalResult",
    "RetrievalOutput",
    "rrf_fuse",
    "retrieve",
]

#: 来源标签的规范顺序（sources 元组按此排序）
SOURCE_ORDER = ("fts", "vector", "graph")

#: SQLite IN 子句单批参数上限（防变量数超限）
_IN_BATCH = 500


# ------------------------------------------------------------------ 向量路


@dataclass(frozen=True)
class VectorHit:
    """向量路单条命中（payload 与 zvec collection 约定对齐）。"""

    file_id: int
    chunk_seq: int
    kind: str
    score: float  # zvec 返回的相似度，越大越相关

    @property
    def chunk_id(self) -> str:
        return chunk_id_for(self.file_id, self.chunk_seq)


@runtime_checkable
class VectorPath(Protocol):
    """向量路句柄协议：注入已打开的 zvec collection 适配器或测试 fake。

    契约：``search(vector, topk=N)`` 返回按相似度降序、最多 N 条的
    :class:`VectorHit` 列表。登记层过滤（status/tag/deleted）由召回层负责，
    句柄不做过滤。
    """

    def search(self, vector: np.ndarray, *, topk: int) -> list[VectorHit]: ...


class ZvecVectorIndex:
    """zvec Collection → :class:`VectorPath` 适配器（zvec 0.7.0 API）。

    collection 由摄入管线（阶段2）建好并打开后注入本适配器；约定向量字段
    ``embedding``，payload 字段 ``file_id``(INT64) / ``chunk_seq``(INT64) /
    ``kind``(STRING)。

    ``query_param`` 必须与向量字段的索引类型匹配（zvec 会校验）：
    HNSW 索引用 ``HnswQueryParam(ef=300)``，FLAT 缺省索引用 ``None``。
    """

    def __init__(
        self,
        collection: "Collection",
        *,
        field_name: str = "embedding",
        query_param: "HnswQueryParam | None" = None,
    ) -> None:
        self._collection = collection
        self._field_name = field_name
        self._query_param = query_param

    def search(self, vector: np.ndarray, *, topk: int) -> list[VectorHit]:
        query = zvec.Query(
            field_name=self._field_name,
            vector=np.asarray(vector, dtype=np.float32),
            param=self._query_param,
        )
        docs = self._collection.query(
            queries=query,
            topk=topk,
            output_fields=["file_id", "chunk_seq", "kind"],
        )
        return [
            VectorHit(
                file_id=int(doc.field("file_id")),
                chunk_seq=int(doc.field("chunk_seq")),
                kind=str(doc.field("kind")),
                score=float(doc.score),
            )
            for doc in docs
        ]


# --------------------------------------------------------------------- RRF


def rrf_fuse(rankings: Sequence[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion：``score = Σ_paths 1/(k + rank)``（rank 从 1 起）。

    ``rankings`` 的每个元素是一路按相关度降序排列的候选键列表；同一路内重复
    键只计一次（首次出现的名次），同一键出现在多路时贡献相加。
    返回 ``键 → 融合分``（只含有分的键）。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


# ----------------------------------------------------------------- 结果结构


@dataclass(frozen=True)
class RetrievalResult:
    """一条召回结果（top-j 中的一项）。"""

    file_id: int
    chunk_id: str
    chunk_seq: int
    kind: str
    score: float  # 最终分数：rerank 分（有 reranker）或融合分
    fused_score: float  # RRF 融合分（rerank 前的排序依据，供溯源/调试）
    sources: tuple[str, ...]  # 命中来源，SOURCE_ORDER 的子集
    file_path: str
    title: str
    text: str

    def as_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "chunk_id": self.chunk_id,
            "chunk_seq": self.chunk_seq,
            "kind": self.kind,
            "score": self.score,
            "fused_score": self.fused_score,
            "sources": list(self.sources),
            "file_path": self.file_path,
            "title": self.title,
            "text": self.text,
        }


@dataclass(frozen=True)
class RetrievalOutput:
    """一次召回的完整输出。"""

    query: str
    items: list[RetrievalResult]
    meta: dict

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "items": [item.as_dict() for item in self.items],
            "meta": dict(self.meta),
        }


# ----------------------------------------------------------------- 内部结构


@dataclass(frozen=True)
class _FileMeta:
    """documents 表过滤投影（召回层登记过滤的最小字段集）。"""

    file_id: int
    is_large: bool
    context_tag: tuple[str, ...]
    status: str
    file_path: str
    title: str


@dataclass
class _Candidate:
    """融合名单中的一条候选（rerank / 组装前）。"""

    chunk_id: str
    file_id: int
    chunk_seq: int
    kind: str
    fused_score: float
    sources: set[str]


# ----------------------------------------------------------------- 登记层过滤


def _active_files(conn: sqlite3.Connection) -> dict[int, _FileMeta]:
    """活跃文件（``deleted_at IS NULL``）的元数据投影，file_id → meta。

    软删除排除在这一步完成；status / tag / file_ids 的细分过滤在
    :func:`_allowed_files` 里做（纯 Python，个人库量级无需 SQL 端优化）。
    """
    rows = conn.execute(
        "SELECT file_id, is_large, context_tag, status, file_path, title "
        "FROM documents WHERE deleted_at IS NULL"
    ).fetchall()
    metas: dict[int, _FileMeta] = {}
    for row in rows:
        raw_tags = row["context_tag"]
        metas[int(row["file_id"])] = _FileMeta(
            file_id=int(row["file_id"]),
            is_large=bool(row["is_large"]),
            context_tag=tuple(json.loads(raw_tags)) if raw_tags else (),
            status=row["status"],
            file_path=row["file_path"],
            title=row["title"],
        )
    return metas


def _allowed_files(
    conn: sqlite3.Connection,
    *,
    status: str | None,
    context_tag: Sequence[str] | None,
    file_ids: Sequence[int] | None,
) -> dict[int, _FileMeta]:
    """圈定可召回文件集。

    - ``deleted_at`` 非空一律排除（登记层不过滤、召回层过滤的定稿语义）；
    - ``status`` 非空时精确匹配（now|library）；
    - ``context_tag`` 非空时按「请求标签 ∩ 文档标签 ≠ ∅」过滤（交集语义）；
    - ``file_ids`` 非 None 时限定集合（注意：空列表 = 不允许任何文件，
      与 None = 不限制 语义不同）。
    """
    wanted_tags = set(context_tag) if context_tag else None
    wanted_ids = set(file_ids) if file_ids is not None else None
    allowed: dict[int, _FileMeta] = {}
    for fid, meta in _active_files(conn).items():
        if status is not None and meta.status != status:
            continue
        if wanted_ids is not None and fid not in wanted_ids:
            continue
        if wanted_tags is not None and not wanted_tags.intersection(meta.context_tag):
            continue
        allowed[fid] = meta
    return allowed


# ------------------------------------------------------------------ 候选取材


def _fetch_chunks(
    conn: sqlite3.Connection, chunk_ids: Sequence[str]
) -> dict[str, sqlite3.Row]:
    """按 chunk_id 批量回 chunks 表取行（分批 IN，防变量数超限）。"""
    out: dict[str, sqlite3.Row] = {}
    ids = list(chunk_ids)
    for start in range(0, len(ids), _IN_BATCH):
        batch = ids[start : start + _IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            "SELECT chunk_id, file_id, chunk_seq, kind, text "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            batch,
        )
        for row in rows:
            out[row["chunk_id"]] = row
    return out


def _representative_chunk(
    conn: sqlite3.Connection, file_id: int
) -> tuple[str, int, str] | None:
    """图扩展命中文件的代表 chunk：summary 优先，否则最小 chunk_seq。

    返回 ``(chunk_id, chunk_seq, kind)``；文件尚无任何 chunk（仅登记元数据）
    时返回 None。
    """
    row = conn.execute(
        "SELECT chunk_id, chunk_seq, kind FROM chunks WHERE file_id = ? "
        "ORDER BY CASE WHEN kind = 'summary' THEN 0 ELSE 1 END, chunk_seq LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return None
    return row["chunk_id"], int(row["chunk_seq"]), row["kind"]


def _recallable(allowed: dict[int, _FileMeta], file_id: int, kind: str) -> bool:
    """候选可召回判定：文件在允许集内，且大文件（is_large）只放行 summary。"""
    meta = allowed.get(file_id)
    if meta is None:
        return False
    if meta.is_large and kind != "summary":
        return False
    return True


# -------------------------------------------------------------------- 主流程


def retrieve(
    store: Store,
    query: str,
    *,
    embedder: Embedder | None = None,
    vector_path: VectorPath | None = None,
    reranker: Reranker | None = None,
    status: str | None = None,
    context_tag: Sequence[str] | None = None,
    file_ids: Sequence[int] | None = None,
    top_j: int = 10,
    k: int = 60,
    per_path_topk: int = 50,
    rerank_candidates: int = 50,
    graph: bool = True,
    graph_max_hops: int = 1,
    graph_edge_types: Sequence[str] | None = None,
    graph_direction: str = "both",
    graph_seed_count: int = 5,
    graph_rank: int = 100,
) -> RetrievalOutput:
    """按定稿顺序执行一次召回。

    参数：
    - ``embedder`` / ``vector_path``：二者都给出才启用向量路；只给
      ``vector_path`` 不给 ``embedder`` 视为配置错误（查询无法向量化），
      抛 :class:`ValueError`。
    - ``reranker``：:class:`~knowledge_vault.protocols.Reranker` 协议实现，
      None = 跳过 rerank。
    - ``status`` / ``context_tag`` / ``file_ids``：登记层过滤（语义见
      :func:`_allowed_files`）。
    - ``top_j``：返回条数上限；``k``：RRF 常数；``per_path_topk``：FTS/向量
      各路的候选取材上限。
    - ``rerank_candidates``：送 rerank 的融合候选上限（<=0 表示不设上限）。
    - ``graph``：是否做图扩展；``graph_max_hops`` / ``graph_edge_types`` /
      ``graph_direction`` 透传 ``graph.expand``；``graph_seed_count``：取融合
      top-N file_id 做扩展种子；``graph_rank``：图命中在 RRF 名单中的固定
      名次（权重 = 1/(k+graph_rank)，默认 100 → 低于任何真实路的 rank≥1 权重）。
    """
    started = time.perf_counter()
    if top_j <= 0:
        raise ValueError(f"top_j 必须为正整数，收到 {top_j}")
    if vector_path is not None and embedder is None:
        raise ValueError(
            "注入了 vector_path 但缺少 embedder（查询无法向量化）；"
            "二者需同时注入，或都为 None 以关闭向量路"
        )

    query = (query or "").strip()
    allowed = _allowed_files(
        store.conn, status=status, context_tag=context_tag, file_ids=file_ids
    )
    meta: dict = {"allowed_files": len(allowed)}

    if not query:
        meta.update(
            fts_hits=0, vector_hits=0, graph_files=0, graph_added=0,
            graph_boosted=0, candidates=0, reranked=False, elapsed_ms=0.0,
        )
        return RetrievalOutput(query=query, items=[], meta=meta)

    chunk_file: dict[str, int] = {}  # chunk_id → file_id（图种子提取用）

    # ---- 路① 全文（bm25 升序 = 相关度降序，返回序即 rank 序）
    fts_ranking: list[str] = []
    for hit in store.search(query, limit=per_path_topk):
        if not _recallable(allowed, hit.file_id, hit.kind):
            continue
        fts_ranking.append(hit.chunk_id)
        chunk_file[hit.chunk_id] = hit.file_id
    meta["fts_hits"] = len(fts_ranking)

    # ---- 路② 向量（embedder + vector_path 双注入才启用）
    vec_ranking: list[str] = []
    if embedder is not None and vector_path is not None:
        query_vec = np.asarray(embedder.encode([query]), dtype=np.float32)[0]
        for vhit in vector_path.search(query_vec, topk=per_path_topk):
            if not _recallable(allowed, vhit.file_id, vhit.kind):
                continue
            vec_ranking.append(vhit.chunk_id)
            chunk_file[vhit.chunk_id] = vhit.file_id
    meta["vector_hits"] = len(vec_ranking)

    # ---- RRF 融合
    fused = rrf_fuse([fts_ranking, vec_ranking], k=k)

    # ---- 路③ 图扩展：融合 top file_id 为种子，邻居代表 chunk 低权重并入
    graph_members: set[str] = set()
    graph_files = 0
    graph_added = 0
    graph_boosted = 0
    if graph and fused:
        seed_ids: list[int] = []
        for cid, _score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(seed_ids) >= graph_seed_count:
                break
            fid = chunk_file.get(cid)
            if fid is not None and fid not in seed_ids:
                seed_ids.append(fid)
        graph_weight = 1.0 / (k + graph_rank)
        seen_neighbors: set[int] = set()
        for seed in seed_ids:
            result = store.expand(
                seed,
                max_hops=graph_max_hops,
                edge_types=graph_edge_types or None,
                direction=graph_direction,
            )
            for fid, _depth in result.nodes:
                if fid in allowed:
                    seen_neighbors.add(fid)
        graph_files = len(seen_neighbors)
        for fid in sorted(seen_neighbors):
            rep = _representative_chunk(store.conn, fid)
            if rep is None:
                continue
            rep_cid, _rep_seq, rep_kind = rep
            if not _recallable(allowed, fid, rep_kind):
                continue
            chunk_file[rep_cid] = fid
            graph_members.add(rep_cid)
            if rep_cid in fused:
                fused[rep_cid] += graph_weight
                graph_boosted += 1
            else:
                fused[rep_cid] = graph_weight
                graph_added += 1
    meta.update(
        graph_files=graph_files, graph_added=graph_added, graph_boosted=graph_boosted
    )

    # ---- 候选截断 + 回 chunks 表取原文（陈旧索引项丢弃）
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if rerank_candidates > 0:
        ordered = ordered[:rerank_candidates]
    rows = _fetch_chunks(store.conn, [cid for cid, _ in ordered])
    fts_set = set(fts_ranking)
    vec_set = set(vec_ranking)
    candidates: list[_Candidate] = []
    for cid, score in ordered:
        row = rows.get(cid)
        if row is None:
            continue
        sources: set[str] = set()
        if cid in fts_set:
            sources.add("fts")
        if cid in vec_set:
            sources.add("vector")
        if cid in graph_members:
            sources.add("graph")
        candidates.append(
            _Candidate(
                chunk_id=cid,
                file_id=int(row["file_id"]),
                chunk_seq=int(row["chunk_seq"]),
                kind=row["kind"],
                fused_score=score,
                sources=sources,
            )
        )
    meta["candidates"] = len(candidates)

    # ---- rerank（注入才启用；重排分数即最终分数）
    used_rerank = False
    final_scores: dict[str, float] = {c.chunk_id: c.fused_score for c in candidates}
    if reranker is not None and candidates:
        scores = reranker.score(query, [rows[c.chunk_id]["text"] for c in candidates])
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"reranker 返回 {len(scores)} 个分数，与候选数 {len(candidates)} 不一致"
            )
        final_scores = {c.chunk_id: float(s) for c, s in zip(candidates, scores)}
        used_rerank = True
    meta["reranked"] = used_rerank

    # ---- top-j（最终分降序；同分先看融合分、再看 chunk_id 保证确定性）
    final_ranked = sorted(
        final_scores.items(), key=lambda kv: (-kv[1], -fused[kv[0]], kv[0])
    )[:top_j]
    by_id = {c.chunk_id: c for c in candidates}
    items = []
    for position, (cid, score) in enumerate(final_ranked):
        c = by_id[cid]
        items.append(
            RetrievalResult(
                file_id=c.file_id,
                chunk_id=c.chunk_id,
                chunk_seq=c.chunk_seq,
                kind=c.kind,
                score=score,
                fused_score=c.fused_score,
                sources=tuple(s for s in SOURCE_ORDER if s in c.sources),
                file_path=allowed[c.file_id].file_path,
                title=allowed[c.file_id].title,
                text=rows[cid]["text"],
            )
        )
    meta["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return RetrievalOutput(query=query, items=items, meta=meta)

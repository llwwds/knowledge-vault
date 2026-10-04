"""chunks 全文索引：jieba 预分词 + FTS5（external content → chunks）。

约定（务必经由本模块读写，不要绕过）：
- ``chunks.text`` 存**原文**；FTS 索引内容 = jieba 分词后空格拼接的文本，由
  :func:`add_chunk` / :func:`add_chunks` 在写入 chunks 的同一事务内同步进
  ``chunks_fts``（external content 模式不会自动同步，也没有触发器）。
- 删除必须走 :func:`delete_chunks_for_file`：external content 模式下 FTS 的
  ``delete`` 命令需要"当初写入索引的精确字符串"，这里用确定性重分词还原。
  若写入与删除之间换了分词环境（jieba 版本/词典），可能留下幽灵索引项。
- 禁止对 ``chunks_fts`` 执行 ``'rebuild'``——那会用 chunks 原文（未分词）重建索引。
- token 构成：span 预处理（``_``/``/`` → 全角占位）→ jieba 精确模式 → 丢弃
  无任何字母数字的 token。查询侧走同一 preprocess + 分词（镜像），
  因此 ``connect_mcp`` / ``mcp__server__tool`` / ``a/b`` 两侧形态一致，
  且不会产生孤立的 ``_`` / ``/`` token。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jieba

#: spike-5 结论：``_`` 与 ``/`` 不在 jieba re_han 的可合并范围（词典原理上修不了），
#: 分词前替换为全角占位。占位字符同样不在 re_han 内 —— 只替换"孤立边界字符出现在
#: 哪个 token 里"，不改变其余文本的分块结构（边界保持）。
DEFAULT_SPAN_MAP: dict[str, str] = {"_": "＿", "/": "／"}

#: FTS5 匹配高亮/snippet 默认标记
DEFAULT_HL_OPEN = "["
DEFAULT_HL_CLOSE = "]"
DEFAULT_ELLIPSIS = "…"
DEFAULT_SNIPPET_TOKENS = 12


def make_span_preprocess(
    span_map: dict[str, str] | None = None,
) -> Callable[[str], str]:
    """构造 span 预处理函数：按 ``span_map`` 逐对做纯文本替换（默认 ``_``/``/`` → 全角）。

    返回值可直接传给 :class:`Tokenizer` 的 ``preprocess`` 参数；传空映射即恒等。
    """
    mapping = dict(DEFAULT_SPAN_MAP) if span_map is None else dict(span_map)

    def preprocess(text: str) -> str:
        for src, dst in mapping.items():
            text = text.replace(src, dst)
        return text

    return preprocess


default_preprocess: Callable[[str], str] = make_span_preprocess()


def _has_alnum(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


_loaded_userdicts: set[str] = set()


class Tokenizer:
    """写入/查询共用的分词器（userdict 进程内幂等加载，两侧形态一致）。"""

    def __init__(
        self,
        *,
        userdict_path: str | Path | None = None,
        preprocess: Callable[[str], str] | None = None,
        hmm: bool = True,
    ) -> None:
        """``userdict_path=None`` 表示不加载任何用户词典；
        ``preprocess`` 缺省用 :data:`default_preprocess`（``_``/``/`` 全角占位）。"""
        self.preprocess = default_preprocess if preprocess is None else preprocess
        self.hmm = hmm
        if userdict_path is not None:
            self.load_userdict(Path(userdict_path))

    @staticmethod
    def load_userdict(path: str | Path) -> None:
        """加载用户词典（每进程每路径只加载一次；jieba 词典本身是进程级单例）。"""
        key = str(path)
        if key in _loaded_userdicts:
            return
        jieba.load_userdict(key)
        _loaded_userdicts.add(key)

    def tokenize(self, text: str) -> list[str]:
        """原文 → token 列表：span 预处理 → jieba 精确模式 → 丢弃无字母数字的 token。"""
        if not text:
            return []
        processed = self.preprocess(text)
        return [
            tok for tok in jieba.dt.cut(processed, HMM=self.hmm) if _has_alnum(tok)
        ]

    def index_text(self, text: str) -> str:
        """原文 → 写入 FTS 的分词文本（空格拼接）。确定性：可由原文重放。"""
        return " ".join(self.tokenize(text))

    def match_query(self, text: str) -> str:
        """原文查询串 → FTS MATCH 表达式（查询侧镜像：同 preprocess、同分词）。

        每个 token 用双引号包裹成短语（token 内部的 ``-``/``.`` 等会被 unicode61
        在短语内再切分），短语之间是隐式 AND。无有效 token 时返回空串。
        """
        return " ".join(
            '"%s"' % tok.replace('"', '""') for tok in self.tokenize(text)
        )


# ------------------------------------------------------------------ chunks 写入

def chunk_id_for(file_id: int, chunk_seq: int) -> str:
    """chunk 主键格式：``f{file_id}-c{seq}``。"""
    return f"f{file_id}-c{chunk_seq}"


def add_chunk(
    conn: sqlite3.Connection,
    tokenizer: Tokenizer,
    file_id: int,
    text: str,
    *,
    chunk_seq: int | None = None,
    kind: str = "chunk",
) -> str:
    """写入单个 chunk 并在同一事务内同步 FTS，返回 chunk_id。

    ``chunk_seq`` 缺省接续该文件当前最大 seq + 1。
    """
    if kind not in ("chunk", "summary"):
        raise ValueError(f"kind 必须是 chunk|summary，收到 {kind!r}")
    if chunk_seq is None:
        row = conn.execute(
            "SELECT COALESCE(MAX(chunk_seq), 0) FROM chunks WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        chunk_seq = int(row[0]) + 1
    cid = chunk_id_for(file_id, chunk_seq)
    indexed = tokenizer.index_text(text)

    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            "INSERT INTO chunks(chunk_id, file_id, chunk_seq, kind, text, char_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, file_id, chunk_seq, kind, text, len(text)),
        )
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            (cur.lastrowid, indexed),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return cid


def add_chunks(
    conn: sqlite3.Connection,
    tokenizer: Tokenizer,
    file_id: int,
    texts: Sequence[str],
    *,
    kind: str = "chunk",
    start_seq: int | None = None,
) -> list[str]:
    """批量写入文本块（seq 连续分配，单事务），返回 chunk_id 列表。

    约定：常规 chunk 从 seq=1 起（``start_seq`` 缺省接续当前最大 seq + 1），
    seq=0 保留给 :func:`add_summary_chunk` 的文件级摘要。
    """
    if kind not in ("chunk", "summary"):
        raise ValueError(f"kind 必须是 chunk|summary，收到 {kind!r}")
    conn.execute("BEGIN IMMEDIATE")
    try:
        if start_seq is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(chunk_seq), 0) FROM chunks WHERE file_id = ?",
                (file_id,),
            ).fetchone()
            start_seq = int(row[0]) + 1
        indexed_texts = [tokenizer.index_text(text) for text in texts]
        for offset, (text, indexed) in enumerate(zip(texts, indexed_texts)):
            cid = chunk_id_for(file_id, start_seq + offset)
            cur = conn.execute(
                "INSERT INTO chunks(chunk_id, file_id, chunk_seq, kind, text, char_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cid, file_id, start_seq + offset, kind, text, len(text)),
            )
            conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                (cur.lastrowid, indexed),
            )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return [chunk_id_for(file_id, start_seq + offset) for offset in range(len(texts))]


def add_summary_chunk(
    conn: sqlite3.Connection, tokenizer: Tokenizer, file_id: int, text: str
) -> str:
    """写入文件级摘要块：约定 ``chunk_seq=0``、``kind='summary'``。"""
    return add_chunk(conn, tokenizer, file_id, text, chunk_seq=0, kind="summary")


def delete_chunks_for_file(
    conn: sqlite3.Connection, tokenizer: Tokenizer, file_id: int
) -> int:
    """删除某文件全部 chunk 并同步清理 FTS（先发 delete 命令再删行），返回删除数。

    FTS 清理依赖"用同一分词环境重放当初的索引文本"；分词环境变化可能留下
    幽灵索引项（external content 模式的固有约束，见模块 docstring）。
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            "SELECT rowid, text FROM chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
                (row["rowid"], tokenizer.index_text(row["text"])),
            )
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return len(rows)


# ------------------------------------------------------------------ 查询侧

def search_tokens(tokenizer: Tokenizer, text: str) -> str:
    """FTS MATCH 查询构造（查询侧分词镜像入口，见 :meth:`Tokenizer.match_query`）。"""
    return tokenizer.match_query(text)


@dataclass(frozen=True)
class ChunkHit:
    """一条全文命中。``score`` 为 bm25 值：越小（越负）越相关。"""

    file_id: int
    chunk_id: str
    chunk_seq: int
    kind: str
    text: str
    snippet: str
    score: float


def search(
    conn: sqlite3.Connection,
    tokenizer: Tokenizer,
    query: str,
    *,
    limit: int = 20,
    file_id: int | None = None,
    kind: str | None = None,
) -> list[ChunkHit]:
    """全文检索：MATCH + bm25 排序，可按 file_id / kind 过滤（登记层不做过滤）。"""
    match_expr = tokenizer.match_query(query)
    if not match_expr:
        return []
    sql = (
        "SELECT c.file_id, c.chunk_id, c.chunk_seq, c.kind, c.text, "
        f"snippet(chunks_fts, 0, ?, ?, ?, ?) AS snip, bm25(chunks_fts) AS score "
        "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ?"
    )
    params: list[object] = [
        DEFAULT_HL_OPEN, DEFAULT_HL_CLOSE, DEFAULT_ELLIPSIS, DEFAULT_SNIPPET_TOKENS,
        match_expr,
    ]
    if file_id is not None:
        sql += " AND c.file_id = ?"
        params.append(file_id)
    if kind is not None:
        sql += " AND c.kind = ?"
        params.append(kind)
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)
    return [
        ChunkHit(
            file_id=int(row["file_id"]),
            chunk_id=row["chunk_id"],
            chunk_seq=int(row["chunk_seq"]),
            kind=row["kind"],
            text=row["text"],
            snippet=row["snip"],
            score=float(row["score"]),
        )
        for row in conn.execute(sql, params)
    ]


def snippet(
    conn: sqlite3.Connection,
    tokenizer: Tokenizer,
    query: str,
    *,
    file_id: int | None = None,
    limit: int = 5,
    hl_open: str = DEFAULT_HL_OPEN,
    hl_close: str = DEFAULT_HL_CLOSE,
    ellipsis: str = DEFAULT_ELLIPSIS,
    tokens: int = DEFAULT_SNIPPET_TOKENS,
) -> list[str]:
    """返回带高亮标记的命中片段（在 chunks 原文上截取，标记包住命中词）。"""
    match_expr = tokenizer.match_query(query)
    if not match_expr:
        return []
    sql = (
        "SELECT snippet(chunks_fts, 0, ?, ?, ?, ?) AS snip "
        "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ?"
    )
    params: list[object] = [hl_open, hl_close, ellipsis, tokens, match_expr]
    if file_id is not None:
        sql += " AND c.file_id = ?"
        params.append(file_id)
    sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(limit)
    return [row["snip"] for row in conn.execute(sql, params)]

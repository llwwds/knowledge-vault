"""切块：md 标题层级分块 + 非 md 文本段落聚合。

语义（设计定稿）：
- md 按**标题层级**切块：一个标题小节（含其子标题内容）聚为一个 chunk；
  标题路径（如 ``一级 / 二级 / 三级``）作为 chunk 元信息**并入文本头**，
  使每个 chunk 自包含、可独立被 embedding 与 FTS 理解。
- 无标题 md 退化为段落聚合路径；非 md 文本文件一律段落聚合。
- 块参数（target≈512 token / overlap≈64 / 单文件最大 100 块）**全部可配置；
  默认值为经验近似，靠后网格实验调参**（见 :class:`ChunkerParams`）。
- token 计数用 :func:`approx_token_count` 近似（CJK 1 字 ≈ 1 token、ASCII 连续
  字母数字串 ≈ 1 token），与 bge-m3 的 XLM-R 分词器量级一致但不逐 token 精确；
  精确换算属于后续网格实验的职责。
- 超过单文件最大块数时，溢出内容**合并进最后一个 chunk**（不静默丢弃：
  FTS 仍可检索全文，embedding 侧由 max_length 截断）。
- fenced code block（``` / ~~~）内的 ``#`` 注释行**不**视为标题。
- frontmatter（开头 ``--- ... ---``）不属于任何 chunk——其字段由登记层
  （documents 行）承载，正文切块从 frontmatter 之后开始。
- kind=summary（seq=0）不在本模块：那是管线的职责（chunker 只产正文块）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Chunk",
    "ChunkerParams",
    "DEFAULT_PARAMS",
    "MARKDOWN_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "approx_token_count",
    "chunk_markdown",
    "chunk_plain",
    "split_frontmatter",
]


# ------------------------------------------------------------------ 参数

@dataclass(frozen=True)
class ChunkerParams:
    """切块参数（全部可配置；默认值为经验近似，**靠后网格实验调参**）。

    - ``target_size``         单 chunk 目标大小（近似 token）。md 小节超过该值时
                              再按段落/行二次切分。
    - ``overlap``             相邻窗口的重叠量（近似 token）；仅二次切分生效，
                              小节级切块不重叠（标题边界天然语义完整）。
    - ``max_chunks_per_file`` 单文件最大块数；溢出合并进最后一块（不丢内容）。
    - ``min_chunk_size``      二次切分产生的尾块小于该值时并入前一块（防碎片）。
    """

    target_size: int = 512
    overlap: int = 64
    max_chunks_per_file: int = 100
    min_chunk_size: int = 32

    def __post_init__(self) -> None:
        if self.target_size <= 0:
            raise ValueError(f"target_size 必须为正，收到 {self.target_size}")
        if not 0 <= self.overlap < self.target_size:
            raise ValueError(
                f"overlap 必须在 [0, target_size) 内，收到 overlap={self.overlap}, "
                f"target_size={self.target_size}"
            )
        if self.max_chunks_per_file < 1:
            raise ValueError(
                f"max_chunks_per_file 必须 >= 1，收到 {self.max_chunks_per_file}"
            )
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size 不能为负，收到 {self.min_chunk_size}")


DEFAULT_PARAMS = ChunkerParams()


@dataclass(frozen=True)
class Chunk:
    """一个正文块。

    ``seq`` 从 1 起（seq=0 保留给文件级 summary，由管线写入）；
    ``text`` 已并入标题路径头（若有）；``heading_path`` 为空元组表示无标题路径。
    """

    seq: int
    text: str
    heading_path: tuple[str, ...] = ()
    char_count: int = field(init=False)
    approx_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))
        object.__setattr__(self, "approx_tokens", approx_token_count(self.text))


#: v1 切块范围（映射入库规则 Q8）：md 按标题层级；txt 按段落聚合。
#: 其他扩展名只登记不切块（管线依据本表判断）。
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
TEXT_EXTENSIONS = MARKDOWN_EXTENSIONS | {".txt"}


# ------------------------------------------------------------ token 近似

_ASCII_WORD = re.compile(r"[A-Za-z0-9]+")


def approx_token_count(text: str) -> int:
    """近似 token 计数（确定性、无外部依赖）。

    规则：连续 ASCII 字母/数字串计 1（下划线视为分隔符、**不计** token——
    与 FTS 侧 span 预处理把 ``_`` 变为独立占位 token 后被丢弃的行为一致，
    ``connect_mcp`` ≈ 2 token）；CJK 字符（含日文假名、韩文谚文、扩展 A/
    兼容区）计 1；其余非空白字符（标点等）计 1；ASCII 空白不计。
    与 XLM-R（bge-m3 底座）量级近似：中文约 1 字 1 token，英文约 1 词 1-2 token。
    """
    if not text:
        return 0
    count = 0
    pos = 0
    for match in _ASCII_WORD.finditer(text):
        for ch in text[pos:match.start()]:
            if ch.isspace() or ch == "_":
                continue
            count += 1  # CJK 与其他非空白字符（标点等）逐字符计 1
        count += 1  # 整个 ASCII 词计 1
        pos = match.end()
    for ch in text[pos:]:
        if ch.isspace() or ch == "_":
            continue
        count += 1
    return count


# ---------------------------------------------------------- frontmatter

_FM_OPEN = re.compile(r"\A\uFEFF?[ \t]*---[ \t]*\n")
_FM_CLOSE = re.compile(r"^[ \t]*---[ \t]*$", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """剥离 md frontmatter，返回 ``(frontmatter | None, body)``。

    - 仅当文件以 ``---`` 行开头（允许 BOM 与行首空白）时视为有 frontmatter；
    - frontmatter 本体（含围栏行）作为第一个元素返回（原样、含结尾换行）；
    - 无 frontmatter 时返回 ``(None, 原文)``。
    """
    open_match = _FM_OPEN.match(text)
    if open_match is None:
        return None, text
    close_match = _FM_CLOSE.search(text, open_match.end())
    if close_match is None:  # 只有开栏没有闭栏：按无 frontmatter 处理
        return None, text
    end = close_match.end()
    if end < len(text) and text[end] == "\n":
        end += 1  # 闭栏行的行尾换行归 frontmatter（body 从下一行起）
    frontmatter = text[:end]
    body = text[end:]
    return frontmatter, body


# ------------------------------------------------------------ 窗口切分

def _split_oversize_line(text: str, target: int, overlap: int) -> list[str]:
    """对单条超长文本做字符级窗口切分（approx token 空间），带 overlap。"""
    if approx_token_count(text) <= target:
        return [text]
    windows: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        # 前向找窗口终点：累计 approx token 达 target
        tokens = 0
        end = start
        while end < n and tokens < target:
            tokens += approx_token_count(text[end])
            end += 1
        window = text[start:end]
        windows.append(window)
        if end >= n:
            break
        # 回退 overlap 个 token 作为下一窗口头部（保证窗口前进）
        back_tokens = 0
        back = end
        while back > start and back_tokens < overlap:
            back -= 1
            back_tokens += approx_token_count(text[back])
        start = back if back > start else end
    return [w for w in windows if w.strip()]


def _tail_chars(text: str, tokens: int) -> str:
    """取 ``text`` 尾部约 ``tokens`` 个 token 的字符（用于字符级 overlap）。"""
    if tokens <= 0 or not text:
        return ""
    collected = 0
    idx = len(text)
    while idx > 0 and collected < tokens:
        idx -= 1
        collected += approx_token_count(text[idx])
    return text[idx:]


def _overlap_tail(acc: list[str], overlap: int) -> list[str]:
    """从已聚合窗口的尾部取 overlap 重叠内容（带进下一窗口头部）。

    先按整行取；单行放不下（行 token > 剩余预算）时截取该行尾部的
    字符片段（字符级 overlap），保证 overlap > 0 时总有非空重叠。
    """
    if overlap <= 0 or not acc:
        return []
    tail: list[str] = []
    budget = overlap
    for prev in reversed(acc):
        t = approx_token_count(prev)
        if t == 0:  # 空行等零成本行：白送
            tail.insert(0, prev)
            continue
        if t <= budget:
            tail.insert(0, prev)
            budget -= t
            continue
        tail.insert(0, _tail_chars(prev, budget))
        return tail
    return tail


def _window_split(
    text: str, *, target: int, overlap: int, min_chunk: int
) -> list[str]:
    """把一段文本按行聚合为 ≤ target 的窗口（超长单行再字符级切分）。

    相邻窗口带 overlap 重叠（取上一窗口尾部的行/字符带进下一窗口头部）；
    尾窗口小于 ``min_chunk`` 时并入前一窗口（合并后允许超出 target，防碎片优先）。
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text] if text.strip() else []

    units: list[str] = []
    for line in lines:
        if approx_token_count(line) > target:
            units.extend(_split_oversize_line(line, target, overlap))
        else:
            units.append(line)

    windows: list[str] = []
    acc: list[str] = []
    acc_tokens = 0
    for unit in units:
        unit_tokens = approx_token_count(unit)
        if acc and acc_tokens + unit_tokens > target:
            windows.append("".join(acc))
            tail = _overlap_tail(acc, overlap)
            acc = [*tail, unit]
            acc_tokens = sum(approx_token_count(t) for t in tail) + unit_tokens
        else:
            acc.append(unit)
            acc_tokens += unit_tokens
    if acc:
        windows.append("".join(acc))

    # 尾碎片合并（先 pop 再拼回最后一个元素，避免下标随列表缩短而错位）
    if len(windows) > 1 and approx_token_count(windows[-1]) < min_chunk:
        last = windows.pop()
        windows[-1] += last
    return [w for w in windows if w.strip()]


# -------------------------------------------------------------- md 切块

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def _heading_sections(body: str) -> list[tuple[tuple[str, ...], str]]:
    """按标题行把正文切成 ``(标题路径, 小节文本)`` 列表（fence 感知）。

    - 标题路径 = 各级祖先标题文本（含当前标题），如 ``("一级", "二级")``；
      深度跳跃（如 h1 直接到 h3）时路径中缺失的祖先层级直接跳过；
    - 小节文本 = 从标题行到下一个**不浅于**当前深度的标题行之前；
    - 无任何标题时返回 ``[((), 全文)]``。
    """
    sections: list[tuple[tuple[str, ...], str]] = []
    parents: dict[int, str] = {}  # 标题深度 -> 标题文本（当前活跃祖先）
    current: list[str] = []
    current_path: tuple[str, ...] = ()
    in_fence: str | None = None

    def flush() -> None:
        text = "".join(current)
        if text.strip():
            sections.append((current_path, text))
        current.clear()

    for line in body.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_match is not None:
            marker = fence_match.group(1)
            if in_fence is None:
                in_fence = marker[0]
            elif marker[0] == in_fence:
                in_fence = None
            current.append(line)
            continue
        if in_fence is None:
            heading_match = _HEADING.match(line)
            if heading_match is not None:
                flush()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                parents[level] = title
                current_path = tuple(
                    parents[depth] for depth in sorted(parents) if depth < level
                ) + (title,)
                current.append(line)
                continue
        current.append(line)
    flush()
    if not sections:
        return [((), body)]
    return sections


def _merge_overflow(
    raw: list[tuple[tuple[str, ...], str]], max_chunks: int, joiner: str
) -> list[tuple[tuple[str, ...], str]]:
    """超过单文件最大块数时，把溢出块合并进最后一块（不丢内容）。

    保留前 ``max_chunks`` 块，第 ``max_chunks`` 块起全部并入第 ``max_chunks`` 块。
    """
    if len(raw) <= max_chunks:
        return raw
    head = raw[:max_chunks]
    overflow = joiner.join(t for _, t in raw[max_chunks:])
    last_path, last_text = head[-1]
    head[-1] = (last_path, last_text + joiner + overflow)
    return head


def chunk_markdown(
    text: str, params: ChunkerParams = DEFAULT_PARAMS
) -> list[Chunk]:
    """md 文本 → chunk 列表（seq 从 1 连续编号）。

    流程：剥 frontmatter → 按标题层级切小节 → 小节超过 target 再按段落/行
    二次切分（带 overlap）→ 标题路径并入每个 chunk 文本头 → 超过单文件
    最大块数时溢出合并进最后一块。
    """
    _, body = split_frontmatter(text)
    if not body.strip():
        return []

    raw_texts: list[tuple[tuple[str, ...], str]] = []
    for path, section in _heading_sections(body):
        header = (" / ".join(path) + "\n") if path else ""
        if approx_token_count(section) <= params.target_size:
            raw_texts.append((path, header + section))
        else:
            for window in _window_split(
                section,
                target=params.target_size,
                overlap=params.overlap,
                min_chunk=params.min_chunk_size,
            ):
                raw_texts.append((path, header + window))

    raw_texts = _merge_overflow(raw_texts, params.max_chunks_per_file, "\n\n")
    return [
        Chunk(seq=i + 1, text=chunk_text, heading_path=path)
        for i, (path, chunk_text) in enumerate(raw_texts)
    ]


# ----------------------------------------------------------- 非 md 切块

def chunk_plain(text: str, params: ChunkerParams = DEFAULT_PARAMS) -> list[Chunk]:
    """非 md 文本 → 段落聚合 chunk 列表（seq 从 1 连续编号，无标题路径）。

    段落 = 空行分隔的文本块（行级聚合，含行尾换行）；贪心聚合到 ≈ target；
    超长单行字符级再切分（带 overlap）；超出单文件最大块数时溢出合并进最后一块。
    """
    if not text.strip():
        return []
    windows = _window_split(
        text,
        target=params.target_size,
        overlap=params.overlap,
        min_chunk=params.min_chunk_size,
    )
    merged = _merge_overflow([((), w) for w in windows], params.max_chunks_per_file, "\n")
    return [Chunk(seq=i + 1, text=chunk_text) for i, (_, chunk_text) in enumerate(merged)]

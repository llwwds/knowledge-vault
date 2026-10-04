"""chunker 单元测试：全合成数据，重点覆盖标题路径 / fence / 溢出合并 / 确定性。"""

from __future__ import annotations

import pytest

from knowledge_vault.chunker import (
    MARKDOWN_EXTENSIONS,
    TEXT_EXTENSIONS,
    ChunkerParams,
    approx_token_count,
    chunk_markdown,
    chunk_plain,
    split_frontmatter,
)


# ------------------------------------------------------------- token 近似

def test_approx_token_count_cjk_is_one_per_char():
    assert approx_token_count("知识库") == 3
    assert approx_token_count("知识库。") == 4  # 中文句号逐字符计 1


def test_approx_token_count_ascii_words():
    assert approx_token_count("hello world") == 2
    # 下划线是分隔符不计 token：connect_mcp ≈ connect + mcp 两个词
    assert approx_token_count("connect_mcp v2") == 3
    assert approx_token_count("") == 0
    assert approx_token_count("   \n\t") == 0


def test_approx_token_count_mixed():
    # 1 个 ASCII 词（bge）+ 4 个 CJK 字（模型 评测）
    assert approx_token_count("bge 模型 评测") == 5


# ---------------------------------------------------------------- 参数校验

def test_params_defaults_documented_values():
    params = ChunkerParams()
    assert params.target_size == 512
    assert params.overlap == 64
    assert params.max_chunks_per_file == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_size": 0},
        {"target_size": -1},
        {"overlap": -1},
        {"overlap": 512},  # overlap >= target
        {"max_chunks_per_file": 0},
        {"min_chunk_size": -1},
    ],
)
def test_params_validation_rejects(kwargs):
    with pytest.raises(ValueError):
        ChunkerParams(**kwargs)


# ------------------------------------------------------------- frontmatter

def test_split_frontmatter_basic():
    text = "---\ntitle: 笔记\n---\n\n# 正文\n内容"
    fm, body = split_frontmatter(text)
    assert fm == "---\ntitle: 笔记\n---\n"
    assert body == "\n# 正文\n内容"


def test_split_frontmatter_absent():
    text = "# 不是 frontmatter\n--- 中间的分隔线"
    fm, body = split_frontmatter(text)
    assert fm is None
    assert body == text


def test_split_frontmatter_unclosed_treated_as_absent():
    text = "---\ntitle: 没有闭栏\n# 正文"
    fm, body = split_frontmatter(text)
    assert fm is None
    assert body == text


def test_split_frontmatter_bom_tolerated():
    text = "\ufeff---\ntags: [a]\n---\nbody"
    fm, body = split_frontmatter(text)
    assert fm is not None
    assert body == "body"


# ------------------------------------------------------------- md 切块

def test_chunk_markdown_heading_paths_and_seq():
    text = (
        "# 总览\n\n总览正文。\n\n"
        "## 检索\n\n检索正文 retrieval。\n\n"
        "### 实现\n\n实现正文。\n"
    )
    chunks = chunk_markdown(text)
    assert [c.seq for c in chunks] == [1, 2, 3]
    assert chunks[0].heading_path == ("总览",)
    assert chunks[1].heading_path == ("总览", "检索")
    assert chunks[2].heading_path == ("总览", "检索", "实现")
    # 标题路径并入文本头，且标题行保留
    assert chunks[1].text.startswith("总览 / 检索\n")
    assert "## 检索" in chunks[1].text
    assert "检索正文" in chunks[1].text


def test_chunk_markdown_depth_jump_skips_missing_ancestors():
    text = "# 一级\n\n\n### 三级跳\n\n内容。\n"
    chunks = chunk_markdown(text)
    assert chunks[0].heading_path == ("一级",)
    assert chunks[1].heading_path == ("一级", "三级跳")


def test_chunk_markdown_fence_protects_hash_lines():
    text = (
        "# 配置\n\n```python\n# 这不是标题注释\nx = 1\n```\n\n结尾。\n"
    )
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("配置",)
    assert "# 这不是标题注释" in chunks[0].text


def test_chunk_markdown_no_heading_falls_back_to_windows():
    text = "\n\n".join(f"普通段落 {i}。" for i in range(8))
    chunks = chunk_markdown(text, ChunkerParams(target_size=16, overlap=0))
    assert len(chunks) >= 2
    assert all(c.heading_path == () for c in chunks)


def test_chunk_markdown_frontmatter_excluded_from_chunks():
    text = "---\ntitle: T\n---\n\n# 正文\n\nfrontmatter 之外的内容 unique_token。\n"
    chunks = chunk_markdown(text)
    assert all("title: T" not in c.text for c in chunks)
    assert any("unique_token" in c.text for c in chunks)


def test_chunk_markdown_large_section_split_with_overlap():
    lines = [f"段落行 {i} 的内容。" for i in range(12)]
    text = "# 大节\n\n" + "\n".join(lines) + "\n"
    params = ChunkerParams(target_size=14, overlap=6, min_chunk_size=2)
    chunks = chunk_markdown(text, params)
    assert len(chunks) >= 3
    # 每个二次切分块都带标题路径头
    assert all(c.text.startswith("大节\n") for c in chunks)

    def body(chunk):
        return chunk.text[len("大节\n"):]

    def overlap_chars(prev_body: str, nxt_body: str) -> int:
        # nxt 的前缀同时是 prev 的后缀的最大长度（字符级 overlap 的不变量）
        for k in range(len(nxt_body), 0, -1):
            if prev_body.endswith(nxt_body[:k]):
                return k
        return 0

    for prev, nxt in zip(chunks, chunks[1:]):
        assert overlap_chars(body(prev), body(nxt)) >= 4


def test_chunk_markdown_max_chunks_merge_overflow():
    paragraphs = "\n\n".join(f"第 {i} 段内容 enough。" for i in range(20))
    text = "# 头\n\n" + paragraphs + "\n"
    params = ChunkerParams(target_size=10, overlap=0, max_chunks_per_file=3)
    chunks = chunk_markdown(text, params)
    assert len(chunks) == 3
    joined = "\n".join(c.text for c in chunks)
    for i in range(20):  # 溢出内容合并进最后一块，不丢内容
        assert f"第 {i} 段" in joined


def test_chunk_markdown_max_chunks_one():
    text = "# 唯一\n\n甲段落。\n\n乙段落。\n"
    chunks = chunk_markdown(
        text, ChunkerParams(max_chunks_per_file=1, target_size=4, overlap=1)
    )
    assert len(chunks) == 1
    assert "甲段落" in chunks[0].text and "乙段落" in chunks[0].text


def test_chunk_markdown_empty_and_whitespace():
    assert chunk_markdown("") == []
    assert chunk_markdown("---\ntitle: t\n---\n\n   \n") == []


def test_chunk_markdown_deterministic():
    text = "# A\n\n内容一。\n\n## B\n\n内容二。\n" * 3
    assert chunk_markdown(text) == chunk_markdown(text)


# --------------------------------------------------------- 非 md 文本切块

def test_chunk_plain_paragraph_aggregation():
    paragraphs = [f"段落 {i}：" + "细节内容。" * 3 for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_plain(text, ChunkerParams(target_size=30, overlap=0))
    assert len(chunks) >= 2
    assert [c.seq for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.heading_path == () for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for p in paragraphs:  # 段落聚合不丢段落
        assert p in joined


def test_chunk_plain_oversize_paragraph_hard_split():
    long_line = "超长单行" * 200  # ≫ target
    chunks = chunk_plain(long_line, ChunkerParams(target_size=50, overlap=10))
    assert len(chunks) >= 3
    assert all(c.approx_tokens <= 60 for c in chunks)  # 允许 overlap 导致的少量超出


def test_chunk_plain_tail_merge_min_chunk():
    # 尾块只有极少量内容时并入前一块
    text = "首段落内容较多，足够长一些。\n\n尾"
    chunks = chunk_plain(text, ChunkerParams(target_size=12, overlap=0, min_chunk_size=30))
    assert len(chunks) == 1


def test_chunk_plain_empty():
    assert chunk_plain("") == []
    assert chunk_plain("  \n  \n") == []


# ------------------------------------------------------------- 扩展名表

def test_extension_sets():
    assert MARKDOWN_EXTENSIONS == {".md", ".markdown"}
    assert TEXT_EXTENSIONS == MARKDOWN_EXTENSIONS | {".txt"}

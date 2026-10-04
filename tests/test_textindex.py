"""textindex：span 预处理 / userdict / FTS 同步 / 检索 / snippet。

红线：本文件只用合成文本，不读取 vault/快照/testdata 的任何真实内容。
"""

from __future__ import annotations

import sqlite3

import pytest

from knowledge_vault import (
    Tokenizer,
    add_chunk,
    add_chunks,
    add_summary_chunk,
    chunk_id_for,
    delete_chunks_for_file,
    make_span_preprocess,
    search,
    search_tokens,
    snippet,
)


# ---------------------------------------------------------------- 分词与镜像

class TestSpanPreprocess:
    """spike-5 结论的落地验证：`_`/`/` 形态不再被拆碎成孤立 token。"""

    @pytest.mark.parametrize(
        "text",
        ["connect_mcp", "mcp__server__tool", "a/b", "知识/图谱", "x_y/z_w"],
    )
    def test_no_isolated_underscore_or_slash_tokens(self, tokenizer, text):
        tokens = tokenizer.tokenize(text)
        assert tokens, "不应产出空 token 流"
        for tok in tokens:
            assert tok not in ("_", "/", "__", "＿＿")
            assert any(ch.isalnum() for ch in tok), tok

    def test_known_token_streams(self, tokenizer):
        assert tokenizer.tokenize("connect_mcp") == ["connect", "mcp"]
        assert tokenizer.tokenize("mcp__server__tool") == ["mcp", "server", "tool"]
        assert tokenizer.tokenize("a/b") == ["a", "b"]
        assert tokenizer.tokenize("知识/图谱") == ["知识", "图谱"]
        assert tokenizer.tokenize("x_y/z_w") == ["x", "y", "z", "w"]

    def test_index_text_space_joined(self, tokenizer):
        assert tokenizer.index_text("connect_mcp") == "connect mcp"
        assert tokenizer.index_text("mcp__server__tool") == "mcp server tool"

    def test_query_mirror(self, tokenizer):
        assert search_tokens(tokenizer, "connect_mcp") == '"connect" "mcp"'
        assert search_tokens(tokenizer, "mcp__server__tool") == '"mcp" "server" "tool"'
        assert search_tokens(tokenizer, "a/b") == '"a" "b"'

    def test_all_punctuation_query_yields_empty_match(self, tokenizer):
        assert tokenizer.tokenize("///") == []
        assert search_tokens(tokenizer, "///") == ""
        assert tokenizer.tokenize("") == []

    def test_placeholder_only_tokens_never_reach_index(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["a/b 与 connect_mcp 的说明"])
        raw = store.conn.execute("SELECT text FROM chunks").fetchone()["text"]
        assert raw == "a/b 与 connect_mcp 的说明"     # chunks 存原文，占位只影响索引
        # FTS 侧不含孤立 `_`/`/`（unicode61 中它们本来就是 separator，这里验证索引存在）
        assert store.search("a/b")
        assert store.search("connect_mcp")


class TestTokenizerConfig:
    def test_userdict_terms_stay_whole(self, tokenizer):
        # 包内打包词典（experiments/userdict_v0.txt 的拷贝）
        assert tokenizer.tokenize("bge-m3") == ["bge-m3"]
        assert tokenizer.tokenize("knowledge-vault") == ["knowledge-vault"]
        assert tokenizer.tokenize("AGENTS.md") == ["AGENTS.md"]
        assert tokenizer.tokenize("Few-Shot") == ["Few-Shot"]
        assert tokenizer.tokenize("Node.js") == ["Node.js"]
        assert search_tokens(tokenizer, "bge-m3") == '"bge-m3"'

    def test_synthetic_userdict_merges_word(self, tmp_path):
        dict_file = tmp_path / "synthetic-dict.txt"
        dict_file.write_text("starfish-project 100000 n\n", encoding="utf-8")
        tok = Tokenizer(userdict_path=dict_file)
        assert tok.tokenize("starfish-project") == ["starfish-project"]

    def test_empty_span_map_identity_preprocess(self):
        tok = Tokenizer(preprocess=make_span_preprocess({}))
        # 占位替换关闭后，丢弃规则仍然兜底：`/` 不会成为孤立 token
        assert tok.tokenize("a/b") == ["a", "b"]

    def test_custom_preprocess_honored(self):
        tok = Tokenizer(preprocess=lambda s: s.replace("自定义词", "苹果"))
        assert tok.tokenize("自定义词测试") == ["苹果", "测试"]

    def test_make_span_preprocess_custom_mapping(self):
        pre = make_span_preprocess({"|": ";"})
        assert pre("a|b") == "a;b"
        assert pre("a_b") == "a_b"      # 未映射的字符保持原样

    def test_english_and_cjk_mixed(self, tokenizer):
        tokens = tokenizer.tokenize("用 zvec 存知识")
        assert "zvec" in tokens
        assert all(any(ch.isalnum() for ch in t) for t in tokens)


# ---------------------------------------------------------------- chunks 写入

class TestChunkWriting:
    def test_chunk_id_format_and_seq(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        ids = store.add_chunks(doc.file_id, ["第一段", "第二段", "第三段"])
        assert ids == [
            f"f{doc.file_id}-c1",
            f"f{doc.file_id}-c2",
            f"f{doc.file_id}-c3",
        ]
        assert chunk_id_for(doc.file_id, 1) == ids[0]

    def test_summary_chunk_convention(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        sid = store.add_summary_chunk(doc.file_id, "全文摘要")
        assert sid == f"f{doc.file_id}-c0"
        row = store.conn.execute(
            "SELECT kind, chunk_seq, char_count FROM chunks WHERE chunk_id = ?", (sid,)
        ).fetchone()
        assert row["kind"] == "summary"
        assert row["chunk_seq"] == 0
        assert row["char_count"] == len("全文摘要")

    def test_add_chunk_auto_seq_continues(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        store.add_chunks(doc.file_id, ["a", "b"])
        assert store.add_chunk(doc.file_id, "c") == f"f{doc.file_id}-c3"

    def test_add_chunks_start_seq_override(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        ids = store.add_chunks(doc.file_id, ["a", "b"], start_seq=5)
        assert ids == [f"f{doc.file_id}-c5", f"f{doc.file_id}-c6"]

    def test_char_count_matches_raw_text(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        store.add_chunks(doc.file_id, ["知识图谱"])
        row = store.conn.execute("SELECT char_count, text FROM chunks").fetchone()
        assert row["char_count"] == len(row["text"]) == 4

    def test_empty_text_chunk_allowed(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        cid = store.add_chunk(doc.file_id, "")
        assert cid == f"f{doc.file_id}-c1"
        assert store.search("任何词") == []

    def test_invalid_kind_rejected(self, store, make_file):
        doc = store.register_file(make_file("doc.md", "x"))
        with pytest.raises(ValueError):
            store.add_chunk(doc.file_id, "x", kind="para")

    def test_fk_enforced_on_unknown_file(self, conn, tokenizer):
        with pytest.raises(sqlite3.IntegrityError):
            add_chunk(conn, tokenizer, 424242, "孤儿块")


class TestFtsSync:
    def test_delete_chunks_for_file_cleans_fts(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["第一版内容向量", "第二版内容"])
        assert len(store.search("向量")) == 1

        assert store.delete_chunks_for_file(doc.file_id) == 2
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert store.search("向量") == []
        assert store.search("内容") == []

    def test_rechunk_after_delete(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["旧文本甲", "旧文本乙"])
        store.delete_chunks_for_file(doc.file_id)
        ids = store.add_chunks(doc.file_id, ["新文本向量库"])
        assert ids == [f"f{doc.file_id}-c1"]        # seq 重新从 1 起
        hits = store.search("向量")
        assert [h.chunk_id for h in hits] == ids
        assert all("旧文本" not in h.text for h in hits)

    def test_delete_file_without_chunks_is_noop(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        assert store.delete_chunks_for_file(doc.file_id) == 0

    def test_only_target_file_chunks_removed(self, store, make_file):
        d1 = store.register_file(make_file("m1.md", "x"))
        d2 = store.register_file(make_file("m2.md", "x"))
        store.add_chunks(d1.file_id, ["甲文件的向量"])
        store.add_chunks(d2.file_id, ["乙文件的向量"])
        store.delete_chunks_for_file(d1.file_id)
        remaining = store.search("向量")
        assert [h.file_id for h in remaining] == [d2.file_id]


# -------------------------------------------------------------------- 查询侧

class TestSearch:
    def test_hit_shape(self, store, make_file):
        doc = store.register_file(make_file("tech.md", "x"))
        store.add_chunks(doc.file_id, ["我们用 zvec 做向量检索", "FTS5 负责关键词检索"])

        hits = store.search("向量")
        assert len(hits) == 1
        hit = hits[0]
        assert hit.file_id == doc.file_id
        assert hit.chunk_id == f"f{doc.file_id}-c1"
        assert hit.chunk_seq == 1
        assert hit.kind == "chunk"
        assert hit.text == "我们用 zvec 做向量检索"        # 原文返回
        assert hit.score < 0                              # bm25 越小越相关
        assert hit.snippet                                # 片段非空（CJK 不保证高亮，见 snippet 测试）

    def test_userdict_term_retrieval(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["用 bge-m3 做嵌入，用 knowledge-vault 登记"])
        hits = store.search("bge-m3")
        assert [h.file_id for h in hits] == [doc.file_id]
        hits2 = store.search("knowledge-vault")
        assert [h.file_id for h in hits2] == [doc.file_id]

    def test_span_terms_end_to_end(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, [
            "connect_mcp 提供连接层",
            "工具命名形如 mcp__server__tool",
            "路径写法 a/b 与 c/d",
        ])
        assert {h.chunk_id for h in store.search("connect_mcp")} == {f"f{doc.file_id}-c1"}
        assert {h.chunk_id for h in store.search("mcp__server__tool")} == {f"f{doc.file_id}-c2"}
        assert {h.chunk_id for h in store.search("a/b")} == {f"f{doc.file_id}-c3"}

    def test_chinese_retrieval(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["知识图谱是知识库的核心表示形式"])
        assert store.search("知识图谱")
        assert store.search("图谱")
        assert store.search("核心表示")  # 多词 AND

    def test_no_hit_for_absent_term(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["完全无关的文本"])
        assert store.search("量子纠缠态") == []

    def test_filters_and_limit(self, store, make_file):
        d1 = store.register_file(make_file("m1.md", "x"))
        d2 = store.register_file(make_file("m2.md", "x"))
        store.add_chunks(d1.file_id, ["向量数据库笔记一", "向量数据库笔记二"])
        store.add_summary_chunk(d1.file_id, "摘要：向量主题")
        store.add_chunks(d2.file_id, ["别人的向量笔记"])

        assert len(store.search("向量")) == 4
        assert [h.chunk_id for h in store.search("向量", kind="summary")] == [
            f"f{d1.file_id}-c0"
        ]
        assert {h.file_id for h in store.search("向量", file_id=d2.file_id)} == {
            d2.file_id
        }
        assert len(store.search("向量", limit=2)) == 2

    def test_scores_ascending(self, store, make_file):
        d1 = store.register_file(make_file("m1.md", "x"))
        d2 = store.register_file(make_file("m2.md", "x"))
        store.add_chunks(d1.file_id, ["向量检索专题"])
        store.add_chunks(d2.file_id, ["无关文本"])
        scores = [h.score for h in store.search("向量 检索")]
        assert scores == sorted(scores)


class TestSnippet:
    def test_ascii_terms_marked_on_raw_text(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["connect_mcp 提供连接层"])
        snips = store.snippet("connect_mcp")
        assert len(snips) == 1
        assert "[connect]_[mcp]" in snips[0]

    def test_custom_marks(self, store, make_file):
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["plain text engine"])
        snips = store.snippet(
            "text", hl_open="<em>", hl_close="</em>", ellipsis="...", tokens=8
        )
        assert snips == ["plain <em>text</em> engine"]

    def test_cjk_returns_usable_fragment(self, store, make_file):
        # 已知限制：external content 下原文按 unicode61 再切分，中文词的高亮标记
        # 可能缺失（索引是 jieba 预分词的），但片段本身可用。
        doc = store.register_file(make_file("m.md", "x"))
        store.add_chunks(doc.file_id, ["我们用 zvec 做向量检索"])
        snips = store.snippet("向量")
        assert len(snips) == 1
        assert "向量" in snips[0]

    def test_snippet_file_filter_and_empty_query(self, store, make_file):
        d1 = store.register_file(make_file("m1.md", "x"))
        d2 = store.register_file(make_file("m2.md", "x"))
        store.add_chunks(d1.file_id, ["alpha beta"])
        store.add_chunks(d2.file_id, ["alpha gamma"])
        hits = store.snippet("alpha", file_id=d2.file_id)
        assert len(hits) == 1 and "gamma" in hits[0]
        assert store.snippet("") == []

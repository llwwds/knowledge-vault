"""CLI 冒烟测试：kv-search / kv-stats / kv-optimize（注入替身，不起服务）。

kv-serve 为阻塞式，其服务器行为已由 test_api.py 以 make_server 覆盖；
这里只验证参数解析与入口分发。全程合成数据，不读真实环境。
"""

from __future__ import annotations

import json

import pytest

from knowledge_vault.cli import build_parser, main

from _phase3_helpers import FakeRerankerByLength, build_kb


@pytest.fixture()
def kb(store, make_file) -> dict:
    return build_kb(store, make_file)


def _parse_stdout_jsonl(capsys) -> list[dict]:
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


class TestKvSearch:
    def test_search_jsonl_output(self, store, kb, capsys):
        code = main(["kv-search", "训练数据", "--no-rerank", "--no-graph"], store=store)
        assert code == 0
        records = _parse_stdout_jsonl(capsys)
        assert records, "应至少输出一行 JSON"
        assert all("chunk_id" in rec and "score" in rec for rec in records)
        assert all(rec["file_id"] != kb["d"] for rec in records)  # 软删除不出

    def test_search_meta_on_stderr(self, store, kb, capsys):
        code = main(["kv-search", "训练数据", "--no-rerank", "--no-graph"], store=store)
        assert code == 0
        err = capsys.readouterr().err
        meta_line = [ln for ln in err.splitlines() if ln.strip()][-1]
        meta = json.loads(meta_line)["meta"]
        assert meta["fts_hits"] == 1
        assert meta["reranked"] is False

    def test_search_with_injected_reranker(self, store, kb, capsys):
        code = main(
            ["kv-search", "机器学习", "--no-graph"],
            store=store,
            reranker=FakeRerankerByLength(),
        )
        assert code == 0
        err = capsys.readouterr().err
        meta = json.loads(err.splitlines()[-1])["meta"]
        assert meta["reranked"] is True

    def test_search_filters(self, store, kb, capsys):
        code = main(
            ["kv-search", "机器学习", "--no-rerank", "--no-graph",
             "--status", "now", "--tag", "ml", "--top-j", "1"],
            store=store,
        )
        assert code == 0
        records = _parse_stdout_jsonl(capsys)
        assert len(records) == 1
        assert records[0]["file_id"] == kb["a"]

    def test_search_no_injected_store_opens_own(self, config, monkeypatch, capsys):
        # 不注入 store：main 内部自建 Store（init=True）并自行关闭
        monkeypatch.setattr(
            "knowledge_vault.textindex.Tokenizer.load_userdict", staticmethod(lambda p: None)
        )
        code = main(["kv-search", "任意", "--no-rerank", "--no-graph"], config=config)
        assert code == 0
        assert _parse_stdout_jsonl(capsys) == []

    def test_search_reranker_defaults_to_flag_impl_lazy(self, store, kb, monkeypatch):
        # 不注入 reranker 且未加 --no-rerank：默认 FlagRerankerImpl（惰性，不在此加载）
        import knowledge_vault.cli as cli

        created = []
        orig = cli.FlagRerankerImpl

        class Spy(orig):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                created.append(self)

        monkeypatch.setattr(cli, "FlagRerankerImpl", Spy)
        code = main(
            ["kv-search", "训练数据", "--no-graph", "--no-rerank"], store=store
        )
        assert code == 0
        assert created == []  # --no-rerank 不构造真实 reranker


class TestKvStatsAndOptimize:
    def test_stats_json(self, store, kb, config, capsys):
        code = main(["kv-stats"], config=config)
        assert code == 0
        stats = json.loads(capsys.readouterr().out)
        assert stats["documents"]["total"] == 5
        assert stats["documents"]["active"] == 4  # d 已软删除
        assert stats["documents"]["deleted"] == 1
        assert stats["chunks"]["total"] == 7
        assert stats["chunks"]["by_kind"] == {"chunk": 6, "summary": 1}
        assert stats["edges"] == 3
        assert stats["version"]

    def test_optimize_json(self, store, kb, config, capsys):
        code = main(["kv-optimize"], config=config)
        assert code == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["fts_optimized"] is True
        assert summary["vector"] is None  # tmp 状态目录无向量库


class TestParserAndErrors:
    def test_parser_accepts_all_subcommands(self):
        parser = build_parser()
        args = parser.parse_args(["kv-search", "q", "--no-rerank"])
        assert args.command == "kv-search" and args.no_rerank
        args = parser.parse_args(["kv-serve", "--port", "1234"])
        assert args.command == "kv-serve" and args.port == 1234
        assert parser.parse_args(["kv-optimize"]).command == "kv-optimize"
        assert parser.parse_args(["kv-stats"]).command == "kv-stats"

    def test_missing_command_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_bad_status_choice_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["kv-search", "q", "--status", "archived"])
        assert exc.value.code == 2

    def test_bad_top_j_exits(self, store):
        with pytest.raises(SystemExit):
            main(["kv-search", "q", "--top-j", "abc"], store=store)

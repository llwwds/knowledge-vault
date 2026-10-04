"""kv 命令行壳（argparse，stdlib only）。

子命令：

- ``kv-search "查询" [--status {now,library}] [--tag TAG ...] [--top-j N]
  [--no-rerank] [--no-graph]``：召回并按行输出 JSONL（每条结果一行 JSON）；
  rerank 默认开启（真实 FlagRerankerImpl，惰性加载），``--no-rerank`` 关闭。
- ``kv-serve [--host H] [--port P]``：阻塞式 HTTP API（见 api.py）。
- ``kv-optimize``：触发 FTS/SQLite/zvec 索引优化，输出 JSON 摘要。
- ``kv-stats``：输出只读运行统计 JSON。

写入不经 CLI/API——写入只走摄入管线；本壳只做读 + optimize 触发。

入口约定：``main(argv) -> int``（进程退出码）。console_scripts 入口
``kv = knowledge_vault.cli:main`` 已在 pyproject 注册。测试可用 ``config`` /
``store`` / ``reranker`` / ``embedder`` / ``vector_path`` 注入替身，不读真实环境。

向量路接线：未注入替身时自动发现——``state/zvec_chunks`` 存在则打开
collection 并配 ``BGEM3Embedder``（惰性加载）；不存在则降级全文+图两路。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .api import (
    DEFAULT_API_PORT,
    collect_stats,
    open_vector_path,
    run_optimize,
    serve,
)
from .config import VaultConfig, load_config
from .embedder import BGEM3Embedder
from .protocols import Embedder, Reranker
from .registry import STATUSES
from .rerank import FlagRerankerImpl
from .retrieve import VectorPath, retrieve
from .store import Store

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv",
        description="knowledge-vault 命令行壳：检索 / 服务 / 优化 / 统计（只读 + optimize）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("kv-search", help="混合召回：全文 + 图扩展（+rerank）")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument(
        "--status", choices=STATUSES, default=None, help="按登记状态过滤（缺省不过滤）"
    )
    p_search.add_argument(
        "--tag",
        action="append",
        default=None,
        dest="tags",
        metavar="TAG",
        help="context_tag 过滤，可重复；与文档标签数组交集非空即命中",
    )
    p_search.add_argument("--top-j", type=int, default=10, dest="top_j", help="返回条数上限")
    p_search.add_argument(
        "--no-rerank", action="store_true", dest="no_rerank", help="跳过 rerank 阶段"
    )
    p_search.add_argument(
        "--no-graph", action="store_true", dest="no_graph", help="跳过图扩展阶段"
    )

    p_serve = sub.add_parser("kv-serve", help="启动 HTTP API 服务（阻塞）")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=None, help=f"默认取 KV_API_PORT 或 {DEFAULT_API_PORT}")

    sub.add_parser("kv-optimize", help="触发 FTS/SQLite/zvec 索引优化")
    sub.add_parser("kv-stats", help="输出运行统计 JSON")
    return parser


def _build_components(
    *,
    config: VaultConfig | None = None,
    reranker: Reranker | None,
    embedder: Embedder | None,
    vector_path: VectorPath | None,
    no_rerank: bool,
):
    """组件装配点：测试注入替身；生产默认 FlagRerankerImpl（惰性加载）。

    向量路自动发现：未注入 ``vector_path`` 且 ``state/zvec_chunks`` 存在时
    打开 collection 并配 :class:`~knowledge_vault.embedder.BGEM3Embedder`
    （均惰性，查询时才加载模型）。返回 ``(reranker, embedder, vector_path,
    closables)``——``closables`` 为本调用打开的 collection 句柄，调用方负责
    关闭。
    """
    closables = []
    if vector_path is None and config is not None:
        auto_path, coll = open_vector_path(config)
        if auto_path is not None:
            vector_path = auto_path
            closables.append(coll)
            if embedder is None:
                embedder = BGEM3Embedder()
    reranker_final = None if no_rerank else (reranker if reranker is not None else FlagRerankerImpl())
    return reranker_final, embedder, vector_path, closables


def _cmd_search(args: argparse.Namespace, *, config: VaultConfig, injected: dict) -> int:
    own_store = injected.get("store") is None
    store = injected.get("store") or Store(config=config)
    closables: list = []
    try:
        reranker, embedder, vector_path, closables = _build_components(
            config=config,
            reranker=injected.get("reranker"),
            embedder=injected.get("embedder"),
            vector_path=injected.get("vector_path"),
            no_rerank=args.no_rerank,
        )
        output = retrieve(
            store,
            args.query,
            embedder=embedder,
            vector_path=vector_path,
            reranker=reranker,
            status=args.status,
            context_tag=args.tags,
            top_j=args.top_j,
            graph=not args.no_graph,
        )
    finally:
        if own_store:
            store.close()
        for coll in closables:
            coll.close()
    for item in output.items:
        print(json.dumps(item.as_dict(), ensure_ascii=False))
    print(
        json.dumps({"meta": output.meta}, ensure_ascii=False),
        file=sys.stderr,
    )
    return 0


def _cmd_serve(args: argparse.Namespace, *, config: VaultConfig, injected: dict) -> int:
    serve(
        config,
        host=args.host,
        port=args.port,
        reranker=injected.get("reranker"),
    )
    return 0


def _cmd_optimize(*, config: VaultConfig) -> int:
    print(json.dumps(run_optimize(config), ensure_ascii=False, indent=2))
    return 0


def _cmd_stats(*, config: VaultConfig) -> int:
    print(json.dumps(collect_stats(config), ensure_ascii=False, indent=2))
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    config: VaultConfig | None = None,
    store: Store | None = None,
    reranker: Reranker | None = None,
    embedder: Embedder | None = None,
    vector_path: VectorPath | None = None,
) -> int:
    """CLI 入口（console_scripts 形态签名：无参调用读环境，返回退出码）。"""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = config or load_config()
    injected = {"store": store, "reranker": reranker, "embedder": embedder, "vector_path": vector_path}
    try:
        if args.command == "kv-search":
            return _cmd_search(args, config=cfg, injected=injected)
        if args.command == "kv-serve":
            return _cmd_serve(args, config=cfg, injected=injected)
        if args.command == "kv-optimize":
            return _cmd_optimize(config=cfg)
        if args.command == "kv-stats":
            return _cmd_stats(config=cfg)
    except BrokenPipeError:  # 管道下游提前关闭（kv-search | head 等）
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":  # python -m knowledge_vault.cli ...
    sys.exit(main())

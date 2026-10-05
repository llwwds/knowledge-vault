#!/usr/bin/env python3
"""knowledge-vault agent 检索入口（低阻力）。

设计目标：agent 一条命令拿到可用的检索结果，零前置配置——
服务离线时自动拉起部署实例的 kv-serve，HTTP 不可用时回退直连 CLI。

用法：
    kv.py search "查询词" [--top-j N] [--rerank] [--tag TAG]... [--status now|library] [--json]
    kv.py stats [--json]
    kv.py file <file_id> [--json]
    kv.py serve-status | serve-start

输出：默认人类可读紧凑格式（score | sources | file_id | title | text 摘要），
--json 输出完整 JSON/JSONL（agent 二次加工用）。

环境变量：
    KV_CLI          kv CLI 全路径（默认 ~/llwwds_application/knowledge-vault/venv-mac/bin/kv）
    KV_API_PORT     HTTP 端口（默认 8770）
仅依赖 Python 标准库。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CLI = os.path.expanduser(
    "~/llwwds_application/knowledge-vault/venv-mac/bin/kv"
)
CLI = os.environ.get("KV_CLI", DEFAULT_CLI)
PORT = int(os.environ.get("KV_API_PORT", "8770"))
BASE = f"http://127.0.0.1:{PORT}"


def _http(method: str, path: str, body: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def health(timeout: float = 0.6) -> bool:
    try:
        _http("GET", "/health", timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 - 探测失败一律视为离线
        return False


def serve_start() -> bool:
    """后台拉起 kv-serve（部署实例），最多等 6s。"""
    if health():
        return True
    subprocess.Popen(
        [CLI, "kv-serve", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(12):
        time.sleep(0.5)
        if health():
            return True
    return False


def _print_items(items: list[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return
    for it in items:
        text = " ".join(str(it.get("text", "")).split())
        print(f"{it.get('score', 0):.4f} [{','.join(it.get('sources', []))}] "
              f"#{it.get('file_id')} {it.get('title', '')}")
        print(f"    {text[:160]}{'…' if len(text) > 160 else ''}")


def cmd_search(args) -> int:
    body = {
        "query": args.query,
        "top_j": args.top_j,
        "no_rerank": not args.rerank,
    }
    filters: dict = {}
    if args.status:
        filters["status"] = args.status
    if args.tag:
        filters["tags"] = args.tag
    if filters:
        body["filters"] = filters
    out = _http("POST", "/search", body, timeout=60.0 if args.rerank else 20.0)
    _print_items(out.get("items", []), args.json)
    meta = out.get("meta", {})
    if not args.json:
        print(f"--- {meta}", file=sys.stderr)
    return 0


def cmd_stats(args) -> int:
    out = _http("GET", "/stats")
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        d = out.get("documents", {})
        print(f"documents: {d.get('total')} (active {d.get('active')}, "
              f"deleted {d.get('deleted')}) | chunks: {out.get('chunks', {}).get('total')} "
              f"| edges: {out.get('edges')} | v{out.get('version')}")
    return 0


def cmd_file(args) -> int:
    out = _http("GET", f"/files/{args.file_id}")
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for k in ("file_id", "file_path", "title", "status", "context_tag",
                  "created_at", "mtime", "deleted_at"):
            print(f"{k}: {out.get(k)}")
    return 0


def cmd_serve_status(_args) -> int:
    print("online" if health() else "offline")
    return 0


def cmd_serve_start(_args) -> int:
    print("online" if serve_start() else "failed")
    return 0 if health() else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="knowledge-vault agent 检索入口")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="混合召回检索（默认快速模式）")
    p.add_argument("query")
    p.add_argument("--top-j", type=int, default=8, dest="top_j")
    p.add_argument("--rerank", action="store_true", help="开启精排（慢 ~4s/64对）")
    p.add_argument("--tag", action="append", default=None, dest="tag",
                   help="context_tag 过滤，可重复")
    p.add_argument("--status", choices=["now", "library"], default=None)
    p.add_argument("--json", action="store_true", help="输出完整 JSON")

    p = sub.add_parser("stats", help="库概况")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("file", help="按 file_id 查登记详情")
    p.add_argument("file_id", type=int)
    p.add_argument("--json", action="store_true")

    sub.add_parser("serve-status", help="HTTP 服务在线状态")
    sub.add_parser("serve-start", help="后台拉起 HTTP 服务（离线时自动执行）")

    args = ap.parse_args()

    # 低阻力核心：search/stats/file 前自动确保服务在线；拉起失败回退直连 CLI
    if args.command in ("search", "stats", "file") and not health():
        if not serve_start():
            if args.command == "search":
                cmd = [CLI, "kv-search", args.query, "--top-j", str(args.top_j),
                       "--no-rerank"]
                if args.tag:
                    for t in args.tag:
                        cmd += ["--tag", t]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                sys.stdout.write(proc.stdout)
                return proc.returncode
            print("服务离线且拉起失败", file=sys.stderr)
            return 1

    return {
        "search": cmd_search, "stats": cmd_stats, "file": cmd_file,
        "serve-status": cmd_serve_status, "serve-start": cmd_serve_start,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

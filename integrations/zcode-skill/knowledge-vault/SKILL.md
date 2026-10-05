---
name: knowledge-vault
description: Search the user's personal knowledge base (knowledge-vault: ~50k registered files — notes, project docs, work records, credentials, university files) with hybrid recall (FTS + vector + graph, optional rerank). Use when the user asks to find or look up their own notes, documents, past work, decisions, or personal knowledge; also use to ground answers in the user's own material before responding. Read-only plus index-optimize; writes happen only via the user's Obsidian editing.
---

# knowledge-vault — personal knowledge base search

Low-friction retrieval over the user's personal knowledge base. One command, structured results, no setup: the helper auto-starts the HTTP service if it is offline, and falls back to the CLI if that fails.

## Quick start

```bash
python "$SKILL_DIR/scripts/kv.py" search "风控 agent 工作任务"
```

- Default mode is fast (no rerank, sub-second). Add `--rerank` for best quality (~4s per 64 candidates) when the query is important.
- Results print as `score [sources] #file_id title` + a text snippet — the snippet is the actual indexed chunk content, usable directly as context.
- Add `--json` for full structured output (per-chunk `file_id`, `chunk_id`, `score`, `sources`, `file_path`, `title`, `text`).

## Typical agent workflow

1. `search "keywords"` — grab top hits; use `text` snippets as grounding context.
2. Need the file's metadata (path, dates, status)? `file <file_id>`.
3. Too many/too few hits? Narrow with `--tag "02 项目"` / `--status library`, widen with `--top-j 20`.
4. `stats` — library overview (document/chunk counts, version) when unsure what's indexed.

## Boundaries

- **Read-only**: search / stats / file are read endpoints; the only write is `POST /optimize` (index maintenance). Never write into the vault or the source area — content changes happen only through the user's Obsidian editing, then ingestion.
- Real vault content is sensitive: treat snippets as user-private data; do not send them to third-party services.
- Sensitive folders (credentials, 04 信息) may be excluded from vector/FTS indexes by configuration — absence from results is not proof of absence from the vault.

## Service management

- `kv.py serve-status` → `online` / `offline`.
- `kv.py serve-start` → background-start the HTTP service (port 8770, override with `KV_API_PORT`).
- All subcommands auto-start the service when needed; direct CLI fallback works even without it.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `KV_CLI` | `~/llwwds_application/knowledge-vault/venv-mac/bin/kv` | CLI fallback path |
| `KV_API_PORT` | `8770` | HTTP port |

Detailed Chinese usage notes: [references/usage.md](references/usage.md).

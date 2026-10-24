#!/usr/bin/env python3
"""Resume legacy catalog reconciliation into local SQLite/FTS without vectors.

The source vault and legacy manifest are read-only mounts. Aliases are created
only after the catalog relative path is present in the source manifest and the
current file SHA-256 exactly matches the historical full SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from knowledge_vault.chunker import MARKDOWN_EXTENSIONS, TEXT_EXTENSIONS
from knowledge_vault.config import load_config
from knowledge_vault.pipeline import IngestPipeline, NoVectorStore, PipelineState
from knowledge_vault.registry import Document, get_file
from knowledge_vault.store import Store

MANIFEST_ROOT = Path(os.environ.get("KV_LEGACY_MANIFEST_DIR", "/legacy-manifest"))
VAULT_ROOT = Path(os.environ.get("KV_VAULT_ROOT", "/obsidian-vault")).resolve()
STATE_DIR = Path(os.environ.get("KV_STATE_DIR", "/app_runtime/state"))
CATALOG_FILE = MANIFEST_ROOT / "01 结构地图与认领" / "document_catalog.jsonl"
SOURCE_MANIFEST_FILE = MANIFEST_ROOT / "00 契约与基线" / "source_manifest.jsonl"
SNAPSHOT_ID = "2026-09-06-legacy-document-catalog"
NAMESPACE = "knowledge-vault-legacy-catalog-v1"
REPORT_PATH = STATE_DIR / "legacy-restore-report.json"
BATCH_SIZE = int(os.environ.get("KV_LEGACY_BATCH_SIZE", "25"))


def _safe_relpath(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        return None
    return rel.as_posix()


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path.name}:{line_number}") from exc


def _full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _init_audit(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS legacy_catalog_audit (
            namespace TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            legacy_file_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            expected_sha256 TEXT,
            observed_sha256 TEXT,
            status TEXT NOT NULL,
            detail TEXT,
            audited_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(namespace, snapshot_id, legacy_file_id)
        );
        CREATE TABLE IF NOT EXISTS legacy_id_alias (
            namespace TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            legacy_file_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            full_sha256 TEXT NOT NULL,
            new_file_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(namespace, snapshot_id, legacy_file_id),
            UNIQUE(namespace, snapshot_id, new_file_id)
        );
        CREATE TABLE IF NOT EXISTS legacy_catalog_source_metadata (
            namespace TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            legacy_file_id TEXT NOT NULL,
            source_metadata_json TEXT NOT NULL,
            audited_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(namespace, snapshot_id, legacy_file_id)
        );
        """
    )
    conn.commit()


def _upsert_audit(conn, legacy_id, rel, expected, observed, status, detail=""):
    conn.execute(
        """INSERT INTO legacy_catalog_audit
           (namespace,snapshot_id,legacy_file_id,relative_path,expected_sha256,observed_sha256,status,detail,audited_at)
           VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
           ON CONFLICT(namespace,snapshot_id,legacy_file_id) DO UPDATE SET
             relative_path=excluded.relative_path, expected_sha256=excluded.expected_sha256,
             observed_sha256=excluded.observed_sha256, status=excluded.status,
             detail=excluded.detail, audited_at=CURRENT_TIMESTAMP""",
        (NAMESPACE, SNAPSHOT_ID, str(legacy_id), rel, expected, observed, status, detail),
    )


def _create_alias(conn, legacy_id, rel, full_sha, new_file_id):
    # Append-only: never rebind a legacy identifier to another integer ID.
    conn.execute(
        """INSERT OR IGNORE INTO legacy_id_alias
           (namespace,snapshot_id,legacy_file_id,relative_path,full_sha256,new_file_id)
           VALUES(?,?,?,?,?,?)""",
        (NAMESPACE, SNAPSHOT_ID, str(legacy_id), rel, full_sha, new_file_id),
    )
    row = conn.execute(
        "SELECT relative_path,full_sha256,new_file_id FROM legacy_id_alias WHERE namespace=? AND snapshot_id=? AND legacy_file_id=?",
        (NAMESPACE, SNAPSHOT_ID, str(legacy_id)),
    ).fetchone()
    if tuple(row) != (rel, full_sha, new_file_id):
        raise RuntimeError(f"legacy alias conflict for id {legacy_id}")


def _store_source_metadata(conn: sqlite3.Connection, catalog: dict) -> None:
    """Keep the historical classification in audit metadata, never business tags."""
    keys = (
        "rel_path", "source_sha256", "title", "status", "created_at",
        "sensitivity_class", "redaction_required", "write_mode",
    )
    metadata = {key: catalog.get(key) for key in keys}
    conn.execute(
        """INSERT INTO legacy_catalog_source_metadata
           (namespace,snapshot_id,legacy_file_id,source_metadata_json,audited_at)
           VALUES(?,?,?,?,CURRENT_TIMESTAMP)
           ON CONFLICT(namespace,snapshot_id,legacy_file_id) DO UPDATE SET
             source_metadata_json=excluded.source_metadata_json,
             audited_at=CURRENT_TIMESTAMP""",
        (NAMESPACE, SNAPSHOT_ID, str(catalog.get("file_id") or "unknown"),
         json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def _get_active_by_path(store: Store, path: str) -> Document | None:
    row = store.conn.execute(
        "SELECT file_id FROM documents WHERE file_path=? AND deleted_at IS NULL", (path,)
    ).fetchone()
    return get_file(store.conn, int(row[0])) if row else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Pilot limit counted in exact-match catalog records; omitted for full restore",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    for required in (CATALOG_FILE, SOURCE_MANIFEST_FILE):
        if not required.is_file():
            raise SystemExit(f"legacy catalog input missing: {required.name}")
    config = load_config()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(config=config)
    state = PipelineState(STATE_DIR / "pipeline.sqlite3")
    vectors = NoVectorStore()
    pipeline = IngestPipeline(store, vectors, state, embedder=None)
    counters: Counter[str] = Counter()
    started = time.time()
    processed = 0
    last_report = 0
    try:
        _init_audit(store.conn)
        # Historical manifest is the exact reconciliation contract. Duplicate
        # paths fail closed instead of choosing one implicitly.
        manifest_by_path: dict[str, dict] = {}
        for record in _jsonl(SOURCE_MANIFEST_FILE):
            rel = _safe_relpath(record.get("rel_path"))
            if rel is None:
                continue
            if rel in manifest_by_path:
                raise RuntimeError("source manifest contains a duplicate relative path")
            manifest_by_path[rel] = record

        for catalog in _jsonl(CATALOG_FILE):
            processed += 1
            _store_source_metadata(store.conn, catalog)
            legacy_id = catalog.get("file_id")
            rel = _safe_relpath(catalog.get("rel_path"))
            if legacy_id is None or rel is None:
                _upsert_audit(store.conn, legacy_id or "unknown", str(catalog.get("rel_path") or ""), catalog.get("source_sha256"), None, "invalid_path_or_id")
                counters["invalid_path_or_id"] += 1
                continue
            manifest = manifest_by_path.get(rel)
            if manifest is None:
                _upsert_audit(store.conn, legacy_id, rel, catalog.get("source_sha256"), None, "not_in_source_manifest")
                counters["not_in_source_manifest"] += 1
                continue
            if PurePosixPath(rel).parts and PurePosixPath(rel).parts[0] == "99 废纸篓":
                _upsert_audit(store.conn, legacy_id, rel, catalog.get("source_sha256"), None, "trash_excluded")
                counters["trash_excluded"] += 1
                continue
            if manifest.get("inclusion_status") != "included":
                _upsert_audit(store.conn, legacy_id, rel, catalog.get("source_sha256"), None, "excluded_by_manifest", str(manifest.get("exclusion_reason") or ""))
                counters["excluded_by_manifest"] += 1
                continue
            if manifest.get("symlink"):
                _upsert_audit(store.conn, legacy_id, rel, catalog.get("source_sha256"), None, "symlink_not_followed")
                counters["symlink_not_followed"] += 1
                continue
            catalog_sha = catalog.get("source_sha256")
            manifest_sha = manifest.get("raw_sha256")
            if catalog_sha and manifest_sha and manifest_sha != catalog_sha:
                _upsert_audit(store.conn, legacy_id, rel, catalog_sha, None, "historical_hash_inconsistent")
                counters["historical_hash_inconsistent"] += 1
                continue
            historical_sha = catalog_sha or manifest_sha
            if not historical_sha:
                _upsert_audit(store.conn, legacy_id, rel, None, None, "historical_hash_missing_or_inconsistent")
                counters["historical_hash_missing_or_inconsistent"] += 1
                continue
            path = VAULT_ROOT.joinpath(*PurePosixPath(rel).parts)
            if path.is_symlink():
                _upsert_audit(store.conn, legacy_id, rel, historical_sha, None, "symlink_not_followed")
                counters["symlink_not_followed"] += 1
                continue
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError:
                _upsert_audit(store.conn, legacy_id, rel, historical_sha, None, "missing_in_current_obsidian")
                counters["missing_in_current_obsidian"] += 1
                continue
            try:
                resolved.relative_to(VAULT_ROOT)
            except ValueError:
                _upsert_audit(store.conn, legacy_id, rel, historical_sha, None, "path_escaped_vault")
                counters["path_escaped_vault"] += 1
                continue
            if resolved.is_symlink() or not resolved.is_file():
                _upsert_audit(store.conn, legacy_id, rel, historical_sha, None, "not_regular_file")
                counters["not_regular_file"] += 1
                continue
            observed_sha = _full_sha256(resolved)
            if observed_sha != historical_sha:
                _upsert_audit(store.conn, legacy_id, rel, historical_sha, observed_sha, "sha256_mismatch")
                counters["sha256_mismatch"] += 1
                continue

            # Keep legacy string IDs separate from the integer runtime key. Bind
            # only exact path+full-SHA matches. The business metadata contract
            # for this restore is deliberately context_tag=[] and status=library.
            existing_alias = store.conn.execute(
                "SELECT relative_path,full_sha256,new_file_id FROM legacy_id_alias WHERE namespace=? AND snapshot_id=? AND legacy_file_id=?",
                (NAMESPACE, SNAPSHOT_ID, str(legacy_id)),
            ).fetchone()
            if existing_alias:
                if existing_alias[0] != rel or existing_alias[1] != observed_sha:
                    raise RuntimeError(f"legacy alias source changed for id {legacy_id}")
                doc = get_file(store.conn, int(existing_alias[2]))
                if doc is None or doc.file_path != str(resolved):
                    raise RuntimeError(f"legacy alias target is missing or changed for id {legacy_id}")
                doc_id = doc.file_id
            else:
                doc = _get_active_by_path(store, str(resolved))
                if doc is None:
                    created_at = catalog.get("created_at") or manifest.get("registered_at")
                    doc = store.register_file(
                        resolved,
                        content_hash=observed_sha[:12],
                        size_bytes=resolved.stat().st_size,
                        title=str(catalog.get("title") or resolved.stem),
                        context_tag=[],
                        status="library",
                        created_at=created_at,
                    )
                elif doc.content_hash != observed_sha[:12]:
                    raise RuntimeError(f"existing registry row hash differs for catalog id {legacy_id}")
                _create_alias(store.conn, legacy_id, rel, observed_sha, doc.file_id)
                doc_id = doc.file_id

            result = pipeline.process_file(resolved)
            if result not in ("indexed", "skipped"):
                _upsert_audit(store.conn, legacy_id, rel, historical_sha, observed_sha, "ingest_failed", result)
                counters["ingest_failed"] += 1
                continue
            eligible = resolved.suffix.lower() in (MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS) and resolved.stat().st_size <= 50 * 1024 * 1024
            _upsert_audit(store.conn, legacy_id, rel, historical_sha, observed_sha, "exact_match_indexed" if eligible else "exact_match_registered_only", f"new_file_id={doc_id}; pipeline={result}")
            counters["exact_match"] += 1
            counters["fts_eligible" if eligible else "registered_only"] += 1
            if args.limit is not None and counters["exact_match"] >= args.limit:
                break
            if processed % BATCH_SIZE == 0:
                store.conn.commit()
                state.conn.commit()
                _write_report(counters, processed, started, final=False, limit=args.limit)
                last_report = processed
                print(f"catalog_rows={processed} exact={counters['exact_match']} mismatch={counters['sha256_mismatch']} missing={counters['missing_in_current_obsidian']}", flush=True)
        store.conn.commit()
        state.conn.commit()
        _write_report(counters, processed, started, final=True, limit=args.limit)
        print(json.dumps({"catalog_rows": processed, "counts": dict(counters), "report": str(REPORT_PATH)}, ensure_ascii=False))
    finally:
        vectors.close()
        state.close()
        store.close()
    return 0


def _write_report(
    counters: Counter, processed: int, started: float, *, final: bool,
    limit: int | None,
) -> None:
    report = {
        "status": ("pilot_incomplete" if limit is not None else "complete") if final else "running",
        "namespace": NAMESPACE,
        "snapshot_id": SNAPSHOT_ID,
        "catalog_rows_processed": processed,
        "pilot_exact_match_limit": limit,
        "full_catalog_restored": bool(final and limit is None),
        "counts": dict(counters),
        "full_sha256_required_for_alias": True,
        "embedding": "disabled; FTS only; no placeholder vectors",
        "max_ingest_bytes": 50 * 1024 * 1024,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(REPORT_PATH)


if __name__ == "__main__":
    sys.exit(main())

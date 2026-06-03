#!/usr/bin/env python3
"""Rebuild text + table vector indexes for all Chunk Studio workspaces."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CHUNKING_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT / "data" / "chunk_studio"
GLOBAL_TEXT_INDEX = ROOT / "data" / "index" / "text_chunks"
GLOBAL_TABLE_INDEX = ROOT / "data" / "index" / "table_summaries"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CHUNKING_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNKING_DIR))

from build_table_vector_db import build as build_table_index  # noqa: E402
from build_text_vector_db import DEFAULT_EMBED_MODEL, build as build_text_index  # noqa: E402
from filing_metadata import parse_source_id  # noqa: E402


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        import os

        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def audit_vector_db(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return {"exists": False, "row_count": 0, "valid": False}
    conn = sqlite3.connect(db_path)
    row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if row_count == 0:
        conn.close()
        return {"exists": True, "row_count": 0, "valid": False, "reason": "empty"}
    null_ticker = conn.execute("SELECT COUNT(*) FROM chunks WHERE ticker IS NULL OR ticker = ''").fetchone()[0]
    bad_source = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_file = 'source.pdf'").fetchone()[0]
    groups = conn.execute(
        "SELECT ticker, fiscal_year, COUNT(*) FROM chunks GROUP BY ticker, fiscal_year ORDER BY 3 DESC"
    ).fetchall()
    conn.close()
    valid = null_ticker == 0 and bad_source == 0
    reason = None
    if not valid:
        if bad_source:
            reason = "stale_source_file"
        elif null_ticker:
            reason = "missing_ticker_metadata"
    return {
        "exists": True,
        "row_count": row_count,
        "valid": valid,
        "null_ticker": null_ticker,
        "bad_source": bad_source,
        "groups": groups,
        "reason": reason,
    }


def update_metadata(workspace: Path, *, vector_status: str | None = None, table_vector_status: str | None = None, vector_error: str | None = None, table_vector_error: str | None = None) -> None:
    meta_path = workspace / "metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if vector_status is not None:
        meta["vector_status"] = vector_status
        meta["vector_error"] = vector_error
    if table_vector_status is not None:
        meta["table_vector_status"] = table_vector_status
        meta["table_vector_error"] = table_vector_error
    meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def rebuild_workspace(workspace: Path, *, rebuild_text: bool, rebuild_table: bool, embed_model: str) -> dict[str, Any]:
    meta_path = workspace / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    label = meta.get("original_filename") or workspace.name
    chunks_path = workspace / "chunks.json"
    assets_path = workspace / "assets.json"
    result: dict[str, Any] = {"workspace": workspace.name, "label": label, "text": None, "table": None}

    if rebuild_text:
        if not chunks_path.exists():
            result["text"] = {"status": "skipped", "reason": "missing chunks.json"}
            update_metadata(workspace, vector_status="not_built", vector_error="missing chunks.json")
        else:
            try:
                build_text_index([chunks_path], workspace / "index", rebuild=True, batch_size=64, embed_model=embed_model)
                text_audit = audit_vector_db(workspace / "index" / "vectors.db")
                if text_audit["valid"]:
                    update_metadata(workspace, vector_status="ready", vector_error=None)
                    result["text"] = {"status": "ready", **text_audit}
                else:
                    update_metadata(workspace, vector_status="failed", vector_error=text_audit.get("reason"))
                    result["text"] = {"status": "failed", **text_audit}
            except Exception as exc:
                update_metadata(workspace, vector_status="failed", vector_error=str(exc))
                result["text"] = {"status": "failed", "error": str(exc)}

    if rebuild_table:
        if not assets_path.exists():
            result["table"] = {"status": "skipped", "reason": "missing assets.json"}
            update_metadata(workspace, table_vector_status="not_built", table_vector_error="missing assets.json")
        else:
            try:
                table_dir = workspace / "index" / "table_vectors"
                build_table_index([assets_path], table_dir, rebuild=True, batch_size=64, embed_model=embed_model)
                built_db = table_dir / "vectors.db"
                canonical_db = workspace / "index" / "table_vectors.db"
                canonical_db.parent.mkdir(parents=True, exist_ok=True)
                if built_db.exists():
                    canonical_db.write_bytes(built_db.read_bytes())
                table_audit = audit_vector_db(canonical_db)
                if table_audit["valid"]:
                    update_metadata(workspace, table_vector_status="ready", table_vector_error=None)
                    result["table"] = {"status": "ready", **table_audit}
                else:
                    update_metadata(workspace, table_vector_status="failed", table_vector_error=table_audit.get("reason"))
                    result["table"] = {"status": "failed", **table_audit}
            except Exception as exc:
                update_metadata(workspace, table_vector_status="failed", table_vector_error=str(exc))
                result["table"] = {"status": "failed", "error": str(exc)}

    return result


def discover_workspaces(workspace_dir: Path) -> list[Path]:
    workspaces: list[Path] = []
    for meta_path in sorted(workspace_dir.glob("*/metadata.json")):
        workspace = meta_path.parent
        if (workspace / "chunks.json").exists():
            workspaces.append(workspace)
    return workspaces


def merge_global_indexes(workspaces: list[Path], embed_model: str) -> dict[str, Any]:
    chunk_paths = [ws / "chunks.json" for ws in workspaces if (ws / "chunks.json").exists()]
    asset_paths = [ws / "assets.json" for ws in workspaces if (ws / "assets.json").exists()]
    build_text_index(chunk_paths, GLOBAL_TEXT_INDEX, rebuild=True, batch_size=64, embed_model=embed_model)
    build_table_index(asset_paths, GLOBAL_TABLE_INDEX, rebuild=True, batch_size=64, embed_model=embed_model)
    return {
        "text": audit_vector_db(GLOBAL_TEXT_INDEX / "vectors.db"),
        "table": audit_vector_db(GLOBAL_TABLE_INDEX / "vectors.db"),
        "chunk_files": len(chunk_paths),
        "asset_files": len(asset_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Chunk Studio text/table vector indexes.")
    parser.add_argument("--workspace-dir", type=Path, default=WORKSPACE_DIR)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--no-global-merge", action="store_true")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    load_env_file()
    import os

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("FIREWORKS_API_KEY is required.")

    rebuild_text = not args.table_only
    rebuild_table = not args.text_only
    workspaces = discover_workspaces(args.workspace_dir)

    results = []
    for workspace in workspaces:
        print(f"\n=== Rebuilding {workspace.name} ===")
        results.append(
            rebuild_workspace(
                workspace,
                rebuild_text=rebuild_text,
                rebuild_table=rebuild_table,
                embed_model=args.model,
            )
        )

    merge_result = None
    if not args.no_global_merge and rebuild_text and rebuild_table:
        print("\n=== Merging global indexes ===")
        merge_result = merge_global_indexes(workspaces, args.model)
        from merge_filing_assets import write_merged_assets  # noqa: E402

        print("\n=== Merging global assets ===")
        assets_payload = write_merged_assets(args.workspace_dir, ROOT / "data" / "index" / "merged_assets.json")
        if merge_result is not None:
            merge_result["assets"] = assets_payload.get("counts")

    print(json.dumps({"workspaces": results, "global": merge_result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

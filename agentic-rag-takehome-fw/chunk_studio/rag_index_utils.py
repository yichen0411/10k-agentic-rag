"""Shared helpers for workspace/global RAG index health checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def workspace_table_db_path(workspace: Path) -> Path | None:
    for rel in ("index/table_vectors.db", "index/table_vectors/vectors.db"):
        path = workspace / rel
        if path.is_file():
            return path
    return None


def vector_db_health(db_path: Path | None, *, include_groups: bool = False) -> dict[str, Any]:
    if not db_path or not db_path.is_file():
        return {"exists": False, "row_count": 0, "valid": False, "reason": "missing"}
    conn = sqlite3.connect(db_path)
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if row_count == 0:
            return {"exists": True, "row_count": 0, "valid": False, "reason": "empty"}
        null_ticker = conn.execute("SELECT COUNT(*) FROM chunks WHERE ticker IS NULL OR ticker = ''").fetchone()[0]
        bad_source = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_file = 'source.pdf'").fetchone()[0]
        valid = null_ticker == 0 and bad_source == 0
        reason = None
        if not valid:
            reason = "stale_source_file" if bad_source else "missing_ticker_metadata"
        payload: dict[str, Any] = {
            "exists": True,
            "row_count": row_count,
            "valid": valid,
            "null_ticker": null_ticker,
            "bad_source": bad_source,
            "reason": reason,
        }
        if include_groups:
            payload["groups"] = conn.execute(
                "SELECT ticker, fiscal_year, COUNT(*) FROM chunks GROUP BY ticker, fiscal_year ORDER BY 3 DESC"
            ).fetchall()
        return payload
    finally:
        conn.close()

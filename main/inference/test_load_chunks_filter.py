#!/usr/bin/env python3
"""Unit tests for metadata-filtered chunk loading."""

from __future__ import annotations

import json
import sqlite3
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = Path(__file__).resolve().parent
CHUNKING_DIR = INFERENCE_DIR.parent / "chunking"
for path in [str(ROOT), str(INFERENCE_DIR), str(CHUNKING_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from text_vector_rag_inference import load_chunks  # noqa: E402


def _embedding_blob() -> bytes:
    return array("f", [1.0, 0.0, 0.0]).tobytes()


def _write_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            ticker TEXT,
            fiscal_year TEXT,
            item TEXT,
            section TEXT,
            source_file TEXT,
            page_start INTEGER,
            page_end INTEGER,
            parent_chunk_id TEXT,
            metadata_json TEXT NOT NULL
        )
        """
    )
    rows = [
        ("MSFT_FY2025_10-K.pdf::text_00001", "msft chunk", "MSFT", "FY2025"),
        ("AAPL_FY2025_10-K.pdf::text_00001", "aapl chunk", "AAPL", "FY2025"),
        ("GOOGL_FY2024_10-K.pdf::text_00001", "googl chunk", "GOOGL", "FY2024"),
    ]
    for chunk_id, content, ticker, fiscal_year in rows:
        conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, content, embedding, ticker, fiscal_year, item, section,
                source_file, page_start, page_end, parent_chunk_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, NULL, ?)
            """,
            (chunk_id, content, _embedding_blob(), ticker, fiscal_year, f"{ticker}_{fiscal_year}.pdf", json.dumps({})),
        )
    conn.commit()
    conn.close()


def test_load_chunks_applies_metadata_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.db"
    _write_db(db_path)

    msft_only = load_chunks(db_path, ticker_filter="MSFT")
    assert len(msft_only) == 1
    assert msft_only[0]["ticker"] == "MSFT"

    fy2025 = load_chunks(db_path, fiscal_year_filter="2025")
    assert len(fy2025) == 2
    assert {row["ticker"] for row in fy2025} == {"MSFT", "AAPL"}

    scoped = load_chunks(db_path, ticker_filter=["MSFT", "GOOGL"], fiscal_year_filter="FY2024")
    assert len(scoped) == 1
    assert scoped[0]["ticker"] == "GOOGL"


if __name__ == "__main__":
    import tempfile

    test_load_chunks_applies_metadata_filters(Path(tempfile.mkdtemp()))
    print("ok")

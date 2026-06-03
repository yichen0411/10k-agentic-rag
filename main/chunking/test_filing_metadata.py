#!/usr/bin/env python3
"""Unit tests for filing metadata helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filing_metadata import (  # noqa: E402
    filing_identity,
    normalize_filter_values,
    normalize_fiscal_year,
    parse_source_id,
    patch_workspace_source_files,
    resolve_source_file,
)


def test_parse_source_id() -> None:
    assert parse_source_id("MSFT_FY2025_10-K.pdf") == ("MSFT", "FY2025")
    assert parse_source_id("source.pdf") == (None, None)


def test_resolve_source_file_from_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "demo-workspace"
    workspace.mkdir()
    (workspace / "metadata.json").write_text(
        json.dumps({"original_filename": "AAPL_FY2024_10-K.pdf"}),
        encoding="utf-8",
    )
    chunks_path = workspace / "chunks.json"
    chunks_path.write_text(json.dumps({"source_file": "source.pdf", "chunks": []}), encoding="utf-8")

    assert resolve_source_file("source.pdf", json_path=chunks_path) == "AAPL_FY2024_10-K.pdf"
    assert filing_identity("source.pdf", json_path=chunks_path) == (
        "AAPL_FY2024_10-K.pdf",
        "AAPL",
        "FY2024",
    )


def test_normalize_filters() -> None:
    assert normalize_filter_values("msft", kind="ticker") == ["MSFT"]
    assert normalize_filter_values(["2025"], kind="fiscal_year") == ["FY2025"]
    assert normalize_fiscal_year("FY2024") == "FY2024"


def test_patch_workspace_source_files(tmp_path: Path) -> None:
    workspace = tmp_path / "1779921176-msft-fy2025-10-k"
    workspace.mkdir()
    (workspace / "metadata.json").write_text(
        json.dumps({"original_filename": "MSFT_FY2025_10-K.pdf"}),
        encoding="utf-8",
    )
    (workspace / "chunks.json").write_text(
        json.dumps(
            {
                "source_file": "source.pdf",
                "chunks": [
                    {
                        "chunk_id": "text_00001",
                        "source_file": "source.pdf",
                        "same_section_next_vector_chunk_id": "source.pdf::text_00002",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = patch_workspace_source_files(workspace)
    assert result["patched"] is True
    payload = json.loads((workspace / "chunks.json").read_text(encoding="utf-8"))
    assert payload["source_file"] == "MSFT_FY2025_10-K.pdf"
    assert payload["chunks"][0]["source_file"] == "MSFT_FY2025_10-K.pdf"
    assert payload["chunks"][0]["same_section_next_vector_chunk_id"] == "MSFT_FY2025_10-K.pdf::text_00002"


if __name__ == "__main__":
    test_parse_source_id()
    test_normalize_filters()
    import tempfile

    test_resolve_source_file_from_metadata(Path(tempfile.mkdtemp()))
    test_patch_workspace_source_files(Path(tempfile.mkdtemp()))
    print("ok")

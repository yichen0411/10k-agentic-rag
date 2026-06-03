#!/usr/bin/env python3
"""Unit tests for generalized filing-year resolution."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
for path in [str(AGENT_DIR), str(MAIN_ROOT / "chunking")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from filing_scope import (  # noqa: E402
    comparative_table_retrieval_query,
    extract_metric_years,
    resolve_filing_year_filter,
    should_retry_with_newer_filing,
)

AVAILABLE = ["FY2024", "FY2025"]


def test_extract_metric_years() -> None:
    assert extract_metric_years("Apple selling and marketing in Japan FY2023") == [2023]
    assert extract_metric_years("Compare FY2024 and FY2025 revenue") == [2024, 2025]


def test_resolve_metric_year_older_than_filing_scope() -> None:
    resolved, meta = resolve_filing_year_filter(
        "What was selling and marketing expense in Japan in FY2023?",
        "FY2024",
        available_filing_years=AVAILABLE,
    )
    assert resolved == "FY2025"
    assert meta is not None
    assert meta["reason"] in {
        "metric_year_not_indexed_as_filing",
        "metric_year_older_than_filing_scope",
    }
    assert meta["metric_years"] == [2023]


def test_resolve_metric_year_older_than_requested_filing() -> None:
    resolved, meta = resolve_filing_year_filter(
        "What was revenue in FY2024?",
        "FY2025",
        available_filing_years=["FY2024", "FY2025", "FY2026"],
    )
    assert resolved == "FY2026"
    assert meta is not None
    assert meta["reason"] == "metric_year_older_than_filing_scope"


def test_resolve_metric_year_not_indexed_as_filing() -> None:
    resolved, meta = resolve_filing_year_filter(
        "What did management say about risks in FY2023?",
        "FY2023",
        available_filing_years=AVAILABLE,
    )
    assert resolved == "FY2025"
    assert meta is not None
    assert meta["reason"] in {
        "requested_filing_not_indexed",
        "metric_year_not_indexed_as_filing",
    }


def test_resolve_keeps_current_filing_when_metric_matches() -> None:
    resolved, meta = resolve_filing_year_filter(
        "What did management say about Services in FY2025?",
        "FY2025",
        available_filing_years=AVAILABLE,
    )
    assert resolved == "FY2025"
    assert meta is None


def test_should_retry_with_newer_filing() -> None:
    retry, bump_to = should_retry_with_newer_filing(
        "The provided context does not support a confident answer.",
        "FY2024",
        "What was operating income in Europe in FY2023?",
        available_filing_years=AVAILABLE,
    )
    assert retry is True
    assert bump_to == "FY2025"

    retry, bump_to = should_retry_with_newer_filing(
        "The provided context does not support a confident answer.",
        "FY2025",
        "What was operating income in Europe in FY2023?",
        available_filing_years=AVAILABLE,
    )
    assert retry is False
    assert bump_to is None


def test_comparative_table_retrieval_query_is_generic() -> None:
    query = comparative_table_retrieval_query(
        "What was selling and marketing expense in Japan in FY2023?",
        [2023],
    )
    assert "FY2023" in query
    assert "segment/geographic operating tables" in query
    assert "Note 13" not in query
    assert "Apple" not in query


if __name__ == "__main__":
    test_extract_metric_years()
    test_resolve_metric_year_older_than_filing_scope()
    test_resolve_metric_year_older_than_requested_filing()
    test_resolve_metric_year_not_indexed_as_filing()
    test_resolve_keeps_current_filing_when_metric_matches()
    test_should_retry_with_newer_filing()
    test_comparative_table_retrieval_query_is_generic()
    print("ok")

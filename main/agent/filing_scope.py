"""Resolve RAG filing-year filters from metric years in the user question."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from filing_metadata import normalize_fiscal_year

FISCAL_YEAR_LABEL_RE = re.compile(r"FY(\d{4})", re.I)
METRIC_YEAR_RE = re.compile(r"\b(?:FY)?(20\d{2})\b", re.I)

# Fallback when the vector index path is unavailable (matches indexed 10-K PDF coverage).
DEFAULT_INDEXED_FILING_YEARS: tuple[str, ...] = ("FY2024", "FY2025")


def fiscal_year_to_int(label: str | None) -> int | None:
    if not label:
        return None
    match = FISCAL_YEAR_LABEL_RE.search(str(label).strip())
    return int(match.group(1)) if match else None


def discover_indexed_filing_years(db_path: Path | None = None) -> list[str]:
    """Return distinct fiscal years present in the text-chunk vector index."""
    if db_path and db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT UPPER(fiscal_year)
                FROM chunks
                WHERE fiscal_year IS NOT NULL AND TRIM(fiscal_year) != ''
                ORDER BY fiscal_year
                """
            ).fetchall()
        finally:
            conn.close()
        years = sorted(
            {normalize_fiscal_year(row[0]) for row in rows if row and row[0]},
            key=lambda label: fiscal_year_to_int(label) or 0,
        )
        if years:
            return years
    return list(DEFAULT_INDEXED_FILING_YEARS)


def newest_filing_year(available_filing_years: list[str]) -> str | None:
    if not available_filing_years:
        return None
    return max(available_filing_years, key=lambda label: fiscal_year_to_int(label) or 0)


def extract_metric_years(question: str) -> list[int]:
    years: set[int] = set()
    for match in METRIC_YEAR_RE.finditer(question or ""):
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            years.add(year)
    return sorted(years)


def resolve_filing_year_filter(
    question: str,
    fiscal_year: str | None,
    *,
    available_filing_years: list[str] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Choose which indexed filing PDF to search.

    The rag ``fiscal_year`` parameter selects a filing document, not necessarily the
    fiscal period column requested in the question. Newer 10-K PDFs often include
    two or three years of comparative columns, so an older metric year may require
    searching a newer filing.
    """
    available = list(available_filing_years or DEFAULT_INDEXED_FILING_YEARS)
    requested = normalize_fiscal_year(fiscal_year) if fiscal_year else None
    metric_years = extract_metric_years(question)
    newest = newest_filing_year(available)
    if not newest:
        return requested, None

    available_by_int = {
        year: label for label in available if (year := fiscal_year_to_int(label)) is not None
    }

    if requested and requested not in available:
        return newest, _resolution(
            reason="requested_filing_not_indexed",
            metric_years=metric_years,
            requested_filing_year=requested,
            resolved_filing_year=newest,
            available_filing_years=available,
        )

    if not metric_years:
        return requested, None

    target_metric = min(metric_years)

    if target_metric not in available_by_int and requested != newest:
        return newest, _resolution(
            reason="metric_year_not_indexed_as_filing",
            metric_years=metric_years,
            requested_filing_year=requested,
            resolved_filing_year=newest,
            available_filing_years=available,
        )

    requested_int = fiscal_year_to_int(requested)
    if requested_int is not None and target_metric < requested_int and requested != newest:
        return newest, _resolution(
            reason="metric_year_older_than_filing_scope",
            metric_years=metric_years,
            requested_filing_year=requested,
            resolved_filing_year=newest,
            available_filing_years=available,
        )

    return requested, None


def should_retry_with_newer_filing(
    answer: str | None,
    current_filing_year: str | None,
    question: str,
    *,
    available_filing_years: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Safety net: after an insufficient answer, try the newest indexed filing once."""
    available = list(available_filing_years or DEFAULT_INDEXED_FILING_YEARS)
    newest = newest_filing_year(available)
    if not newest or not extract_metric_years(question):
        return False, None
    if current_filing_year == newest:
        return False, None

    resolved, resolution = resolve_filing_year_filter(
        question,
        current_filing_year,
        available_filing_years=available,
    )
    if resolution and resolved == newest:
        return True, newest

    # Question already mapped to the current filing, but the answer still failed.
    if current_filing_year and current_filing_year != newest:
        return True, newest
    return False, None


def comparative_table_retrieval_query(question: str, metric_years: list[int] | None = None) -> str:
    years = metric_years or extract_metric_years(question)
    year_text = ", ".join(f"FY{year}" for year in years) if years else "the requested fiscal year(s)"
    return (
        "Retrieve comparative financial statement or segment/geographic operating tables that "
        f"disclose the requested metric for {year_text}, including row labels, breakdown "
        f"dimensions, and multi-year columns. Original question: {question.strip()}"
    )


def _resolution(**fields: Any) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}

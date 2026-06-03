"""Score adapter output against golden table cases."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def combined_corpus(tables: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for table in tables:
        parts.append(table.get("text") or "")
        parts.append(table.get("markdown") or "")
        parts.append(table.get("raw_text") or "")
        for row in table.get("rows") or []:
            if isinstance(row, list):
                parts.append(" | ".join(str(c) for c in row if c))
            else:
                parts.append(str(row))
    return normalize_text("\n".join(parts))


def table_covers_pages(tables: list[dict[str, Any]], pages: list[int]) -> bool:
    if not tables or not pages:
        return False
    page_set = set(pages)
    for table in tables:
        t_pages = set(table.get("pages") or [])
        if not t_pages and table.get("page") is not None:
            t_pages = {int(table["page"])}
        if page_set.issubset(t_pages):
            return True
        if table.get("cross_page") and t_pages & page_set:
            # cross-page table touching case pages
            if min(t_pages) <= min(page_set) and max(t_pages) >= max(page_set):
                return True
    return any(
        (table.get("cross_page") or len(set(table.get("pages") or [])) > 1)
        and set(table.get("pages") or []) & set(pages)
        for table in tables
    )


def score_case(case: dict[str, Any], tables: list[dict[str, Any]], corpus: str | None = None) -> dict[str, Any]:
    corpus = corpus if corpus is not None else combined_corpus(tables)
    must_text = case.get("must_find_text") or []
    must_rows = case.get("must_find_rows") or []
    pages = case.get("pages") or []

    text_hits = [term for term in must_text if normalize_text(term) in corpus]
    row_hits = [term for term in must_rows if normalize_text(term) in corpus]
    cross_page_ok = True
    if case.get("cross_page"):
        cross_page_ok = table_covers_pages(tables, pages) or any(
            table.get("cross_page") for table in tables if set(table.get("pages") or []) & set(pages)
        )

    text_score = len(text_hits) / max(len(must_text), 1)
    row_score = len(row_hits) / max(len(must_rows), 1) if must_rows else 1.0
    page_tables = [t for t in tables if set(t.get("pages") or []) & set(pages)]
    overall = (text_score * 0.6 + row_score * 0.2 + (1.0 if cross_page_ok else 0.0) * 0.2) if case.get("cross_page") else (text_score * 0.8 + row_score * 0.2)

    return {
        "case_id": case["id"],
        "title": case.get("title"),
        "score": round(overall, 3),
        "text_hits": text_hits,
        "text_misses": [t for t in must_text if t not in text_hits],
        "row_hits": row_hits,
        "row_misses": [t for t in must_rows if t not in row_hits],
        "cross_page_ok": cross_page_ok,
        "tables_on_pages": len(page_tables),
        "table_count_total": len(tables),
    }


def score_adapter(casebook: dict[str, Any], adapter_result: dict[str, Any]) -> dict[str, Any]:
    cases = casebook.get("cases") or []
    tables = adapter_result.get("tables") or []
    corpus = combined_corpus(tables)
    if adapter_result.get("full_text"):
        corpus = normalize_text(corpus + "\n" + adapter_result["full_text"])
    per_case = [score_case(case, tables, corpus) for case in cases]
    avg = sum(c["score"] for c in per_case) / max(len(per_case), 1)
    return {
        "adapter": adapter_result.get("adapter"),
        "status": adapter_result.get("status", "ok"),
        "error": adapter_result.get("error"),
        "latency_sec": adapter_result.get("latency_sec"),
        "install_note": adapter_result.get("install_note"),
        "applicability": adapter_result.get("applicability"),
        "avg_score": round(avg, 3),
        "cases": per_case,
        "table_count": len(tables),
    }

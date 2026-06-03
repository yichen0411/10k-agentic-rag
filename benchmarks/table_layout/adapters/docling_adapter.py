"""Docling (IBM, MIT) — layout + TableFormer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def run(pdf_path: Path, pages: set[int]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        return {
            "adapter": "docling",
            "status": "skipped",
            "error": str(exc),
            "tables": [],
            "latency_sec": round(time.perf_counter() - t0, 3),
            "install_note": "pip install docling  # first run downloads layout/table models",
            "applicability": "Open weights; strong table structure; full-doc convert can be slow; no 10-K Item map.",
        }

    tables: list[dict[str, Any]] = []
    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document
        for idx, table in enumerate(getattr(doc, "tables", []) or []):
            try:
                df = table.export_to_dataframe(doc=doc)
                md = df.to_markdown(index=False) if hasattr(df, "to_markdown") else str(df)
                text = df.to_csv(index=False) if hasattr(df, "to_csv") else md
            except Exception:
                md = getattr(table, "export_to_markdown", lambda: "")()
                text = md
            prov_pages = _table_pages(table)
            if prov_pages and not set(prov_pages) & pages:
                continue
            if not prov_pages:
                prov_pages = sorted(pages)
            tables.append(
                {
                    "table_id": f"docling_{idx:03d}",
                    "pages": prov_pages,
                    "cross_page": len(prov_pages) > 1,
                    "markdown": md,
                    "raw_text": text,
                    "row_count": len(getattr(table, "rows", []) or []),
                }
            )
    except Exception as exc:
        return {
            "adapter": "docling",
            "status": "error",
            "error": str(exc),
            "tables": tables,
            "latency_sec": round(time.perf_counter() - t0, 3),
            "install_note": "pip install docling",
            "applicability": "See docling docs; page filter may require post-filter on provenance.",
        }

    return {
        "adapter": "docling",
        "status": "ok",
        "tables": tables,
        "latency_sec": round(time.perf_counter() - t0, 3),
        "install_note": "pip install docling",
        "applicability": "Best open-source table structure; still needs section-aware chunking layer for 10-K.",
    }


def _table_pages(table: Any) -> list[int]:
    pages: set[int] = set()
    for attr in ("prov", "provenance", "page_no", "page"):
        val = getattr(table, attr, None)
        if val is None:
            continue
        if isinstance(val, int):
            pages.add(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, int):
                    pages.add(item)
                elif hasattr(item, "page_no"):
                    pages.add(int(item.page_no))
                elif isinstance(item, dict) and item.get("page_no"):
                    pages.add(int(item["page_no"]))
    return sorted(pages)

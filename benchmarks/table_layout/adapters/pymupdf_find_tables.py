"""PyMuPDF page.find_tables() — what we used before layout models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def run(pdf_path: Path, pages: set[int]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        import fitz
    except ImportError as exc:
        return _err("pymupdf_find_tables", exc, t0)

    tables: list[dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    try:
        for page_no in sorted(pages):
            page = doc[page_no - 1]
            found = page.find_tables()
            for idx, tab in enumerate(found.tables or []):
                rows = [[(c or "").strip() or None for c in row] for row in tab.extract()]
                text = "\n".join(" | ".join(c or "" for c in row) for row in rows if any(row))
                tables.append(
                    {
                        "table_id": f"p{page_no}_find_{idx:02d}",
                        "pages": [page_no],
                        "cross_page": False,
                        "row_count": len(rows),
                        "raw_text": text,
                        "markdown": text,
                        "rows": rows,
                        "bbox": list(tab.bbox),
                    }
                )
    finally:
        doc.close()

    return {
        "adapter": "pymupdf_find_tables",
        "status": "ok",
        "tables": tables,
        "latency_sec": round(time.perf_counter() - t0, 3),
        "install_note": "pip install pymupdf",
        "applicability": "Fast baseline; fragments wide tables, misses header-only bands and cross-page merge.",
    }


def _err(name: str, exc: Exception, t0: float) -> dict[str, Any]:
    return {
        "adapter": name,
        "status": "error",
        "error": str(exc),
        "tables": [],
        "latency_sec": round(time.perf_counter() - t0, 3),
    }

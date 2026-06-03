"""Read-only baseline from existing table_pipeline assets.json."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run(pdf_path: Path, pages: set[int], assets_path: Path | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    if not assets_path or not assets_path.exists():
        return {
            "adapter": "baseline_table_pipeline_v2",
            "status": "skipped",
            "error": f"assets not found: {assets_path}",
            "tables": [],
            "latency_sec": round(time.perf_counter() - t0, 3),
            "install_note": "Uses pre-built assets.json from table_pipeline_v2 (no PDF re-parse).",
            "applicability": "Production path for this repo; includes section/subsection linking and cross-page heuristics.",
        }

    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    tables: list[dict[str, Any]] = []
    for table in assets.get("tables", []):
        t_pages = list(range(int(table.get("page_start") or 0), int(table.get("page_end") or 0) + 1))
        if table.get("bbox_by_page"):
            t_pages = sorted({int(c["page"]) for c in table["bbox_by_page"] if c.get("page")})
        if not set(t_pages) & pages:
            continue
        cross = len(t_pages) > 1 or bool(table.get("bbox_by_page") and len(table["bbox_by_page"]) > 1)
        rows = table.get("raw_rows") or []
        text = table.get("raw_text") or ""
        tables.append(
            {
                "table_id": table.get("table_id"),
                "pages": t_pages,
                "cross_page": cross,
                "row_count": table.get("row_count"),
                "markdown": text.replace(" | ", " | "),
                "raw_text": text,
                "rows": rows,
                "subsection": (table.get("subsection_ref") or {}).get("title"),
            }
        )

    return {
        "adapter": "baseline_table_pipeline_v2",
        "status": "ok",
        "tables": tables,
        "latency_sec": round(time.perf_counter() - t0, 3),
        "install_note": "Read-only load of assets.json",
        "applicability": "Best for 10-K + Item/subsection RAG when heuristics are acceptable.",
    }

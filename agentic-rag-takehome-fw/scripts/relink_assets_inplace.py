#!/usr/bin/env python3
"""Re-run table section/subsection relink on existing assets.json (no full re-process)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHUNKING = REPO / "main" / "chunking"
sys.path.insert(0, str(CHUNKING))


def relink_workspace(workspace: Path, pdf_path: Path | None = None) -> dict:
    import fitz
    from section_asset_extractor import (
        annotate_page_ranges,
        collect_heading_positions,
        compact_section_refs,
        load_section_refs,
    )
    from subsection_table_filter import relink_tables_to_subsections

    sections_path = workspace / "sections.json"
    assets_path = workspace / "assets.json"
    if not sections_path.is_file() or not assets_path.is_file():
        raise FileNotFoundError(f"Missing sections.json or assets.json in {workspace}")

    pdf = pdf_path or REPO / "data" / "pdfs" / (json.loads(assets_path.read_text()).get("source_file") or "")
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    tables = assets.get("tables") or []
    images = assets.get("images") or []

    section_refs = load_section_refs(sections_path)
    doc = fitz.open(pdf)
    annotate_page_ranges(section_refs, doc.page_count)
    heading_positions = collect_heading_positions(doc, section_refs)
    doc.close()

    relink_tables_to_subsections(tables, section_refs, heading_positions)
    assets["tables"] = tables
    assets["sections"] = compact_section_refs(section_refs, tables, images)
    assets_path.write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")

    samples = []
    for table in tables:
        if table.get("table_id") in {"table_group_008", "table_group_009"}:
            sec = (table.get("section_ref") or {}).get("item")
            sub = (table.get("subsection_ref") or {}).get("title")
            samples.append(f"{table['table_id']} p{table.get('page_start')} → Item {sec} > {sub}")
    return {"workspace": str(workspace), "tables": len(tables), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d",
    )
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()
    result = relink_workspace(args.workspace, args.pdf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

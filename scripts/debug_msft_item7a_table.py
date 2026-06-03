#!/usr/bin/env python3
"""Debug MSFT FY2025 Item 7A sensitivity table linkage (table_group_008)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHUNKING = REPO / "main" / "chunking"
sys.path.insert(0, str(CHUNKING))

WORKSPACE = REPO / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
PDF = REPO / "data" / "pdfs" / "MSFT_FY2025_10-K.pdf"
SECTIONS = WORKSPACE / "sections.json"
ASSETS = WORKSPACE / "assets.json"


def main() -> None:
    import fitz
    from section_asset_extractor import (
        annotate_page_ranges,
        collect_heading_positions,
        find_section_for_asset,
        find_subsection_for_asset,
        load_section_refs,
    )
    from subsection_table_filter import (
        collect_table_regions,
        relink_tables_to_subsections,
        _filter_heading_positions_for_tables,
    )

    assets = json.loads(ASSETS.read_text(encoding="utf-8"))
    tables = assets.get("tables") or []
    t008 = next((t for t in tables if t.get("table_id") == "table_group_008"), None)
    t009 = next((t for t in tables if t.get("table_id") == "table_group_009"), None)

    print("=== Before relink (stored assets.json) ===")
    for label, table in [("table_group_008 (sensitivity)", t008), ("table_group_009 (income stmt)", t009)]:
        if not table:
            print(f"{label}: MISSING")
            continue
        sec = (table.get("section_ref") or {}).get("item")
        sub = (table.get("subsection_ref") or {}).get("title")
        print(
            f"{label}: p{table.get('page_start')} bbox_y0={table.get('bbox', [None])[1]} "
            f"→ Item {sec} / {sub}"
        )

    item7a = next(
        (s for s in assets.get("sections") or [] if (s.get("item") == "Item 7A")),
        None,
    )
    if item7a:
        print(f"\nItem 7A section table_refs: {item7a.get('table_refs')}")
        sens = next(
            (sub for sub in item7a.get("subsections") or [] if sub.get("title") == "SENSITIVITY ANALYSIS"),
            None,
        )
        print(f"SENSITIVITY ANALYSIS subsection table_refs: {(sens or {}).get('table_refs')}")

    section_refs = load_section_refs(SECTIONS)
    doc = fitz.open(PDF)
    annotate_page_ranges(section_refs, doc.page_count)
    heading_positions = collect_heading_positions(doc, section_refs)
    doc.close()

    overlap = [r for r in section_refs if r["start_pdf_page"] == 47]
    print("\n=== Page 47 section overlap ===")
    for ref in overlap:
        print(f"  {ref['item']}: start={ref['start_pdf_page']} end={ref['end_pdf_page']}")

    if t008 and t008.get("bbox"):
        y0 = float(t008["bbox"][1])
        sec = find_section_for_asset(section_refs, 47, y0)
        print(f"\nfind_section_for_asset(47, y0={y0}): {sec and sec.get('item')}")

    relink_tables_to_subsections(tables, section_refs, heading_positions)
    t008 = next((t for t in tables if t.get("table_id") == "table_group_008"), None)
    t009 = next((t for t in tables if t.get("table_id") == "table_group_009"), None)

    print("\n=== After relink (with fix) ===")
    for label, table in [("table_group_008", t008), ("table_group_009", t009)]:
        sec = (table.get("section_ref") or {}).get("item")
        sub = (table.get("subsection_ref") or {}).get("title")
        path = [sec, sub]
        print(f"{label}: Item {sec} > {sub}  (path: {' > '.join(p for p in path if p)})")

    regions = collect_table_regions(tables)
    filtered = _filter_heading_positions_for_tables(heading_positions, regions)
    item7a_headings = [
        h for hs in filtered.values() for h in hs if h.get("section_ref_id") == "Part II::Item 7A"
    ]
    print(f"\nItem 7A headings after table filter: {[h['title'] for h in item7a_headings]}")


if __name__ == "__main__":
    main()

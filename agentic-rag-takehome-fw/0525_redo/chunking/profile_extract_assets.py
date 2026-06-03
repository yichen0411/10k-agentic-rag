#!/usr/bin/env python3
"""Profile build_asset_payload / extracting_assets step."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fitz

CHUNKING_DIR = Path(__file__).resolve().parent
ROOT = CHUNKING_DIR.parents[1]
sys.path.insert(0, str(CHUNKING_DIR))

from section_asset_extractor import (  # noqa: E402
    annotate_page_ranges,
    annotate_table_text_anchors,
    build_asset_payload,
    collect_heading_positions,
    extract_images,
    load_section_refs,
)
from subsection_table_filter import filter_subsections_in_tables  # noqa: E402
from table_pipeline import (  # noqa: E402
    TableGroupIdAllocator,
    _link_pass,
    collect_table_regions,
    dedupe_contained_regions,
    enrich_regions_from_words,
)
from toc_guided_section_probe import build_line_records, collect_pages  # noqa: E402

WORKSPACES = [
    (
        "AAPL FY2025",
        ROOT / "data/chunk_studio/1780210521-aapl-fy2025-10-k-74ba618f81/source.pdf",
        ROOT / "data/chunk_studio/1780210521-aapl-fy2025-10-k-74ba618f81/sections.json",
    ),
    (
        "MSFT FY2025",
        ROOT / "data/chunk_studio/1779921176-msft-fy2025-10-k-8d505c867d/source.pdf",
        ROOT / "data/chunk_studio/1779921176-msft-fy2025-10-k-8d505c867d/sections.json",
    ),
]


def tick(label: str, start: float, total: float) -> float:
    now = time.perf_counter()
    print(f"  {now - start:6.1f}s  cum {now - total:6.1f}s  {label}", flush=True)
    return now


def profile_one(label: str, pdf_path: Path, sections_path: Path) -> None:
    print(f"\n=== {label} ===", flush=True)
    total = time.perf_counter()
    t = total

    doc = fitz.open(pdf_path)
    print(f"  pages: {doc.page_count}", flush=True)

    section_refs = load_section_refs(sections_path)
    annotate_page_ranges(section_refs, doc.page_count)
    heading_positions = collect_heading_positions(doc, section_refs)
    pages = collect_pages(doc)
    full_text, records = build_line_records(pages)
    sections_payload = json.loads(sections_path.read_text(encoding="utf-8"))
    t = tick("setup", t, total)

    t_find = time.perf_counter()
    n_raw = sum(len(doc[i].find_tables().tables) for i in range(doc.page_count))
    tick(f"find_tables() x{doc.page_count} pages -> {n_raw} raw", t_find, total)

    raw_regions = collect_table_regions(doc, section_refs, heading_positions)
    t = tick(f"collect_table_regions -> {len(raw_regions)}", t, total)

    sections_payload, stats = filter_subsections_in_tables(
        sections_payload, full_text, records, raw_regions
    )
    t = tick(
        f"filter_subsections_in_tables (removed {stats['removed_subsections']})",
        t,
        total,
    )

    section_refs = load_section_refs(sections_path)
    annotate_page_ranges(section_refs, doc.page_count)
    heading_positions = collect_heading_positions(doc, section_refs)
    allocator = TableGroupIdAllocator()
    raw = collect_table_regions(doc, section_refs, heading_positions)
    linked = _link_pass(raw, doc, cross_page=False, allocator=allocator)
    t = tick(f"2nd collect + same-page link -> {len(linked)}", t, total)

    enriched = enrich_regions_from_words(doc, linked)
    t = tick(f"word-layer rescan x{len(enriched)}", t, total)

    deduped = dedupe_contained_regions(enriched)
    relinked = _link_pass(deduped, doc, cross_page=False, allocator=allocator)
    t = tick(f"dedupe + relink -> {len(relinked)}", t, total)

    final = _link_pass(relinked, doc, cross_page=True, allocator=allocator)
    t = tick(f"cross-page link -> {len(final)} tables", t, total)

    annotate_table_text_anchors(doc, section_refs, final)
    t = tick("annotate_table_text_anchors", t, total)

    images = extract_images(doc, section_refs)
    t = tick(f"extract_images ({len(images)})", t, total)
    doc.close()

    print(f"  TOTAL decomposed: {time.perf_counter() - total:.1f}s", flush=True)

    t2 = time.perf_counter()
    payload = build_asset_payload(pdf_path, sections_path)
    print(
        f"  build_asset_payload: {time.perf_counter() - t2:.1f}s "
        f"-> {len(payload['tables'])} tables",
        flush=True,
    )


def main() -> None:
    for label, pdf, sections in WORKSPACES:
        if pdf.exists() and sections.exists():
            profile_one(label, pdf, sections)
        else:
            print(f"skip {label}: missing files", flush=True)


if __name__ == "__main__":
    main()

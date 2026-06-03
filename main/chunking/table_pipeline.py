"""Unified table region pipeline for 10-K PDF extraction.

Flow:
  1. collect           — find_tables + synthetic page-bottom headers + continuations
  2. link (same-page)  — merge find_tables fragments before rescan
  3. enrich            — word-layer rescan to fill missing rows
  4. dedupe            — drop fragments absorbed by a larger region on the same page
  5. link (cross-page) — merge header-at-bottom + data-at-top groups
  6. emit              — singles or table_group with bbox_by_page
"""

from __future__ import annotations

from typing import Any

import fitz

from section_asset_extractor import (
    PageTableFindCache,
    PageWordCache,
    extract_tables,
    looks_like_header_row,
    rescan_table_rows_from_words,
    row_text,
    same_page_fragment_candidate,
    should_merge_tables,
    supplement_header_only_continuations,
    synthesize_page_bottom_header_tables,
    table_complexity,
)


class TableGroupIdAllocator:
    """Monotonic table_group_* IDs shared across same-page and cross-page link passes."""

    def __init__(self) -> None:
        self._next = 1
        self._used: set[str] = set()

    def register(self, table_id: str) -> None:
        self._used.add(table_id)
        if table_id.startswith("table_group_"):
            try:
                num = int(table_id.rsplit("_", 1)[-1])
            except ValueError:
                return
            self._next = max(self._next, num + 1)

    def next_id(self) -> str:
        while True:
            candidate = f"table_group_{self._next:03d}"
            self._next += 1
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate


def collect_table_regions(
    doc: fitz.Document,
    section_refs: list[dict[str, Any]],
    heading_positions: dict[str, list[dict[str, Any]]],
    *,
    table_find_cache: PageTableFindCache | None = None,
    word_cache: PageWordCache | None = None,
) -> list[dict[str, Any]]:
    regions = extract_tables(
        doc,
        section_refs,
        heading_positions=heading_positions,
        table_find_cache=table_find_cache,
    )
    regions = synthesize_page_bottom_header_tables(doc, regions, section_refs, heading_positions)
    regions = supplement_header_only_continuations(doc, regions, section_refs, heading_positions)
    return regions


def enrich_regions_from_words(
    doc: fitz.Document,
    regions: list[dict[str, Any]],
    *,
    word_cache: PageWordCache | None = None,
) -> list[dict[str, Any]]:
    return [rescan_table_rows_from_words(doc, region, word_cache=word_cache) for region in regions]


def _skip_dedupe(region: dict[str, Any]) -> bool:
    if region.get("header_only") or region.get("pending_merge") or region.get("synthetic_header_band"):
        return True
    return region.get("asset_type") in {"table_header_band", "table_continuation"}


def dedupe_contained_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove smaller regions whose vertical span is contained in a larger sibling."""
    absorbed: set[str] = set()
    candidates = [r for r in regions if not _skip_dedupe(r)]
    by_page_sub: dict[tuple[int, str | None], list[dict[str, Any]]] = {}
    for region in candidates:
        sub_id = (region.get("subsection_ref") or {}).get("subsection_ref_id")
        by_page_sub.setdefault((region["page_start"], sub_id), []).append(region)

    for group in by_page_sub.values():
        ordered = sorted(group, key=lambda r: r["bbox"][3] - r["bbox"][1], reverse=True)
        for idx, big in enumerate(ordered):
            by0, by1 = big["bbox"][1], big["bbox"][3]
            for small in ordered[idx + 1 :]:
                if small["table_id"] in absorbed:
                    continue
                sy0, sy1 = small["bbox"][1], small["bbox"][3]
                if sy0 >= by0 - 2 and sy1 <= by1 + 2:
                    absorbed.add(small["table_id"])

    if not absorbed:
        return regions
    return [r for r in regions if r["table_id"] not in absorbed]


def regions_should_link_same_page(prev: dict[str, Any], nxt: dict[str, Any]) -> bool:
    if nxt["page_start"] != prev["page_start"]:
        return False
    return same_page_fragment_candidate(prev, nxt)


def regions_should_link_cross_page(prev: dict[str, Any], nxt: dict[str, Any], doc: fitz.Document) -> bool:
    if nxt["page_start"] != prev["page_end"] + 1:
        return False
    ok, _, _ = should_merge_tables(prev, nxt, doc)
    return ok


def regions_should_link(prev: dict[str, Any], nxt: dict[str, Any], doc: fitz.Document) -> bool:
    return regions_should_link_same_page(prev, nxt) or regions_should_link_cross_page(prev, nxt, doc)


def _combine_rows(group: list[dict[str, Any]]) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for idx, region in enumerate(group):
        region_rows = list(region.get("raw_rows") or [])
        if idx > 0 and region_rows and looks_like_header_row(region_rows[0]):
            region_rows = region_rows[1:]
        rows.extend(region_rows)
    return rows


def _materialize_group(group: list[dict[str, Any]], allocator: TableGroupIdAllocator) -> dict[str, Any]:
    if len(group) == 1:
        region = dict(group[0])
        region.pop("continued_from", None)
        region.pop("continued_to", None)
        region.pop("pending_merge", None)
        allocator.register(region["table_id"])
        return region

    rows = _combine_rows(group)
    cross_page = group[0]["page_start"] != group[-1]["page_end"]
    bbox = [
        min(r["bbox"][0] for r in group),
        min(r["bbox"][1] for r in group),
        max(r["bbox"][2] for r in group),
        max(r["bbox"][3] for r in group),
    ]
    score, reasons = table_complexity(rows, bbox)
    header = group[0]

    payload: dict[str, Any] = {
        "table_id": allocator.next_id(),
        "asset_type": "table_group",
        "page_start": header["page_start"],
        "page_end": group[-1]["page_end"],
        "source_table_ids": [r["table_id"] for r in group],
        "bbox": [round(x, 2) for x in bbox],
        "row_count": len(rows),
        "col_count": max((len(row) for row in rows), default=0),
        "raw_rows": rows,
        "raw_text": "\n".join(row_text(row) for row in rows),
        "first_row": rows[0] if rows else [],
        "last_row": rows[-1] if rows else [],
        "complexity_score": score,
        "complexity_reasons": reasons + ["linked_table_group"],
        "complexity": "complex" if score >= 5 else "simple",
        "section_ref": header.get("section_ref"),
        "subsection_ref": header.get("subsection_ref"),
        "header_only": False,
        "pending_merge": False,
    }

    if cross_page:
        payload["bbox_by_page"] = [
            {"page": int(r["page_start"]), "bbox": [round(x, 2) for x in r["bbox"]]}
            for r in group
        ]
        payload["subsection_attribution"] = "header_page"
        payload.pop("bbox", None)

    return payload


def _link_pass(
    regions: list[dict[str, Any]],
    doc: fitz.Document,
    *,
    cross_page: bool,
    allocator: TableGroupIdAllocator,
) -> list[dict[str, Any]]:
    ordered = sorted(regions, key=lambda r: (r["page_start"], r["bbox"][1], r["bbox"][0]))
    groups: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(ordered):
        group = [ordered[idx]]
        idx += 1
        while idx < len(ordered):
            prev, nxt = group[-1], ordered[idx]
            if cross_page:
                ok = regions_should_link_cross_page(prev, nxt, doc)
            else:
                ok = regions_should_link_same_page(prev, nxt)
            if not ok:
                break
            group.append(nxt)
            idx += 1
        groups.append(group)

    return [_materialize_group(group, allocator) for group in groups]


def run_table_pipeline(
    doc: fitz.Document,
    section_refs: list[dict[str, Any]],
    heading_positions: dict[str, list[dict[str, Any]]],
    *,
    table_find_cache: PageTableFindCache | None = None,
    word_cache: PageWordCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    allocator = TableGroupIdAllocator()
    raw = collect_table_regions(
        doc, section_refs, heading_positions, table_find_cache=table_find_cache
    )
    linked_same_page = _link_pass(raw, doc, cross_page=False, allocator=allocator)
    enriched = enrich_regions_from_words(doc, linked_same_page, word_cache=word_cache)
    deduped = dedupe_contained_regions(enriched)
    # Rescan expands bboxes and can split previously mergeable same-page stacks.
    relinked_same_page = _link_pass(deduped, doc, cross_page=False, allocator=allocator)
    final = _link_pass(relinked_same_page, doc, cross_page=True, allocator=allocator)
    seen_ids = {table["table_id"] for table in final}
    if len(seen_ids) != len(final):
        raise RuntimeError("table_pipeline produced duplicate table_id values")
    stats = {
        "raw_regions": len(raw),
        "after_same_page_link": len(linked_same_page),
        "after_dedupe": len(deduped),
        "after_relink_same_page": len(relinked_same_page),
        "final_tables": len(final),
        "table_groups": sum(1 for t in final if t.get("asset_type") == "table_group"),
    }
    return final, stats

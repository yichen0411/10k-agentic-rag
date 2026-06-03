"""Post-process font-detected subsections: drop headings that fall inside table bboxes."""

from __future__ import annotations

from typing import Any

from toc_guided_section_probe import build_subsection_path, clean_text


def collect_table_regions(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for table in tables:
        if table.get("bbox_by_page"):
            for crop in table["bbox_by_page"]:
                page = int(crop.get("page") or 0)
                bbox = crop.get("bbox")
                if page and bbox and len(bbox) >= 4:
                    regions.append({"page": page, "bbox": [float(x) for x in bbox]})
            continue
        bbox = table.get("bbox")
        page = int(table.get("page_start") or table.get("page") or 0)
        if page and bbox and len(bbox) >= 4:
            regions.append({"page": page, "bbox": [float(x) for x in bbox]})
    return regions


def _line_bbox(record: dict[str, Any]) -> list[float] | None:
    x0, y0, x1, y1 = record.get("x0"), record.get("y0"), record.get("x1"), record.get("y1")
    if None in (x0, y0, x1, y1):
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def heading_inside_table_region(
    page: int,
    line_bbox: list[float],
    regions: list[dict[str, Any]],
    *,
    y_enter_margin: float = 4.0,
    x_pad: float = 6.0,
) -> bool:
    """True when the heading line sits inside a table's body (not a caption above it)."""
    lx0, ly0, lx1, ly1 = line_bbox
    line_cy = (ly0 + ly1) / 2.0
    for region in regions:
        if int(region["page"]) != int(page):
            continue
        tx0, ty0, tx1, ty1 = region["bbox"]
        table_x0, table_x1 = tx0 - x_pad, tx1 + x_pad
        if lx1 < table_x0 or lx0 > table_x1:
            continue
        # Caption / statement titles sit above the table top — keep those.
        if ly1 <= ty0 - 1.0:
            continue
        if line_cy >= ty0 + y_enter_margin and ly0 <= ty1 + 2.0:
            return True
    return False


def _record_for_heading(records: list[dict[str, Any]], heading: dict[str, Any]) -> dict[str, Any] | None:
    offset = heading.get("offset")
    page = heading.get("page")
    title = heading.get("title")
    for record in records:
        if record.get("offset") == offset and record.get("page") == page:
            return record
    for record in records:
        if record.get("page") == page and record.get("line") == title:
            return record
    return None


def _rebuild_subsection_chunks(
    item: dict[str, Any],
    headings: list[dict[str, Any]],
    full_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stack: dict[int, str] = {}
    chunks: list[dict[str, Any]] = []
    section_end = int(item["end_offset"])
    for idx, candidate in enumerate(headings):
        chunk_start = int(candidate["offset"])
        chunk_end = int(headings[idx + 1]["offset"]) if idx + 1 < len(headings) else section_end
        chunk_text = clean_text(full_text[chunk_start:chunk_end])
        path = build_subsection_path(stack, int(candidate["level"]), candidate["title"])
        chunks.append(
            {
                **candidate,
                "path": path,
                "end_offset": chunk_end,
                "char_count": len(chunk_text),
                "text": chunk_text,
            }
        )
    return headings, chunks


def filter_subsections_in_tables(
    sections_payload: dict[str, Any],
    full_text: str,
    records: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    regions = collect_table_regions(tables)
    removed: list[dict[str, Any]] = []
    kept_total = 0

    for part in sections_payload.get("parts") or []:
        for item in part.get("items") or []:
            kept_headings: list[dict[str, Any]] = []
            for heading in item.get("subsection_headings") or []:
                record = _record_for_heading(records, heading)
                line_bbox = _line_bbox(record) if record else None
                if line_bbox and heading_inside_table_region(int(heading["page"]), line_bbox, regions):
                    removed.append(
                        {
                            "item": item.get("item"),
                            "title": heading.get("title"),
                            "page": heading.get("page"),
                        }
                    )
                    continue
                kept_headings.append(heading)
            kept_total += len(kept_headings)
            headings, chunks = _rebuild_subsection_chunks(item, kept_headings, full_text)
            item["subsection_headings"] = headings
            item["subsection_chunks"] = chunks
            item["subsections"] = chunks
            if headings:
                item["preamble_text"] = clean_text(full_text[int(item["start_offset"]) : int(headings[0]["offset"])])
            else:
                item["preamble_text"] = clean_text(full_text[int(item["start_offset"]) : int(item["end_offset"])])

    subsection_counts: dict[str, int] = {}
    for part in sections_payload.get("parts") or []:
        for item in part.get("items") or []:
            item_key = item.get("item")
            if item_key:
                subsection_counts[item_key] = len(item.get("subsection_headings") or [])

    sections_payload["subsection_counts"] = subsection_counts
    stats = {
        "table_regions": len(regions),
        "removed_subsections": len(removed),
        "kept_subsections": kept_total,
        "removed_samples": removed[:40],
    }
    return sections_payload, stats


def _filter_heading_positions_for_tables(
    heading_positions: dict[str, list[dict[str, Any]]],
    regions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Drop heading lines that are table row labels inside a detected table bbox."""
    filtered: dict[str, list[dict[str, Any]]] = {}
    for page_key, headings in heading_positions.items():
        kept: list[dict[str, Any]] = []
        for heading in headings:
            y0 = float(heading.get("y0") or 0)
            line_bbox = [42.0, y0, 570.0, y0 + 11.0]
            if heading_inside_table_region(int(heading["page"]), line_bbox, regions):
                continue
            kept.append(heading)
        filtered[page_key] = kept
    return filtered


def relink_tables_to_subsections(
    tables: list[dict[str, Any]],
    section_refs: list[dict[str, Any]],
    heading_positions: dict[str, list[dict[str, Any]]],
) -> None:
    from section_asset_extractor import (
        find_section_for_asset,
        find_subsection_for_asset,
        section_metadata,
        subsection_metadata,
    )

    regions = collect_table_regions(tables)
    filtered_headings = _filter_heading_positions_for_tables(heading_positions, regions)

    for table in tables:
        bbox = table.get("bbox")
        page = int(table.get("page_start") or 0)
        if not bbox and table.get("bbox_by_page"):
            crop = table["bbox_by_page"][0]
            bbox = crop.get("bbox")
            page = int(crop.get("page") or page)
        if not bbox or not page:
            continue
        y0 = float(bbox[1])
        section = find_section_for_asset(section_refs, page, y0)
        if section is None:
            continue
        subsection = find_subsection_for_asset(section, filtered_headings, page, y0)
        table["section_ref"] = section_metadata(section)
        table["subsection_ref"] = subsection_metadata(subsection)

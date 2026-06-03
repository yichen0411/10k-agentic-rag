#!/usr/bin/env python3
"""Export subsection hierarchy: (1) current font method, (2) Layout+TOC+font-fallback hybrid."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parents[1]
CHUNKING_DIR = REPO_ROOT / "main" / "chunking"
sys.path.insert(0, str(CHUNKING_DIR))

from toc_guided_section_probe import build_subsection_path, subsection_level  # noqa: E402

from compare_layout_toc import (  # noqa: E402
    DEFAULT_PDF,
    DEFAULT_SECTIONS,
    assign_items,
    box_text,
    clean_text,
    extract_layout_headings,
    is_layout_noise,
    load_font_headings,
    match_heading,
    title_similarity,
)

ITEM_LINE_RE = re.compile(r"^\s*ITEM\s+\d", re.I)


def is_item_boundary_title(title: str) -> bool:
    return bool(ITEM_LINE_RE.match(clean_text(title)))


def infer_level_from_box(box: dict[str, Any], title: str) -> int:
    spans: list[dict[str, Any]] = []
    for line in box.get("textlines") or []:
        spans.extend(line.get("spans") or [])
    if not spans:
        return subsection_level({"line": title, "is_bold": False, "is_italic": False})
    fonts = [s.get("font") or "" for s in spans]
    flags = [int(s.get("flags") or 0) for s in spans]
    is_bold = any("bold" in f.lower() for f in fonts) or any(f & 16 for f in flags)
    is_italic = any("italic" in f.lower() or "oblique" in f.lower() for f in fonts) or any(f & 2 for f in flags)
    bbox = (box.get("textlines") or [{}])[0].get("bbox") or [0, 0, 0, 0]
    page_width = 612.0
    line_center = (float(bbox[0]) + float(bbox[2])) / 2.0
    page_center = page_width / 2.0
    alpha = [c for c in title if c.isalpha()]
    all_caps = bool(alpha) and sum(1 for c in alpha if c.isupper()) / len(alpha) >= 0.85
    centered_score = 0
    if abs(line_center - page_center) <= max(48.0, page_width * 0.09):
        centered_score += 2
    if all_caps:
        centered_score += 2
    record = {
        "line": title,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "max_size": max(float(s.get("size") or 10) for s in spans),
    }
    if centered_score >= 4:
        return 2
    return subsection_level(record)


def load_item_sections(sections_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(sections_path.read_text(encoding="utf-8"))
    sections: list[dict[str, Any]] = []
    for part in payload.get("parts") or []:
        for section in part.get("items") or part.get("sections") or []:
            sections.append(
                {
                    "part": section.get("part") or part.get("part"),
                    "item": section.get("item"),
                    "item_title": section.get("matched_title") or section.get("toc_title") or "",
                    "start_pdf_page": section.get("start_pdf_page"),
                    "subsection_headings": section.get("subsection_headings") or [],
                }
            )
    return sections


def attach_paths(headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stack: dict[int, str] = {}
    out: list[dict[str, Any]] = []
    for h in headings:
        level = int(h.get("level") or 2)
        path = build_subsection_path(stack, level, h["title"])
        out.append({**h, "level": level, "path": path})
    return out


def group_by_level(headings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {f"L{i}": [] for i in range(1, 5)}
    for h in headings:
        lvl = int(h.get("level") or 2)
        key = f"L{lvl}" if 1 <= lvl <= 4 else "L4"
        grouped[key].append(
            {
                "title": h["title"],
                "page": h.get("page"),
                "path": h.get("path") or [h["title"]],
                "source": h.get("source"),
            }
        )
    return grouped


def build_font_hierarchy(sections_path: Path) -> dict[str, Any]:
    items_out: list[dict[str, Any]] = []
    for section in load_item_sections(sections_path):
        item = section.get("item")
        raw = [
            {
                "title": h.get("title") or "",
                "level": h.get("level"),
                "page": h.get("page"),
                "source": "font",
            }
            for h in section.get("subsection_headings") or []
        ]
        with_paths = attach_paths(raw)
        items_out.append(
            {
                "part": section.get("part"),
                "item": item,
                "item_title": section.get("item_title"),
                "item_page": section.get("start_pdf_page"),
                "subsection_count": len(with_paths),
                "by_level": group_by_level(with_paths),
                "reading_order": [
                    {
                        "level": h["level"],
                        "page": h.get("page"),
                        "path": h.get("path"),
                        "title": h["title"],
                        "source": h.get("source"),
                    }
                    for h in with_paths
                ],
            }
        )
    return {
        "method": "font_heuristics (current code)",
        "description": "TOC Item offset + per-line font/bold/italic/centered rules in toc_guided_section_probe.py",
        "document": sections_path.parent.name,
        "item_count": len(items_out),
        "total_subsections": sum(x["subsection_count"] for x in items_out),
        "items": items_out,
    }


def extract_layout_rich(pdf_path: Path, cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("headings") and cached["headings"][0].get("y0") is not None:
            return cached["headings"]

    import pymupdf4llm

    raw = pymupdf4llm.to_json(str(pdf_path), header=False, footer=False)
    data = json.loads(raw)
    headings: list[dict[str, Any]] = []
    for page in data.get("pages") or []:
        page_no = int(page.get("page_number") or 0)
        for box in page.get("boxes") or []:
            if (box.get("boxclass") or box.get("class")) != "section-header":
                continue
            title = box_text(box)
            if not title or len(title) > 120 or is_layout_noise(title):
                continue
            bbox = (box.get("textlines") or [{}])[0].get("bbox") or [0, 0, 0, 0]
            headings.append(
                {
                    "source": "layout",
                    "item": None,
                    "title": title,
                    "page": page_no,
                    "y0": float(bbox[1]),
                    "level": infer_level_from_box(box, title),
                }
            )
    headings.sort(key=lambda h: (h["page"], h["y0"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"headings": headings}, ensure_ascii=False), encoding="utf-8")
    return headings


def build_hybrid_headings_for_item(
    item: str,
    page_lo: int,
    page_hi: int,
    layout_headings: list[dict[str, Any]],
    font_headings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    layout_in = [
        h
        for h in layout_headings
        if page_lo <= int(h["page"]) <= page_hi and not is_item_boundary_title(h["title"])
    ]
    font_in = [h for h in font_headings if h.get("item") == item]
    used_font_idx: set[int] = set()
    merged: list[dict[str, Any]] = []

    for lh in layout_in:
        idx = match_heading(lh, font_in, used_font_idx)
        if idx is not None:
            used_font_idx.add(idx)
            fh = font_in[idx]
            merged.append(
                {
                    "title": fh["title"],
                    "level": fh["level"],
                    "page": lh["page"],
                    "y0": lh.get("y0", 0),
                    "source": "layout",
                }
            )
        else:
            merged.append(
                {
                    "title": lh["title"],
                    "level": lh.get("level") or 2,
                    "page": lh["page"],
                    "y0": lh.get("y0", 0),
                    "source": "layout_only",
                }
            )

    for idx, fh in enumerate(font_in):
        if idx in used_font_idx:
            continue
        merged.append(
            {
                "title": fh["title"],
                "level": fh["level"],
                "page": fh["page"],
                "y0": 10**9,
                "source": "font_fallback",
            }
        )

    merged.sort(key=lambda h: (int(h["page"]), float(h.get("y0") or 0)))
    return merged


def build_hybrid_hierarchy(
    pdf_path: Path,
    sections_path: Path,
    layout_cache: Path,
) -> dict[str, Any]:
    font_all = load_font_headings(sections_path)
    bounds = assign_items(sections_path)
    layout_all = extract_layout_rich(pdf_path, layout_cache)

    items_out: list[dict[str, Any]] = []
    for section in load_item_sections(sections_path):
        item = section.get("item") or ""
        page_lo, page_hi = bounds.get(item, (0, 10**9))
        merged = build_hybrid_headings_for_item(item, page_lo, page_hi, layout_all, font_all)
        with_paths = attach_paths(merged)
        sources = {"layout": 0, "layout_only": 0, "font_fallback": 0}
        for h in with_paths:
            sources[h.get("source") or "layout"] = sources.get(h.get("source") or "layout", 0) + 1

        items_out.append(
            {
                "part": section.get("part"),
                "item": item,
                "item_title": section.get("item_title"),
                "item_page": section.get("start_pdf_page"),
                "page_range": [page_lo, page_hi],
                "subsection_count": len(with_paths),
                "sources": sources,
                "by_level": group_by_level(with_paths),
                "reading_order": [
                    {
                        "level": h["level"],
                        "page": h.get("page"),
                        "path": h.get("path"),
                        "title": h["title"],
                        "source": h.get("source"),
                    }
                    for h in with_paths
                ],
            }
        )

    return {
        "method": "layout_hybrid (recommended)",
        "description": "TOC Item page range + PyMuPDF-Layout section-header primary + font heuristics fallback",
        "document": sections_path.parent.name,
        "item_count": len(items_out),
        "total_subsections": sum(x["subsection_count"] for x in items_out),
        "items": items_out,
    }


def main() -> None:
    out_dir = BENCH_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "layout_headings_rich_cache.json"

    font_payload = build_font_hierarchy(DEFAULT_SECTIONS)
    hybrid_payload = build_hybrid_hierarchy(DEFAULT_PDF, DEFAULT_SECTIONS, cache)

    font_path = out_dir / "font_subsection_hierarchy.json"
    hybrid_path = out_dir / "layout_hybrid_subsection_hierarchy.json"
    font_path.write_text(json.dumps(font_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    hybrid_path.write_text(json.dumps(hybrid_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {font_path}")
    print(f"  items={font_payload['item_count']} subsections={font_payload['total_subsections']}")
    print(f"Wrote {hybrid_path}")
    print(f"  items={hybrid_payload['item_count']} subsections={hybrid_payload['total_subsections']}")

    item7_font = next((x for x in font_payload["items"] if x["item"] == "Item 7"), None)
    item7_hybrid = next((x for x in hybrid_payload["items"] if x["item"] == "Item 7"), None)
    if item7_font and item7_hybrid:
        print("\nItem 7 by_level preview:")
        for lvl in ("L1", "L2", "L3", "L4"):
            fc = len(item7_font["by_level"].get(lvl) or [])
            hc = len(item7_hybrid["by_level"].get(lvl) or [])
            print(f"  {lvl}: font={fc}  hybrid={hc}")


if __name__ == "__main__":
    main()

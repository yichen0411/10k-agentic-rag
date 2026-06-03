#!/usr/bin/env python3
"""Compare subsection heading detection: PyMuPDF-Layout vs font heuristics vs TOC+hybrid."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parents[1]
CHUNKING_DIR = REPO_ROOT / "0525_redo" / "chunking"
DEFAULT_PDF = REPO_ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d" / "source.pdf"
DEFAULT_SECTIONS = DEFAULT_PDF.parent / "sections.json"

PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
RUNNING_HEADER_RE = re.compile(r"^part\s+[ivx]+.*item\s+\d", re.I)


def is_layout_noise(title: str) -> bool:
    t = clean_text(title)
    if not t or PAGE_NUMBER_RE.fullmatch(t):
        return True
    if RUNNING_HEADER_RE.search(normalize_title(t).replace(" ", "")):
        return True
    if "form 10-k" in t.lower() and "fiscal year" in t.lower():
        return True
    if t.lower().startswith("part ") and "item" in t.lower() and len(t) < 40:
        return True
    return False


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def normalize_title(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(left: str, right: str) -> float:
    a, b = normalize_title(left), normalize_title(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def box_text(box: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in box.get("textlines") or []:
        for span in line.get("spans") or []:
            parts.append(span.get("text") or "")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def load_font_headings(sections_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(sections_path.read_text(encoding="utf-8"))
    headings: list[dict[str, Any]] = []
    for part in payload.get("parts") or []:
        for section in part.get("items") or part.get("sections") or []:
            item = section.get("item")
            for h in section.get("subsection_headings") or []:
                headings.append(
                    {
                        "source": "font",
                        "item": item,
                        "title": h.get("title") or "",
                        "level": h.get("level"),
                        "page": h.get("page"),
                    }
                )
    return headings


def load_toc_item_headings(sections_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(sections_path.read_text(encoding="utf-8"))
    headings: list[dict[str, Any]] = []
    for h in payload.get("menu_guided_headings") or []:
        if h.get("chosen_offset") is None and h.get("chosen_pdf_page") is None:
            continue
        headings.append(
            {
                "source": "toc_item",
                "item": h.get("item"),
                "title": h.get("matched_title") or h.get("candidate_title") or h.get("toc_title") or "",
                "level": 1,
                "page": h.get("chosen_pdf_page"),
            }
        )
    return headings


def extract_layout_headings(pdf_path: Path, cache_path: Path | None = None) -> tuple[list[dict[str, Any]], float]:
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data["headings"], float(data.get("latency_sec") or 0)

    import pymupdf4llm

    t0 = time.perf_counter()
    raw = pymupdf4llm.to_json(str(pdf_path), header=False, footer=False)
    data = json.loads(raw)
    elapsed = time.perf_counter() - t0

    headings: list[dict[str, Any]] = []
    for page in data.get("pages") or []:
        page_no = int(page.get("page_number") or 0)
        for box in page.get("boxes") or []:
            boxclass = box.get("boxclass") or box.get("class") or ""
            if boxclass != "section-header":
                continue
            title = box_text(box)
            if not title or len(title) > 120 or is_layout_noise(title):
                continue
            headings.append(
                {
                    "source": "layout",
                    "item": None,
                    "title": title,
                    "level": None,
                    "page": page_no,
                }
            )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"latency_sec": elapsed, "headings": headings}, ensure_ascii=False),
            encoding="utf-8",
        )
    return headings, elapsed


def match_heading(target: dict[str, Any], pool: list[dict[str, Any]], used: set[int], threshold: float = 0.82) -> int | None:
    best_idx = None
    best_score = threshold
    t_page = target.get("page")
    for idx, cand in enumerate(pool):
        if idx in used:
            continue
        if t_page is not None and cand.get("page") is not None:
            if abs(int(t_page) - int(cand["page"])) > 1:
                continue
        score = title_similarity(target.get("title") or "", cand.get("title") or "")
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def recall_report(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str) -> dict[str, Any]:
    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    for ref in reference:
        idx = match_heading(ref, candidate, used)
        if idx is None:
            missed.append(ref)
        else:
            used.add(idx)
            matched.append({"ref": ref, "hit": candidate[idx]})
    extra = [c for i, c in enumerate(candidate) if i not in used]
    total = len(reference) or 1
    return {
        "label": label,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matched": len(matched),
        "missed": len(missed),
        "extra": len(extra),
        "recall_pct": round(100.0 * len(matched) / total, 1),
        "missed_titles": [m["title"] for m in missed[:30]],
        "extra_titles": [e["title"] for e in extra[:30]],
    }


def assign_items(sections_path: Path) -> dict[str, tuple[int, int]]:
    payload = json.loads(sections_path.read_text(encoding="utf-8"))
    bounds: dict[str, tuple[int, int]] = {}
    items: list[dict[str, Any]] = []
    for part in payload.get("parts") or []:
        items.extend(part.get("items") or part.get("sections") or [])
    for idx, section in enumerate(items):
        item = section.get("item")
        if not item:
            continue
        start = int(section.get("start_pdf_page") or 0)
        nxt = items[idx + 1] if idx + 1 < len(items) else None
        end = int(nxt.get("start_pdf_page") or start) - 1 if nxt else start + 200
        bounds[item] = (start, max(end, start))
    return bounds


def filter_by_item(headings: list[dict[str, Any]], item: str, bounds: dict[str, tuple[int, int]], *, by_page_only: bool = False) -> list[dict[str, Any]]:
    lo, hi = bounds.get(item, (0, 10**9))
    out = []
    for h in headings:
        if not by_page_only and h.get("item") and h.get("item") != item:
            continue
        page = h.get("page")
        if page is not None and lo <= int(page) <= hi:
            out.append(h)
        elif not by_page_only and h.get("item") == item:
            out.append(h)
    return out


def markdown_heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not m:
        return None
    return len(m.group(1))


def extract_markdown_headings(pdf_path: Path, pages: list[int] | None = None) -> list[dict[str, Any]]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=pages,
        page_chunks=True,
        header=False,
        footer=False,
    )
    headings: list[dict[str, Any]] = []
    for chunk in chunks:
        page_no = int((chunk.get("metadata") or {}).get("page_number") or chunk.get("page") or 0)
        for line in (chunk.get("text") or "").splitlines():
            lvl = markdown_heading_level(line)
            if lvl is None:
                continue
            title = re.sub(r"^#{1,6}\s+", "", line.strip())
            headings.append({"source": "layout_md", "item": None, "title": title, "level": lvl, "page": page_no})
    return headings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--item", default="Item 7", help="Focus item for detailed dump")
    parser.add_argument("--out", type=Path, default=BENCH_ROOT / "results" / "layout_vs_font.json")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if not args.sections.exists():
        raise SystemExit(f"sections.json not found: {args.sections}")

    font_headings = load_font_headings(args.sections)
    toc_items = load_toc_item_headings(args.sections)
    layout_headings, layout_sec = extract_layout_headings(args.pdf, BENCH_ROOT / "results" / "layout_headings_cache.json")
    bounds = assign_items(args.sections)

    layout_vs_font = recall_report(font_headings, layout_headings, "layout_section_header vs font")
    md_headings = extract_markdown_headings(args.pdf)
    md_vs_font = recall_report(font_headings, md_headings, "layout_markdown(#) vs font")

    # Hybrid: layout hits + font fallback for misses
    used_layout: set[int] = set()
    hybrid: list[dict[str, Any]] = []
    for ref in font_headings:
        idx = match_heading(ref, layout_headings, used_layout)
        if idx is not None:
            hit = dict(layout_headings[idx])
            hit["source"] = "layout"
            hit["level"] = ref.get("level")
            hit["item"] = ref.get("item")
            hybrid.append(hit)
        else:
            hybrid.append(dict(ref) | {"source": "font_fallback"})

    # Hybrid concept: layout hits + font fallback for misses (union covers 100% of font set)

    item = args.item
    item_font = [h for h in font_headings if h.get("item") == item]
    item_layout = filter_by_item(layout_headings, item, bounds, by_page_only=True)
    item_layout_vs_font = recall_report(item_font, item_layout, f"{item} layout vs font")

    payload = {
        "pdf": str(args.pdf),
        "layout_latency_sec": round(layout_sec, 2),
        "summary": {
            "font_subsection_count": len(font_headings),
            "layout_section_header_count": len(layout_headings),
            "layout_markdown_heading_count": len(md_headings),
            "toc_item_count": len(toc_items),
            "layout_recall_vs_font_pct": layout_vs_font["recall_pct"],
            "markdown_recall_vs_font_pct": md_vs_font["recall_pct"],
            "font_missed_by_layout": layout_vs_font["missed"],
            "layout_extra_not_in_font": layout_vs_font["extra"],
            "hybrid_would_use_font_fallback": layout_vs_font["missed"],
        },
        "full_doc": {
            "layout_vs_font": layout_vs_font,
            "markdown_vs_font": md_vs_font,
        },
        "focus_item": item,
        "focus_item_report": item_layout_vs_font,
        "focus_item_side_by_side": {
            "font": [{"page": h["page"], "level": h["level"], "title": h["title"]} for h in item_font],
            "layout": [{"page": h["page"], "title": h["title"]} for h in item_layout],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"\nWrote {args.out}")
    print(f"\n=== {item} side-by-side (first 20) ===")
    for row in payload["focus_item_side_by_side"]["font"][:20]:
        print(f"  FONT  p{row['page']:>2} L{row['level']}  {row['title'][:70]}")
    print("---")
    for row in payload["focus_item_side_by_side"]["layout"][:20]:
        print(f"  LAY   p{row['page']:>2}       {row['title'][:70]}")


if __name__ == "__main__":
    main()

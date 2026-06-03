from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Any

import fitz


CHUNKING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHUNKING_DIR.parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "pdfs" / "AAPL_FY2025_10-K.pdf"
ITEM_RE = re.compile(r"^\s*Item\s+(\d+[A-Z]?)\.\s*(.*)$", re.I)
PART_RE = re.compile(r"^Part\s+[IVX]+$", re.I)
TABLE_HEADING_RE = re.compile(r"^Table\s+\d+\b", re.I)
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
MONTH_DATE_LINE_RE = re.compile(
    r"^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"\d{1,2},\s+\d{4}$",
    re.I,
)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def normalize_heading(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(left: str, right: str) -> float:
    left_norm = normalize_heading(left)
    right_norm = normalize_heading(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def part_for_item(item: str) -> str:
    match = re.search(r"\d+", item)
    if not match:
        return "Unknown Part"
    base = int(match.group(0))
    if 1 <= base <= 4:
        return "Part I"
    if 5 <= base <= 9:
        return "Part II"
    if 10 <= base <= 14:
        return "Part III"
    if 15 <= base <= 16:
        return "Part IV"
    return "Unknown Part"


def is_toc_page(lines: list[str]) -> bool:
    item_count = sum(1 for line in lines if ITEM_RE.match(line))
    isolated_page_numbers = sum(1 for line in lines if re.fullmatch(r"\d+", line))
    has_toc_title = any(line.lower() == "table of contents" for line in lines[:8])
    short_item_count = sum(1 for line in lines if ITEM_RE.match(line) and len(line) < 30)
    return (has_toc_title and item_count >= 4 and isolated_page_numbers >= 4) or (
        item_count >= 6 and short_item_count >= 4 and isolated_page_numbers >= 4
    )


def page_lines(doc: fitz.Document, page_no: int) -> list[str]:
    return [clean_text(line) for line in doc[page_no - 1].get_text("text").splitlines() if clean_text(line)]


def page_styled_lines(doc: fitz.Document, page_no: int) -> list[dict[str, Any]]:
    styled: list[dict[str, Any]] = []
    for block in doc[page_no - 1].get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = clean_text("".join(span.get("text", "") for span in spans))
            if not text:
                continue
            fonts = [span.get("font", "") for span in spans]
            flags = [span.get("flags", 0) for span in spans]
            sizes = [span.get("size", 0.0) for span in spans]
            bbox = line.get("bbox", [0, 0, 0, 0])
            leading_heading = leading_emphasis_heading(spans)
            styled.append(
                {
                    "text": text,
                    "page": page_no,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "line_center": (bbox[0] + bbox[2]) / 2.0,
                    "font_names": sorted(set(fonts)),
                    "max_size": max(sizes),
                    "is_bold": any("bold" in font.lower() for font in fonts) or any(flag & 16 for flag in flags),
                    "is_italic": any(("oblique" in font.lower() or "italic" in font.lower()) for font in fonts) or any(
                        flag & 2 for flag in flags
                    ),
                    "leading_heading": leading_heading,
                }
            )
    return styled


def span_is_bold(span: dict[str, Any]) -> bool:
    return "bold" in str(span.get("font", "")).lower() or bool(span.get("flags", 0) & 16)


def span_is_italic(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    return "oblique" in font or "italic" in font or bool(span.get("flags", 0) & 2)


def leading_emphasis_heading(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Detect inline headings such as 'Agentforce Sales. Our offering ...'."""
    nonempty = [span for span in spans if clean_text(span.get("text", ""))]
    if len(nonempty) < 2:
        return None
    first = nonempty[0]
    first_text = clean_text(first.get("text", ""))
    if not first_text or not (span_is_bold(first) or span_is_italic(first)):
        return None
    # Only split true run-in headings: emphasized first span followed by normal body text.
    if span_is_bold(nonempty[1]) or span_is_italic(nonempty[1]):
        return None
    title = first_text.rstrip(".").strip()
    word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", title))
    if not re.search(r"[A-Za-z]", title) or len(title) > 80 or word_count > 10:
        return None
    if is_excluded_subsection_noise(title):
        return None
    return {
        "text": title,
        "x0": first.get("bbox", [None])[0],
        "x1": first.get("bbox", [None, None, None])[2],
        "y0": first.get("bbox", [None, None])[1],
        "y1": first.get("bbox", [None, None, None, None])[3],
        "max_size": first.get("size", 0.0),
        "is_bold": span_is_bold(first),
        "is_italic": span_is_italic(first),
        "font_names": [first.get("font", "")],
    }


def collect_pages(doc: fitz.Document, max_toc_scan_pages: int = 8) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_no in range(1, doc.page_count + 1):
        lines = page_lines(doc, page_no)
        # Visible TOC pages live before the first filing item. The loose
        # "many Item rows" heuristic is unsafe later in the filing, where
        # text can reference many Items in exhibit indexes or notes.
        page_rect = doc[page_no - 1].rect
        pages.append(
            {
                "page": page_no,
                "lines": lines,
                "styled_lines": page_styled_lines(doc, page_no),
                "page_width": round(page_rect.width, 2),
                "is_toc": page_no <= max_toc_scan_pages and is_toc_page(lines),
            }
        )
    return pages


def detect_toc_pages(pages: list[dict[str, Any]]) -> list[int]:
    return [page["page"] for page in pages if page["is_toc"]]


def parse_visible_toc(pages: list[dict[str, Any]], toc_pages: list[int]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    page_by_no = {page["page"]: page for page in pages}
    for page_no in toc_pages:
        lines = page_by_no[page_no]["lines"]
        idx = 0
        while idx < len(lines):
            match = ITEM_RE.match(lines[idx])
            if not match:
                idx += 1
                continue

            item = f"Item {match.group(1).upper()}"
            title_parts = [match.group(2).strip()] if match.group(2).strip() else []
            j = idx + 1
            printed_page: int | None = None
            while j < len(lines):
                line = lines[j]
                if ITEM_RE.match(line) or PART_RE.match(line):
                    break
                if re.fullmatch(r"\d+", line):
                    printed_page = int(line)
                    j += 1
                    break
                if line.lower() not in {"page"}:
                    title_parts.append(line)
                j += 1

            entries.append(
                {
                    "item": item,
                    "title": clean_text(" ".join(title_parts)),
                    "toc_pdf_page": page_no,
                    "printed_page": printed_page,
                }
            )
            idx = j
    return entries


def build_line_records(pages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    full_parts: list[str] = []
    records: list[dict[str, Any]] = []
    offset = 0
    for page in pages:
        page_no = page["page"]
        marker = f"\n\n[[PAGE {page_no}]]\n"
        full_parts.append(marker)
        offset += len(marker)
        for line in page["lines"]:
            line_start = offset
            full_parts.append(line + "\n")
            offset += len(line) + 1
            styled_match = next((styled for styled in page["styled_lines"] if styled["text"] == line), {})
            page_width = page.get("page_width") or 612.0
            x0 = styled_match.get("x0")
            x1 = styled_match.get("x1")
            line_center = None
            if x0 is not None and x1 is not None:
                line_center = (float(x0) + float(x1)) / 2.0
            records.append(
                {
                    "page": page_no,
                    "offset": line_start,
                    "line": line,
                    "is_toc": page["is_toc"],
                    "x0": x0,
                    "x1": x1,
                    "line_center": line_center,
                    "y0": styled_match.get("y0"),
                    "y1": styled_match.get("y1"),
                    "page_width": page_width,
                    "page_center": page_width / 2.0,
                    "max_size": styled_match.get("max_size"),
                    "is_bold": styled_match.get("is_bold", False),
                    "is_italic": styled_match.get("is_italic", False),
                    "font_names": styled_match.get("font_names", []),
                }
            )
            leading = styled_match.get("leading_heading") if styled_match else None
            if leading and leading.get("text") and leading.get("text") != line:
                heading_x0 = leading.get("x0")
                heading_x1 = leading.get("x1")
                heading_center = None
                if heading_x0 is not None and heading_x1 is not None:
                    heading_center = (float(heading_x0) + float(heading_x1)) / 2.0
                records.append(
                    {
                        "page": page_no,
                        "offset": line_start,
                        "line": leading["text"],
                        "is_toc": page["is_toc"],
                        "x0": heading_x0,
                        "x1": heading_x1,
                        "line_center": heading_center,
                        "y0": leading.get("y0"),
                        "y1": leading.get("y1"),
                        "page_width": page_width,
                        "page_center": page_width / 2.0,
                        "max_size": leading.get("max_size"),
                        "is_bold": leading.get("is_bold", False),
                        "is_italic": leading.get("is_italic", False),
                        "font_names": leading.get("font_names", []),
                        "synthetic_inline_heading": True,
                    }
                )
    enrich_vertical_gaps(records)
    return "".join(full_parts), records


def enrich_vertical_gaps(records: list[dict[str, Any]]) -> None:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_page.setdefault(record["page"], []).append(record)
    for page_records in by_page.values():
        page_records.sort(key=lambda rec: (rec.get("y0") or 0, rec.get("x0") or 0))
        for idx, record in enumerate(page_records):
            if idx > 0:
                prev = page_records[idx - 1]
                if record.get("y0") is not None and prev.get("y1") is not None:
                    record["gap_before"] = float(record["y0"]) - float(prev["y1"])
            if idx + 1 < len(page_records):
                nxt = page_records[idx + 1]
                if record.get("y1") is not None and nxt.get("y0") is not None:
                    record["gap_after"] = float(nxt["y0"]) - float(record["y1"])


def find_global_headings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, rec in enumerate(records):
        if rec["is_toc"]:
            continue
        match = ITEM_RE.match(rec["line"])
        if not match or len(rec["line"]) > 180:
            continue
        item = f"Item {match.group(1).upper()}"
        if item in seen:
            continue
        title = match.group(2).strip()
        if not title:
            for nxt in records[idx + 1 : idx + 5]:
                if nxt["page"] != rec["page"]:
                    break
                if not ITEM_RE.match(nxt["line"]):
                    title = nxt["line"]
                    break
        seen.add(item)
        headings.append({"item": item, "title": title, "page": rec["page"], "offset": rec["offset"], "line": rec["line"]})
    return headings


def menu_guided_headings(toc_entries: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the visible TOC as an expected section menu, not as page mapping."""
    candidate_by_item: dict[str, list[dict[str, Any]]] = {}
    for idx, rec in enumerate(records):
        if rec["is_toc"]:
            continue
        match = ITEM_RE.match(rec["line"])
        if not match or len(rec["line"]) > 220:
            continue
        item = f"Item {match.group(1).upper()}"
        line_title = match.group(2).strip()
        nearby_title_parts = [line_title] if line_title else []
        if not line_title:
            for nxt in records[idx + 1 : idx + 5]:
                if nxt["page"] != rec["page"]:
                    break
                if ITEM_RE.match(nxt["line"]):
                    break
                nearby_title_parts.append(nxt["line"])
        candidate_by_item.setdefault(item, []).append(
            {
                "item": item,
                "page": rec["page"],
                "offset": rec["offset"],
                "line": rec["line"],
                "candidate_title": clean_text(" ".join(nearby_title_parts)),
            }
        )

    guided: list[dict[str, Any]] = []
    last_offset = -1
    for entry in toc_entries:
        candidates = [
            candidate for candidate in candidate_by_item.get(entry["item"], [])
            if candidate["offset"] > last_offset
        ]
        scored = []
        for candidate in candidates:
            sim = title_similarity(entry["title"], candidate["candidate_title"])
            scored.append((sim, candidate))
        scored.sort(key=lambda pair: (pair[0], -pair[1]["offset"]), reverse=True)

        chosen = None
        score = 0.0
        status = "missing"
        if scored:
            score, chosen = scored[0]
            if score >= 0.45 or not entry["title"]:
                status = "menu_title_match"
            else:
                status = "item_only_low_title_similarity"
            last_offset = chosen["offset"]

        guided.append(
            {
                **entry,
                "chosen_pdf_page": chosen["page"] if chosen else None,
                "chosen_offset": chosen["offset"] if chosen else None,
                "chosen_line": chosen["line"] if chosen else None,
                "candidate_title": chosen["candidate_title"] if chosen else None,
                "title_similarity": round(score, 3),
                "search_status": status,
            }
        )
    return guided


def build_sections(full_text: str, matched_headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = [heading for heading in matched_headings if heading["chosen_offset"] is not None]
    ordered.sort(key=lambda heading: heading["chosen_offset"])
    sections: list[dict[str, Any]] = []
    for idx, heading in enumerate(ordered):
        start = heading["chosen_offset"]
        end = ordered[idx + 1]["chosen_offset"] if idx + 1 < len(ordered) else len(full_text)
        text = clean_text(full_text[start:end])
        sections.append(
            {
                "part": part_for_item(heading["item"]),
                "item": heading["item"],
                "toc_title": heading["title"],
                "matched_title": heading["candidate_title"],
                "title_similarity": heading["title_similarity"],
                "match_status": heading["search_status"],
                "start_offset": start,
                "end_offset": end,
                "start_pdf_page": heading["chosen_pdf_page"],
                "char_count": len(text),
                "text": text,
            }
        )
    return sections


def build_part_hierarchy(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    by_part: dict[str, dict[str, Any]] = {}
    for section in sections:
        part = by_part.setdefault(section["part"], {"part": section["part"], "items": []})
        part["items"].append(section)
    for part_name in ["Part I", "Part II", "Part III", "Part IV", "Unknown Part"]:
        if part_name in by_part:
            parts.append(by_part[part_name])
    return parts


def is_excluded_subsection_noise(line: str) -> bool:
    if ITEM_RE.match(line) or PART_RE.match(line) or TABLE_HEADING_RE.match(line):
        return True
    if re.match(r"^MATTERS,\s+AND\s+ISSUER", line, re.I):
        return True
    if PAGE_NUMBER_RE.match(line):
        return True
    if MONTH_DATE_LINE_RE.match(line):
        return True
    if re.fullmatch(r"[A-Z0-9 .,&/-]{8,90}", line) and MONTH_DATE_LINE_RE.search(line):
        return True
    return False


def centered_heading_score(record: dict[str, Any]) -> int:
    line = record["line"]
    if is_excluded_subsection_noise(line):
        return 0
    if not re.search(r"[A-Za-z]", line):
        return 0
    word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", line))
    if len(line) > 90 or word_count > 14 or word_count < 2:
        return 0
    if line.endswith((".", ";", ":")):
        return 0

    score = 0
    page_width = float(record.get("page_width") or 612.0)
    page_center = page_width / 2.0
    center_x = record.get("line_center")
    if center_x is None:
        x0 = record.get("x0")
        x1 = record.get("x1")
        if x0 is not None and x1 is not None:
            center_x = (float(x0) + float(x1)) / 2.0
        elif x0 is not None:
            center_x = float(x0)
    if center_x is not None and abs(float(center_x) - page_center) <= max(48.0, page_width * 0.09):
        score += 2

    alpha = [char for char in line if char.isalpha()]
    all_caps = bool(alpha) and sum(1 for char in alpha if char.isupper()) / len(alpha) >= 0.85
    if all_caps:
        score += 2

    line_height = float(record.get("max_size") or 10.0)
    gap_before = record.get("gap_before")
    gap_after = record.get("gap_after")
    gap_threshold = 1.5 * line_height if not all_caps else 1.25 * line_height
    if gap_before is not None and float(gap_before) > gap_threshold:
        score += 1
    if gap_after is not None and float(gap_after) > gap_threshold:
        score += 1
    return score


def is_subsection_candidate(record: dict[str, Any]) -> bool:
    line = record["line"]
    if is_excluded_subsection_noise(line):
        return False
    if not re.search(r"[A-Za-z]", line):
        return False
    word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", line))
    if len(line) > 100 or word_count > 16:
        return False
    if line.endswith((".", ";", ":")):
        return False
    if line.startswith("Apple Inc. |"):
        return False
    if (record.get("max_size") or 0) < 7.8:
        return False
    lower = line.lower()
    if lower.startswith(("the ", "this ", "these ", "those ", "we ", "our ", "it ", "there ", "because ", "while ", "although ")):
        return False
    if word_count >= 9 and re.search(
        r"\b(?:is|are|was|were|be|been|being|has|have|had|may|might|could|would|will|can|should|including)\b",
        lower,
    ):
        return False

    if centered_heading_score(record) >= 4:
        return True

    if record.get("x0") is None or record.get("x0") > 120:
        return False
    if record.get("is_bold"):
        return True
    if record.get("is_italic"):
        # Italic spans often mark lower-level headings, but Apple also uses
        # italic for intro paragraphs. Keep only heading-like short phrases.
        if len(line) > 65 or line.endswith(".") or re.match(r"^\d", line):
            return False
        if line.lower().startswith(("in ", "and ", "or ", "for ", "the ", "this ")):
            return False
        return True
    return False


def subsection_level(record: dict[str, Any]) -> int:
    line = record["line"]
    if re.match(r"^Note\s+\d+\s+[–-]", line):
        return 1
    if centered_heading_score(record) >= 4:
        return 2
    if record.get("is_bold"):
        return 2
    if record.get("is_italic"):
        return 3
    return 4


def build_subsection_path(stack: dict[int, str], level: int, title: str) -> list[str]:
    stack[level] = title
    for stale_level in [key for key in stack if key > level]:
        del stack[stale_level]
    return [stack[key] for key in sorted(stack) if key <= level]


def attach_subsections(sections: list[dict[str, Any]], records: list[dict[str, Any]], full_text: str) -> None:
    candidates = [record for record in records if not record["is_toc"] and is_subsection_candidate(record)]
    for section in sections:
        start = section["start_offset"]
        end = section["end_offset"]
        section_candidates = sorted(
            [
                {
                    "title": record["line"],
                    "level": subsection_level(record),
                    "offset": record["offset"],
                    "page": record["page"],
                    "style": {
                        "bold": record.get("is_bold", False),
                        "italic": record.get("is_italic", False),
                        "centered_heading_score": centered_heading_score(record),
                        "font_names": record.get("font_names", []),
                        "max_size": record.get("max_size"),
                        "x0": record.get("x0"),
                    },
                }
                for record in candidates
                if start < record["offset"] < end
            ],
            key=lambda candidate: candidate["offset"],
        )

        stack: dict[int, str] = {}
        subsection_chunks = []
        for idx, candidate in enumerate(section_candidates):
            chunk_start = candidate["offset"]
            chunk_end = section_candidates[idx + 1]["offset"] if idx + 1 < len(section_candidates) else end
            chunk_text = clean_text(full_text[chunk_start:chunk_end])
            path = build_subsection_path(stack, candidate["level"], candidate["title"])
            subsection_chunks.append(
                {
                    **candidate,
                    "path": path,
                    "end_offset": chunk_end,
                    "char_count": len(chunk_text),
                    "text": chunk_text,
                }
            )

        section["subsection_headings"] = section_candidates
        section["subsection_chunks"] = subsection_chunks
        if section_candidates:
            section["preamble_text"] = clean_text(full_text[start : section_candidates[0]["offset"]])
        else:
            # No font-detected subsections: keep full Item body for RAG chunking.
            section["preamble_text"] = clean_text(full_text[start:end])
        # Backward-compatible alias while this is still a probe.
        section["subsections"] = subsection_chunks


def summarize_subsection_counts(sections: list[dict[str, Any]]) -> dict[str, int]:
    return {
        section["item"]: len(section.get("subsection_chunks", []))
        for section in sections
        if section.get("subsection_chunks")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Menu-first 10-K section chunking probe.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=CHUNKING_DIR / "AAPL_FY2025_menu_guided_sections.json")
    parser.add_argument("--max-toc-scan-pages", type=int, default=8)
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    pages = collect_pages(doc, max_toc_scan_pages=args.max_toc_scan_pages)
    toc_pages = detect_toc_pages(pages)
    toc_entries = parse_visible_toc(pages, toc_pages)
    full_text, records = build_line_records(pages)
    global_headings = find_global_headings(records)
    menu_guided = menu_guided_headings(toc_entries, records)
    sections = build_sections(full_text, menu_guided)
    attach_subsections(sections, records, full_text)

    payload = {
        "source_file": args.pdf.name,
        "toc_pages_detected": toc_pages,
        "method": "visible_toc_menu_then_body_regex_offsets",
        "toc_entry_count": len(toc_entries),
        "global_heading_count": len(global_headings),
        "matched_heading_count": sum(1 for heading in menu_guided if heading["chosen_offset"] is not None),
        "subsection_counts": summarize_subsection_counts(sections),
        "toc_entries": toc_entries,
        "global_headings": global_headings,
        "menu_guided_headings": menu_guided,
        "parts": build_part_hierarchy(sections),
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    doc.close()

    print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                "toc_pages_detected": payload["toc_pages_detected"],
                "toc_entry_count": payload["toc_entry_count"],
                "global_heading_count": payload["global_heading_count"],
                "matched_heading_count": payload["matched_heading_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

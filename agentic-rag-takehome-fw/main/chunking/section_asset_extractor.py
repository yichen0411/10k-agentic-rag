from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz

from toc_guided_section_probe import build_line_records, collect_pages  # noqa: E402


CHUNKING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHUNKING_DIR.parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "pdfs" / "AAPL_FY2025_10-K.pdf"
DEFAULT_SECTIONS = CHUNKING_DIR / "AAPL_FY2025_menu_guided_sections.json"
DEFAULT_OUTPUT = CHUNKING_DIR / "AAPL_FY2025_section_assets.json"
NUMBER_RE = re.compile(r"^\(?\$?[-–—]?\d[\d,]*(?:\.\d+)?%?\)?$")
MONTH_DATE_CELL_RE = re.compile(
    r"^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"\d{1,2},\s+\d{4}$",
    re.I,
)
UNIT_HEADER_RE = re.compile(r"\(in\s+(millions|billions|thousands)\)", re.I)
PERIOD_HEADER_RE = re.compile(
    r"(?:three|six|nine|twelve)\s+months?\s+ending|months?\s+ending|year\s+ended|quarter\s+ending",
    re.I,
)

# Extra vertical context around detected table bboxes for UI/VLM screenshots.
TABLE_CROP_TOP_PAD = 96.0
TABLE_CROP_BOTTOM_PAD = 96.0


def bbox_height(bbox: list[float]) -> float:
    return float(bbox[3] - bbox[1])


def is_header_only_table_crop(
    table: dict[str, Any],
    crop_idx: int,
    bbox: list[float],
    page_height: float,
) -> bool:
    """Tight clip for cross-page header bands — not full data tables."""
    if table.get("header_only") or table.get("synthetic_header_band"):
        return True
    multi = table.get("bbox_by_page") or []
    return (
        len(multi) > 1
        and crop_idx == 0
        and bbox_height(bbox) < 40
        and float(bbox[1]) > page_height * 0.82
    )


def table_crop_padding(
    table: dict[str, Any],
    crop_idx: int,
    bbox: list[float],
    page_height: float,
) -> tuple[float, float]:
    """Return (top_pad, bottom_pad) in PDF points for table screenshot clips."""
    multi = table.get("bbox_by_page") or []
    if is_header_only_table_crop(table, crop_idx, bbox, page_height):
        return (4.0, 8.0 if crop_idx == 0 else 6.0)
    if len(multi) > 1:
        if crop_idx == 0:
            return (TABLE_CROP_TOP_PAD, 16.0)
        if crop_idx == len(multi) - 1:
            return (56.0, 32.0)
        return (24.0, 16.0)
    return (TABLE_CROP_TOP_PAD, TABLE_CROP_BOTTOM_PAD)


def table_crop_clip(page: fitz.Page, bbox: list[float], top_pad: float, bottom_pad: float) -> fitz.Rect:
    return fitz.Rect(
        page.rect.x0,
        max(page.rect.y0, float(bbox[1]) - top_pad),
        page.rect.x1,
        min(page.rect.y1, float(bbox[3]) + bottom_pad),
    ) & page.rect


class PageTableFindCache:
    """Cache PyMuPDF page.find_tables() results — the dominant cost in asset extraction."""

    def __init__(self) -> None:
        self._by_page: dict[int, Any] = {}

    def find_tables(self, page: fitz.Page) -> Any:
        page_no = page.number + 1
        if page_no not in self._by_page:
            self._by_page[page_no] = page.find_tables()
        return self._by_page[page_no]

    def pages_scanned(self) -> int:
        return len(self._by_page)


def collect_table_bbox_stubs(
    doc: fitz.Document,
    *,
    table_find_cache: PageTableFindCache | None = None,
) -> list[dict[str, Any]]:
    """Lightweight table bboxes for subsection-in-table filtering (no tab.extract)."""
    stubs: list[dict[str, Any]] = []
    for page_idx in range(doc.page_count):
        page_no = page_idx + 1
        page = doc[page_idx]
        found = table_find_cache.find_tables(page) if table_find_cache else page.find_tables()
        for tab in found.tables:
            stubs.append({"page_start": page_no, "bbox": [round(x, 2) for x in tab.bbox]})
    return stubs


class PageWordCache:
    """Cache page.get_text('words') for word-layer table rescans."""

    def __init__(self) -> None:
        self._by_page: dict[int, list[Any]] = {}

    def words(self, page: fitz.Page) -> list[Any]:
        page_no = page.number + 1
        if page_no not in self._by_page:
            self._by_page[page_no] = page.get_text("words")
        return self._by_page[page_no]


TABLE_LINE_ITEM_PREFIXES = (
    "revenue",
    "cost of revenue",
    "operating expenses",
    "operating income",
    "gross margin",
    "net income",
    "total revenue",
    "total cost",
    "diluted earnings",
    "basic",
)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def clean_cell(cell: Any) -> str | None:
    if cell is None:
        return None
    text = clean_text(str(cell))
    return text or None


def is_numeric_cell(cell: str | None) -> bool:
    if not cell:
        return False
    if cell in {"—", "-", "–"}:
        return True
    return bool(NUMBER_RE.match(cell.replace("$", "")))


def _dollar_only_cell_count(rows: list[list[str | None]]) -> int:
    return sum(1 for row in rows for cell in row if cell and cell.strip() == "$")


def _empty_ratio(rows: list[list[str | None]]) -> float:
    all_cells = [cell for row in rows for cell in row]
    return sum(1 for cell in all_cells if not cell) / max(len(all_cells), 1)


def _row_fill_counts(rows: list[list[str | None]]) -> list[int]:
    return [sum(1 for cell in row if cell) for row in rows]


def _has_financial_line_item_grid(rows: list[list[str | None]]) -> bool:
    return _dollar_only_cell_count(rows) >= 2


def _grid_utilization(rows: list[list[str | None]]) -> float:
    all_cells = [cell for row in rows for cell in row]
    if not all_cells:
        return 0.0
    return sum(1 for cell in all_cells if cell) / len(all_cells)


def _row_is_compact_label_row(row: list[str | None]) -> bool:
    """All filled cells are short — heading/label splits across columns, not data tables."""
    nonempty = [cell for cell in row if cell]
    if len(nonempty) < 2:
        return False
    if any(is_numeric_cell(cell) or (cell and cell.strip() == "$") for cell in nonempty):
        return False
    word_total = sum(len(cell.split()) for cell in nonempty)
    return word_total <= 10 and max(len(cell) for cell in nonempty) <= 48


def _row_is_short_long_pair(row: list[str | None]) -> bool:
    """One short token cell plus one longer body cell — list markers, not tabular rows."""
    nonempty = [cell for cell in row if cell]
    if len(nonempty) != 2:
        return False
    if any(is_numeric_cell(cell) or (cell and cell.strip() == "$") for cell in nonempty):
        return False
    word_counts = [len(cell.split()) for cell in nonempty]
    return min(word_counts) <= 2 and max(word_counts) >= 4


def _row_is_prose_wrap(row: list[str | None]) -> bool:
    nonempty = [cell for cell in row if cell]
    if not nonempty:
        return False
    row_span = max(len(row), 1)
    row_empty_ratio = 1 - len(nonempty) / row_span
    if len(nonempty) == 1 and len(nonempty[0].split()) >= 5:
        return True
    if row_empty_ratio >= 0.35 and any(len(cell.split()) >= 4 for cell in nonempty):
        return True
    return False


def _looks_like_underfilled_prose_grid(rows: list[list[str | None]], cells: list[str]) -> bool:
    """Core false-positive signal: sparse grid slots filled with sentence-like text, not tabular data."""
    if not cells:
        return False
    if _has_financial_line_item_grid(rows) or _has_tabular_data_rows(rows):
        return False

    if len(rows) == 1 and _row_is_compact_label_row(rows[0]):
        return True

    prose_ratio = _prose_content_ratio(cells)
    if prose_ratio < 0.5:
        return False

    if len(rows) == 1 and _row_is_short_long_pair(rows[0]):
        return True

    utilization = _grid_utilization(rows)
    short_long_rows = sum(1 for row in rows if _row_is_short_long_pair(row))
    wrap_rows = sum(1 for row in rows if _row_is_prose_wrap(row))
    fills = _row_fill_counts(rows)
    fill_spread = (max(fills) - min(fills)) if fills else 0

    if utilization <= 0.55:
        if short_long_rows >= 1:
            return True
        if wrap_rows >= max(1, len(rows) // 2):
            return True
        if len(cells) <= 3 and prose_ratio >= 0.8:
            return True

    if (
        len(rows) >= 2
        and fill_spread >= 1
        and _empty_ratio(rows) >= 0.35
        and prose_ratio >= 0.55
        and wrap_rows >= 1
    ):
        return True

    return False


def _is_tabular_data_row(row: list[str | None]) -> bool:
    if _row_is_prose_wrap(row):
        return False
    nonempty = [cell for cell in row if cell]
    if not nonempty:
        return False
    numeric = sum(1 for cell in nonempty if is_numeric_cell(cell))
    if numeric >= 2:
        return True
    if numeric >= 1 and any(len(cell) <= 32 for cell in nonempty if not is_numeric_cell(cell)):
        return True
    date_cells = [cell for cell in nonempty if MONTH_DATE_CELL_RE.match(cell)]
    if not date_cells:
        return False
    non_date = [cell for cell in nonempty if cell not in date_cells]
    if not non_date:
        return False
    if any(len(cell.split()) >= 4 for cell in non_date):
        return False
    if any(cell.lstrip()[:1] in ",;'\"" for cell in non_date):
        return False
    return max(len(cell) for cell in non_date) <= 32


def _has_tabular_data_rows(rows: list[list[str | None]]) -> bool:
    data_rows = sum(1 for row in rows if _is_tabular_data_row(row))
    if data_rows >= max(1, len(rows) // 3):
        return True
    numeric_by_col: dict[int, int] = {}
    for row in rows:
        for idx, cell in enumerate(row):
            if cell and is_numeric_cell(cell):
                numeric_by_col[idx] = numeric_by_col.get(idx, 0) + 1
    return any(count >= 2 for count in numeric_by_col.values())


def _cell_prose_weight(cell: str) -> float:
    if not cell or is_numeric_cell(cell) or cell.strip() == "$":
        return 0.0
    word_count = len(cell.split())
    if word_count >= 4:
        return 1.0
    if word_count >= 2:
        return 0.65
    return 0.25


def _prose_content_ratio(cells: list[str]) -> float:
    if not cells:
        return 0.0
    return sum(_cell_prose_weight(cell) for cell in cells) / len(cells)


def _joined_alpha_word_ratio(cells: list[str]) -> float:
    words = " ".join(cells).split()
    if len(words) < 6:
        return 0.0
    alpha = sum(1 for word in words if any(ch.isalpha() for ch in word))
    return alpha / len(words)


def _fill_pattern_is_irregular(rows: list[list[str | None]]) -> bool:
    counts = [count for count in _row_fill_counts(rows) if count > 0]
    if len(counts) < 2:
        return False
    if max(counts) - min(counts) >= 1:
        return True
    return _empty_ratio(rows) >= 0.3


def looks_like_prose_table(rows: list[list[str | None]]) -> bool:
    """Reject PyMuPDF false positives: multi-column prose paragraphs detected as tables."""
    cells = [cell for row in rows for cell in row if cell]
    if not cells:
        return False
    if _looks_like_underfilled_prose_grid(rows, cells):
        return True
    if len(cells) < 3:
        return False
    if _has_financial_line_item_grid(rows) or _has_tabular_data_rows(rows):
        return False

    col_count = max((len(row) for row in rows), default=0)
    row_count = len(rows)
    empty_ratio = _empty_ratio(rows)
    avg_len = sum(len(cell) for cell in cells) / len(cells)
    numeric_ratio = sum(1 for cell in cells if is_numeric_cell(cell)) / len(cells)
    prose_ratio = _prose_content_ratio(cells)
    alpha_ratio = _joined_alpha_word_ratio(cells)
    joined_len = len(" ".join(cells))
    long_cells = sum(1 for cell in cells if len(cell) >= 45)

    if (
        row_count <= 6
        and 2 <= col_count <= 8
        and empty_ratio >= 0.2
        and _fill_pattern_is_irregular(rows)
        and prose_ratio >= 0.55
        and alpha_ratio >= 0.75
        and joined_len >= 40
        and numeric_ratio < 0.3
    ):
        return True
    if numeric_ratio >= 0.25:
        return False
    if col_count <= 4 and avg_len >= 50 and long_cells >= 2:
        return True
    if empty_ratio >= 0.45 and avg_len >= 40 and long_cells >= 2 and numeric_ratio < 0.15:
        return True
    if numeric_ratio < 0.1 and empty_ratio >= 0.35 and joined_len >= 120 and col_count >= 5:
        return True
    return False


def table_complexity(rows: list[list[str | None]], bbox: list[float]) -> tuple[int, list[str]]:
    row_count = len(rows)
    col_count = max((len(row) for row in rows), default=0)
    cells = [cell for row in rows for cell in row]
    empty_count = sum(1 for cell in cells if not cell)
    empty_ratio = empty_count / max(len(cells), 1)
    numeric_count = sum(1 for cell in cells if is_numeric_cell(cell))
    numeric_ratio = numeric_count / max(len(cells) - empty_count, 1)

    score = 0
    reasons: list[str] = []
    if col_count > 8:
        score += 3
        reasons.append("many_columns")
    if col_count > 14:
        score += 2
        reasons.append("very_wide_table")
    if row_count > 15:
        score += 2
        reasons.append("many_rows")
    if empty_ratio > 0.25:
        score += 2
        reasons.append("many_empty_cells")
    if numeric_ratio > 0.45:
        score += 1
        reasons.append("numeric_dense")
    if bbox[2] - bbox[0] > 480:
        score += 1
        reasons.append("full_width")
    return score, reasons


def load_section_refs(sections_path: Path) -> list[dict[str, Any]]:
    data = json.loads(sections_path.read_text(encoding="utf-8"))
    refs: list[dict[str, Any]] = []
    for part in data.get("parts", []):
        for item in part.get("items", []):
            section_id = item["item"]
            section_ref = {
                "section_ref_id": f"{part['part']}::{section_id}",
                "part": part["part"],
                "item": section_id,
                "section_title": item["toc_title"],
                "start_offset": item["start_offset"],
                "end_offset": item["end_offset"],
                "start_pdf_page": item["start_pdf_page"],
                "char_count": item["char_count"],
                "table_refs": [],
                "image_refs": [],
                "subsections": [],
            }
            for idx, sub in enumerate(item.get("subsection_chunks", []), 1):
                section_ref["subsections"].append(
                    {
                        "subsection_ref_id": f"{part['part']}::{section_id}::sub_{idx:03d}",
                        "title": sub["title"],
                        "path": sub.get("path", [sub["title"]]),
                        "level": sub["level"],
                        "page": sub["page"],
                        "start_offset": sub["offset"],
                        "end_offset": sub["end_offset"],
                        "char_count": sub["char_count"],
                        "style": sub.get("style", {}),
                        "table_refs": [],
                        "image_refs": [],
                    }
                )
            refs.append(section_ref)
    refs.sort(key=lambda ref: ref["start_offset"])
    return refs


def annotate_page_ranges(section_refs: list[dict[str, Any]], doc_page_count: int) -> None:
    for idx, ref in enumerate(section_refs):
        next_page = section_refs[idx + 1]["start_pdf_page"] if idx + 1 < len(section_refs) else doc_page_count + 1
        ref["end_pdf_page"] = max(ref["start_pdf_page"], next_page - 1)


def find_section_for_page(
    section_refs: list[dict[str, Any]],
    page: int,
    y0: float | None = None,
    *,
    page_height: float = 792.0,
) -> dict[str, Any] | None:
    candidates = [ref for ref in section_refs if ref["start_pdf_page"] <= page <= ref["end_pdf_page"]]
    if not candidates:
        prior = [ref for ref in section_refs if ref["start_pdf_page"] <= page]
        return prior[-1] if prior else None
    if len(candidates) == 1 or y0 is None:
        return candidates[-1]
    # MSFT/AAPL pattern: Item 7A and Item 8 can share the same start_pdf_page (e.g. 47).
    # The next Item banner sits in the page footer; tables above it belong to the earlier Item.
    same_page = [ref for ref in candidates if ref["start_pdf_page"] == page]
    if len(same_page) >= 2:
        ordered = sorted(same_page, key=lambda ref: ref["start_offset"])
        footer_y = page_height * 0.91
        return ordered[-1] if y0 >= footer_y else ordered[0]
    return candidates[-1]


def find_section_for_asset(
    section_refs: list[dict[str, Any]],
    page: int,
    y0: float,
    *,
    page_height: float = 792.0,
) -> dict[str, Any] | None:
    """Section lookup that respects page-top continuations from the prior page."""
    if y0 < 150:
        # Section that began on or before the previous page and continues onto this one
        # (e.g. Item 8 financials on MSFT page 48 while Item 7A ends on page 47).
        spanning = [
            ref
            for ref in section_refs
            if ref["start_pdf_page"] <= page - 1 <= ref["end_pdf_page"] and ref["end_pdf_page"] >= page
        ]
        if spanning:
            return spanning[-1]
        prior_page = [ref for ref in section_refs if ref["end_pdf_page"] == page - 1]
        if prior_page:
            return prior_page[-1]
    return find_section_for_page(section_refs, page, y0, page_height=page_height)


def collect_heading_positions(doc: fitz.Document, section_refs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = {}
    needed = []
    for section in section_refs:
        for sub in section["subsections"]:
            needed.append((section, sub))

    for section, sub in needed:
        page_key = str(sub["page"])
        if page_key not in by_page:
            by_page[page_key] = []
        page = doc[sub["page"] - 1]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = clean_text("".join(span.get("text", "") for span in line.get("spans", [])))
                if text == sub["title"]:
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    by_page[page_key].append(
                        {
                            "section_ref_id": section["section_ref_id"],
                            "subsection_ref_id": sub["subsection_ref_id"],
                            "title": sub["title"],
                            "path": sub["path"],
                            "page": sub["page"],
                            "y0": bbox[1],
                            "offset": sub["start_offset"],
                        }
                    )
                    break
    for headings in by_page.values():
        headings.sort(key=lambda h: (h["page"], h["y0"], h["offset"]))
    return by_page


def section_heading_stream(
    section: dict[str, Any],
    heading_positions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for page_headings in heading_positions.values():
        headings.extend(
            heading for heading in page_headings if heading["section_ref_id"] == section["section_ref_id"]
        )
    headings.sort(key=lambda heading: (heading["page"], heading["y0"], heading["offset"]))
    return headings


def find_subsection_for_asset(
    section: dict[str, Any] | None,
    heading_positions: dict[str, list[dict[str, Any]]],
    page: int,
    y0: float,
) -> dict[str, Any] | None:
    if not section:
        return None

    # Use reading order across pages. Tables at the top of a page often sit
    # under the previous page's active subsection, before the first heading on
    # the current page (e.g. MSFT Item 7 summary table on page 37).
    active: dict[str, Any] | None = None
    for heading in section_heading_stream(section, heading_positions):
        if (heading["page"] < page) or (heading["page"] == page and heading["y0"] <= y0 + 2):
            active = heading
        else:
            break
    if active:
        return {
            "section_ref_id": active["section_ref_id"],
            "subsection_ref_id": active["subsection_ref_id"],
            "title": active["title"],
            "path": active["path"],
            "page": active["page"],
            "offset": active["offset"],
        }

    prior = [sub for sub in section["subsections"] if sub["page"] <= page]
    if prior:
        sub = prior[-1]
        return {
            "section_ref_id": section["section_ref_id"],
            "subsection_ref_id": sub["subsection_ref_id"],
            "title": sub["title"],
            "path": sub["path"],
            "page": sub["page"],
            "offset": sub["start_offset"],
        }
    return None


def row_text(row: list[str | None]) -> str:
    return " | ".join(cell or "" for cell in row)


def looks_like_header_row(row: list[str | None]) -> bool:
    nonempty = [cell for cell in row if cell]
    if not nonempty:
        return False
    if is_data_row(row):
        return False
    numeric = sum(1 for cell in nonempty if is_numeric_cell(cell))
    text = " ".join(nonempty).lower()
    header_words = (
        "year",
        "fair value",
        "assets",
        "liabilities",
        "revenue",
        "2025",
        "2024",
        "2023",
        "september",
        "declaration",
        "record date",
        "payment date",
        "dividend",
        "per share",
        "amount",
        "periods",
        "total number",
    )
    return numeric <= max(1, len(nonempty) // 3) and any(word in text for word in header_words)


def is_data_row(row: list[str | None]) -> bool:
    nonempty = [cell for cell in row if cell]
    if not nonempty:
        return False
    if any(MONTH_DATE_CELL_RE.match(cell) for cell in nonempty):
        return True
    numeric = sum(1 for cell in nonempty if is_numeric_cell(cell))
    return numeric >= max(2, len(nonempty) // 2)


def is_header_only_table(rows: list[list[str | None]]) -> bool:
    if not rows:
        return False
    data_rows = [row for row in rows if is_data_row(row)]
    if data_rows:
        return False
    if len(rows) == 1 and looks_like_header_row(rows[0]):
        return True
    return all(looks_like_header_row(row) for row in rows)


def table_touches_page_bottom(bbox: list[float], page_height: float, margin: float = 95.0) -> bool:
    return page_height - bbox[3] < margin


def split_line_into_cells(line: str, expected_cols: int | None = None) -> list[str | None]:
    parts = [clean_text(part) for part in re.split(r"\s{2,}|\t+", line) if clean_text(part)]
    if len(parts) < 2:
        parts = [clean_text(part) for part in line.split("  ") if clean_text(part)]
    if len(parts) < 2:
        return []
    if expected_cols and len(parts) > expected_cols:
        parts = parts[:expected_cols]
    return [part or None for part in parts]


def words_to_row_cells(parts: list[tuple[float, str]]) -> list[str]:
    cells: list[str] = []
    label_buf: list[str] = []
    for x0, text in sorted(parts, key=lambda item: item[0]):
        token = clean_text(text)
        if not token:
            continue
        if x0 < 120:
            label_buf.append(token)
        else:
            if label_buf:
                cells.append(" ".join(label_buf))
                label_buf = []
            cells.append(token)
    if label_buf:
        cells.append(" ".join(label_buf))
    return cells


def is_table_line_item_row(cells: list[str]) -> bool:
    if not cells:
        return False
    if is_data_row(cells) or looks_like_header_row(cells):
        return True
    label = (cells[0] or "").lower().strip()
    if label in {"total", "thereafter"}:
        return True
    return any(label == prefix or label.startswith(f"{prefix} ") for prefix in TABLE_LINE_ITEM_PREFIXES)


def is_narrow_schedule_table(table: dict[str, Any]) -> bool:
    rows = table.get("raw_rows") or []
    if not rows:
        return False
    col_count = table.get("col_count") or max((len(row) for row in rows), default=0)
    if col_count > 4:
        return False
    schedule_like = 0
    for row in rows:
        if not row:
            continue
        label = (row[0] or "").strip()
        lower = label.lower()
        if MONTH_DATE_CELL_RE.match(label) or lower in {"thereafter", "total"}:
            schedule_like += 1
        elif any(is_numeric_cell(cell) for cell in row[1:]):
            schedule_like += 1
    return schedule_like >= max(1, len(rows) // 2)


def collect_table_word_rows(
    page: fitz.Page,
    y0: float,
    y1: float,
    *,
    word_cache: PageWordCache | None = None,
) -> list[tuple[float, list[str], list[float]]]:
    rows_by_y: dict[float, list[tuple[float, str]]] = {}
    words = word_cache.words(page) if word_cache else page.get_text("words")
    for word in words:
        x0, top, x1, bottom, text, *_ = word
        if float(top) < y0 or float(top) > y1:
            continue
        y_key = round(float(top), 1)
        rows_by_y.setdefault(y_key, []).append((float(x0), text))

    collected: list[tuple[float, list[str], list[float]]] = []
    for y_key in sorted(rows_by_y):
        parts = rows_by_y[y_key]
        cells = words_to_row_cells(parts)
        combined = " ".join(cell for cell in cells if cell)
        if is_prose_not_table_header(combined):
            if collected:
                break
            continue
        if not is_table_line_item_row(cells):
            if collected:
                break
            continue
        xs = [p[0] for p in parts]
        bbox = [min(xs), y_key, max(p[0] + len(p[1]) * 4 for p in parts), y_key + 12.0]
        collected.append((y_key, cells, bbox))
    return collected


def rescan_table_rows_from_words(
    doc: fitz.Document,
    table: dict[str, Any],
    *,
    word_cache: PageWordCache | None = None,
) -> dict[str, Any]:
    """Fill in subtotal / line-item rows that find_tables() missed on the text layer."""
    if table.get("header_only") or table.get("pending_merge") or table.get("synthetic_header_band"):
        return table
    if table.get("asset_type") in {"table_header_band", "table_continuation"}:
        return table

    page = doc[int(table["page_start"]) - 1]
    y0 = float(table["bbox"][1]) - 2
    scan_below = 56 if is_narrow_schedule_table(table) else 36
    y1 = float(table["bbox"][3]) + scan_below
    word_rows = collect_table_word_rows(page, y0, y1, word_cache=word_cache)

    if len(word_rows) <= len(table.get("raw_rows") or []):
        return table

    rows = [cells for _, cells, _ in word_rows]
    bbox = list(table["bbox"])
    same_page_bottom = max(
        (row_bbox[3] for _, _, row_bbox in word_rows),
        default=bbox[3],
    )
    bbox[3] = round(max(bbox[3], same_page_bottom), 2)

    score, reasons = table_complexity(rows, bbox)
    table.update(
        {
            "bbox": bbox,
            "row_count": len(rows),
            "col_count": max((len(row) for row in rows), default=table.get("col_count") or 0),
            "raw_rows": rows,
            "raw_text": "\n".join(row_text(row) for row in rows),
            "first_row": rows[0] if rows else [],
            "last_row": rows[-1] if rows else [],
            "complexity_score": score,
            "complexity_reasons": reasons + ["rescanned_word_layer_rows"],
            "complexity": "complex" if score >= 5 else "simple",
        }
    )
    return table


def supplement_rescanned_word_rows(doc: fitz.Document, tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [rescan_table_rows_from_words(doc, table) for table in tables]


def build_continuation_from_page_top(
    doc: fitz.Document,
    page_no: int,
    prev_table: dict[str, Any],
    section_refs: list[dict[str, Any]],
    heading_positions: dict[str, list[dict[str, Any]]],
    max_top_gap: float = 160.0,
) -> dict[str, Any] | None:
    """Fallback when find_tables() misses data rows at the top of the next page."""
    page = doc[page_no - 1]
    word_rows = collect_table_word_rows(page, 0, max_top_gap)
    if not word_rows or not any(is_data_row(cells) for _, cells, _ in word_rows):
        return None

    rows = [cells for _, cells, _ in word_rows[:1]]
    bbox_parts = [row_bbox for _, _, row_bbox in word_rows[:1]]
    bbox = [
        min(part[0] for part in bbox_parts),
        min(part[1] for part in bbox_parts),
        max(part[2] for part in bbox_parts),
        max(part[3] for part in bbox_parts),
    ]
    section = find_section_for_asset(section_refs, page_no, bbox[1])
    if section is None:
        return None
    subsection_ref = prev_table.get("subsection_ref")
    if not subsection_ref:
        subsection = find_subsection_for_asset(section, heading_positions, page_no, bbox[1])
        subsection_ref = subsection_metadata(subsection)

    score, reasons = table_complexity(rows, bbox)
    return {
        "table_id": f"table_continuation_{page_no:03d}_{len(rows):02d}",
        "asset_type": "table_continuation",
        "page_start": page_no,
        "page_end": page_no,
        "bbox": [round(x, 2) for x in bbox],
        "row_count": len(rows),
        "col_count": max((len(row) for row in rows), default=0),
        "raw_rows": rows,
        "raw_text": "\n".join(row_text(row) for row in rows),
        "first_row": rows[0],
        "last_row": rows[-1],
        "complexity_score": score,
        "complexity_reasons": reasons + ["synthetic_top_of_page_continuation"],
        "complexity": "complex" if score >= 5 else "simple",
        "section_ref": section_metadata(section),
        "subsection_ref": subsection_ref,
        "merge_group_id": None,
        "continued_from": prev_table["table_id"],
        "continued_to": None,
        "pending_merge": False,
        "header_only": False,
        "synthetic_continuation": True,
        "merge_reasons": ["synthetic_top_of_page_continuation"],
    }


def is_prose_not_table_header(text: str) -> bool:
    lower = text.lower()
    word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text))
    if len(text) > 72 or word_count > 12:
        return True
    if text.endswith(":") and word_count > 5:
        return True
    if lower.startswith(("we ", "our ", "following ", "highlights ", "the ", "this ", "refer to ")):
        return True
    if " compared with " in lower or " included:" in lower or " increased " in lower:
        return True
    return False


def is_table_header_band(rows: list[list[str | None]]) -> bool:
    if not rows or any(is_data_row(row) for row in rows):
        return False
    combined = " ".join(cell for row in rows for cell in row if cell)
    if is_prose_not_table_header(combined):
        return False
    combined_lower = combined.lower()
    if UNIT_HEADER_RE.search(combined) and PERIOD_HEADER_RE.search(combined):
        return True
    if PERIOD_HEADER_RE.search(combined) and len(rows) <= 3:
        return True
    header_keywords = (
        "declaration",
        "record date",
        "payment date",
        "per share",
        "amount",
        "period",
        "year ended",
        "fair value",
        "total number",
    )
    keyword_hits = sum(1 for word in header_keywords if word in combined_lower)
    nonempty_cells = sum(1 for row in rows for cell in row if cell)
    if keyword_hits >= 2:
        return True
    if nonempty_cells >= 4 and is_header_only_table(rows):
        return True
    return False


def cluster_line_fragments(fragments: list[dict[str, Any]], y_tolerance: float = 4.0) -> list[list[dict[str, Any]]]:
    if not fragments:
        return []
    ordered = sorted(fragments, key=lambda frag: (frag["y0"], frag["x0"]))
    clusters: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    for frag in ordered[1:]:
        if abs(frag["y0"] - current[-1]["y0"]) <= y_tolerance:
            current.append(frag)
        else:
            clusters.append(current)
            current = [frag]
    clusters.append(current)
    return clusters


def fragments_to_row(cluster: list[dict[str, Any]]) -> list[str | None]:
    cells = [clean_text(frag["text"]) for frag in sorted(cluster, key=lambda frag: frag["x0"]) if clean_text(frag["text"])]
    return [cell or None for cell in cells]


def rows_from_fragment_clusters(clusters: list[list[dict[str, Any]]]) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            cells = split_line_into_cells(cluster[0]["text"])
            row = cells if cells else fragments_to_row(cluster)
        else:
            row = fragments_to_row(cluster)
        if row:
            rows.append(row)
    return rows


def extract_trailing_header_band_rows(rows: list[list[str | None]]) -> list[list[str | None]]:
    best: list[list[str | None]] = []
    best_score = -1
    for start in range(len(rows)):
        candidate = rows[start:]
        if not is_table_header_band(candidate):
            continue
        combined = " ".join(cell for row in candidate for cell in row if cell)
        score = len(candidate)
        if UNIT_HEADER_RE.search(combined):
            score += 10
        if PERIOD_HEADER_RE.search(combined):
            score += 5
        if score > best_score:
            best_score = score
            best = candidate
    return best


def bbox_for_header_rows(
    merged_cluster: list[dict[str, Any]],
    header_rows: list[list[str | None]],
) -> list[float]:
    header_cells = {cell for row in header_rows for cell in row if cell}
    matched = [
        frag
        for frag in merged_cluster
        if any(cell in frag["text"] or frag["text"] in cell for cell in header_cells)
    ]
    source = matched or merged_cluster
    return [
        min(frag["bbox"][0] for frag in source),
        min(frag["bbox"][1] for frag in source),
        max(frag["bbox"][2] for frag in source),
        max(frag["bbox"][3] for frag in source),
    ]


def merge_adjacent_bottom_clusters(
    clusters: list[list[dict[str, Any]]],
    page_height: float,
    max_gap: float = 18.0,
) -> list[list[dict[str, Any]]]:
    if not clusters:
        return []
    merged: list[list[dict[str, Any]]] = []
    for cluster in clusters:
        cluster_bbox = [
            min(frag["bbox"][0] for frag in cluster),
            min(frag["bbox"][1] for frag in cluster),
            max(frag["bbox"][2] for frag in cluster),
            max(frag["bbox"][3] for frag in cluster),
        ]
        if merged:
            prev = merged[-1]
            prev_bbox = [
                min(frag["bbox"][0] for frag in prev),
                min(frag["bbox"][1] for frag in prev),
                max(frag["bbox"][2] for frag in prev),
                max(frag["bbox"][3] for frag in prev),
            ]
            gap = cluster_bbox[1] - prev_bbox[3]
            if table_touches_page_bottom(prev_bbox, page_height) and gap <= max_gap:
                merged[-1] = prev + cluster
                continue
        merged.append(list(cluster))
    return merged


def synthesize_page_bottom_header_tables(
    doc: fitz.Document,
    tables: list[dict[str, Any]],
    section_refs: list[dict[str, Any]],
    heading_positions: dict[str, list[dict[str, Any]]],
    bottom_margin: float = 130.0,
) -> list[dict[str, Any]]:
    """Create pending_merge tables when find_tables() misses header-only bands at page bottom."""
    existing_keys = {
        (table["page_start"], round(table["bbox"][1], 1), round(table["bbox"][3], 1))
        for table in tables
    }
    extras: list[dict[str, Any]] = []

    for page_idx in range(doc.page_count):
        page_no = page_idx + 1
        page = doc[page_idx]
        page_height = page.rect.height
        fragments: list[dict[str, Any]] = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                bbox = line.get("bbox", [0, 0, 0, 0])
                if float(bbox[1]) < page_height - bottom_margin:
                    continue
                text = clean_text("".join(span.get("text", "") for span in line.get("spans", [])))
                if not text or len(text) < 3:
                    continue
                fragments.append(
                    {
                        "x0": float(bbox[0]),
                        "y0": float(bbox[1]),
                        "x1": float(bbox[2]),
                        "y1": float(bbox[3]),
                        "text": text,
                        "bbox": [float(x) for x in bbox],
                    }
                )

        if not fragments:
            continue

        line_clusters = cluster_line_fragments(fragments)
        for merged_cluster in merge_adjacent_bottom_clusters(line_clusters, page_height):
            flat_rows = rows_from_fragment_clusters(cluster_line_fragments(merged_cluster))
            header_rows = extract_trailing_header_band_rows(flat_rows)
            if not header_rows:
                continue

            bbox = bbox_for_header_rows(merged_cluster, header_rows)
            if not table_touches_page_bottom(bbox, page_height):
                continue

            key = (page_no, round(bbox[1], 1), round(bbox[3], 1))
            if key in existing_keys:
                continue

            section = find_section_for_asset(section_refs, page_no, bbox[1])
            if section is None:
                continue
            subsection = find_subsection_for_asset(section, heading_positions, page_no, bbox[1])
            score, reasons = table_complexity(header_rows, bbox)
            synthetic_id = f"table_header_{page_no:03d}_{len(extras) + 1:02d}"
            extras.append(
                {
                    "table_id": synthetic_id,
                    "asset_type": "table_header_band",
                    "page_start": page_no,
                    "page_end": page_no,
                    "bbox": [round(x, 2) for x in bbox],
                    "row_count": len(header_rows),
                    "col_count": max((len(row) for row in header_rows), default=0),
                    "raw_rows": header_rows,
                    "raw_text": "\n".join(row_text(row) for row in header_rows),
                    "first_row": header_rows[0],
                    "last_row": header_rows[-1],
                    "complexity_score": score,
                    "complexity_reasons": reasons + ["synthetic_page_bottom_header_band"],
                    "complexity": "complex" if score >= 5 else "simple",
                    "section_ref": section_metadata(section),
                    "subsection_ref": subsection_metadata(subsection),
                    "merge_group_id": None,
                    "continued_from": None,
                    "continued_to": None,
                    "header_only": True,
                    "pending_merge": True,
                    "synthetic_header_band": True,
                }
            )
            existing_keys.add(key)
    return tables + extras


def supplement_header_only_continuations(
    doc: fitz.Document,
    tables: list[dict[str, Any]],
    section_refs: list[dict[str, Any]],
    heading_positions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    extras: list[dict[str, Any]] = []
    for prev in tables:
        if not prev.get("pending_merge"):
            continue
        next_page = prev["page_end"] + 1
        if next_page > doc.page_count:
            continue
        has_top_candidate = any(
            table["page_start"] == next_page and table["bbox"][1] < 150 for table in tables
        )
        if has_top_candidate:
            continue
        synthetic = build_continuation_from_page_top(doc, next_page, prev, section_refs, heading_positions)
        if synthetic:
            prev["continued_to"] = synthetic["table_id"]
            extras.append(synthetic)
    return tables + extras


def extract_tables(
    doc: fitz.Document,
    section_refs: list[dict[str, Any]],
    *,
    heading_positions: dict[str, list[dict[str, Any]]] | None = None,
    table_find_cache: PageTableFindCache | None = None,
) -> list[dict[str, Any]]:
    if heading_positions is None:
        heading_positions = collect_heading_positions(doc, section_refs)
    tables: list[dict[str, Any]] = []
    for page_idx in range(doc.page_count):
        page_no = page_idx + 1
        page = doc[page_idx]
        found = table_find_cache.find_tables(page) if table_find_cache else page.find_tables()
        for tab in found.tables:
            rows = [[clean_cell(cell) for cell in row] for row in tab.extract()]
            if not any(any(cell for cell in row) for row in rows):
                continue
            if looks_like_prose_table(rows):
                continue
            bbox = [round(x, 2) for x in tab.bbox]
            section = find_section_for_asset(section_refs, page_no, bbox[1])
            if section is None:
                continue
            subsection = find_subsection_for_asset(section, heading_positions, page_no, bbox[1])
            score, reasons = table_complexity(rows, bbox)
            header_only = is_header_only_table(rows)
            page_height = page.rect.height
            pending_merge = header_only and table_touches_page_bottom(bbox, page_height)
            tables.append(
                {
                    "table_id": f"table_{len(tables) + 1:03d}",
                    "asset_type": "table",
                    "page_start": page_no,
                    "page_end": page_no,
                    "bbox": bbox,
                    "row_count": len(rows),
                    "col_count": max((len(row) for row in rows), default=0),
                    "raw_rows": rows,
                    "raw_text": "\n".join(row_text(row) for row in rows),
                    "first_row": rows[0] if rows else [],
                    "last_row": rows[-1] if rows else [],
                    "complexity_score": score,
                    "complexity_reasons": reasons,
                    "complexity": "complex" if score >= 5 else "simple",
                    "section_ref": section_metadata(section),
                    "subsection_ref": subsection_metadata(subsection),
                    "merge_group_id": None,
                    "continued_from": None,
                    "continued_to": None,
                    "header_only": header_only,
                    "pending_merge": pending_merge,
                }
            )
    return tables


def section_metadata(section: dict[str, Any] | None) -> dict[str, Any] | None:
    if not section:
        return None
    return {
        "section_ref_id": section["section_ref_id"],
        "part": section["part"],
        "item": section["item"],
        "section_title": section["section_title"],
    }


def subsection_metadata(subsection: dict[str, Any] | None) -> dict[str, Any] | None:
    if not subsection:
        return None
    return {
        "subsection_ref_id": subsection["subsection_ref_id"],
        "title": subsection["title"],
        "path": subsection["path"],
    }


def similar_width(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_width = left["bbox"][2] - left["bbox"][0]
    right_width = right["bbox"][2] - right["bbox"][0]
    return abs(left_width - right_width) <= 80


def is_full_width_table(table: dict[str, Any]) -> bool:
    bbox = table.get("bbox") or []
    if len(bbox) < 4:
        return False
    return (bbox[2] - bbox[0]) >= 480


def same_page_fragment_candidate(prev: dict[str, Any], nxt: dict[str, Any]) -> bool:
    if nxt["page_start"] != prev["page_start"]:
        return False
    if prev.get("section_ref", {}).get("section_ref_id") != nxt.get("section_ref", {}).get("section_ref_id"):
        return False

    prev_sub = (prev.get("subsection_ref") or {}).get("subsection_ref_id")
    next_sub = (nxt.get("subsection_ref") or {}).get("subsection_ref_id")
    if prev_sub != next_sub:
        return False

    left_delta = abs(prev["bbox"][0] - nxt["bbox"][0])
    if left_delta > 25 or not similar_width(prev, nxt):
        return False

    vertical_gap = nxt["bbox"][1] - prev["bbox"][3]
    # Word-layer rescans can expand bboxes so adjacent statement slices overlap.
    # Full-width stacks in the same subsection are one financial statement table.
    if is_full_width_table(prev) and is_full_width_table(nxt) and -35 <= vertical_gap <= 50:
        return True

    if vertical_gap < -2 or vertical_gap > 28:
        return False

    col_delta = abs(prev["col_count"] - nxt["col_count"])
    if col_delta > 2:
        return False

    # PyMuPDF often slices wide financial tables into one-row tables. Avoid
    # merging ordinary prose/list fragments by requiring table-like density.
    if is_narrow_schedule_table(prev) and is_narrow_schedule_table(nxt):
        return True
    return max(prev["col_count"], nxt["col_count"]) >= 4


def merge_same_page_fragments(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tables = sorted(tables, key=lambda table: (table["page_start"], table["bbox"][1], table["bbox"][0]))
    merged: list[dict[str, Any]] = []
    idx = 0
    while idx < len(tables):
        group = [tables[idx]]
        idx += 1
        while idx < len(tables) and same_page_fragment_candidate(group[-1], tables[idx]):
            group.append(tables[idx])
            idx += 1

        if len(group) == 1:
            merged.append(group[0])
            continue

        rows: list[list[str | None]] = []
        for table in group:
            rows.extend(table.get("raw_rows", []))
        bbox = [
            min(table["bbox"][0] for table in group),
            min(table["bbox"][1] for table in group),
            max(table["bbox"][2] for table in group),
            max(table["bbox"][3] for table in group),
        ]
        score, reasons = table_complexity(rows, bbox)
        source_ids = [table["table_id"] for table in group]
        merged.append(
            {
                **group[0],
                "table_id": f"{source_ids[0]}_merged",
                "asset_type": "table_group",
                "bbox": [round(x, 2) for x in bbox],
                "row_count": len(rows),
                "col_count": max((len(row) for row in rows), default=0),
                "raw_rows": rows,
                "raw_text": "\n".join(row_text(row) for row in rows),
                "first_row": rows[0] if rows else [],
                "last_row": rows[-1] if rows else [],
                "complexity_score": score,
                "complexity_reasons": reasons + ["merged_same_page_fragments"],
                "complexity": "complex" if score >= 5 else "simple",
                "source_table_ids": source_ids,
                "continued_from": None,
                "continued_to": None,
            }
        )
    return merged


def _table_skip_absorption(table: dict[str, Any]) -> bool:
    if table.get("header_only") or table.get("pending_merge") or table.get("synthetic_header_band"):
        return True
    return table.get("asset_type") in {"table_header_band", "table_continuation"}


def collapse_absorbed_same_page_fragments(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop fragment tables fully contained in a larger rescanned table on the same page."""
    absorbed_ids: set[str] = set()
    candidates = [table for table in tables if not _table_skip_absorption(table)]
    by_page_sub: dict[tuple[int, str | None], list[dict[str, Any]]] = {}
    for table in candidates:
        subsection_id = (table.get("subsection_ref") or {}).get("subsection_ref_id")
        by_page_sub.setdefault((table["page_start"], subsection_id), []).append(table)

    for group in by_page_sub.values():
        ordered = sorted(group, key=lambda table: (table["bbox"][3] - table["bbox"][1]), reverse=True)
        for idx, big in enumerate(ordered):
            big_y0, big_y1 = big["bbox"][1], big["bbox"][3]
            for small in ordered[idx + 1 :]:
                if small["table_id"] in absorbed_ids:
                    continue
                small_y0, small_y1 = small["bbox"][1], small["bbox"][3]
                if small_y0 >= big_y0 - 2 and small_y1 <= big_y1 + 2:
                    absorbed_ids.add(small["table_id"])

    if not absorbed_ids:
        return tables
    return [table for table in tables if table["table_id"] not in absorbed_ids]


def should_merge_tables(prev: dict[str, Any], nxt: dict[str, Any], doc: fitz.Document) -> tuple[bool, int, list[str]]:
    if nxt["page_start"] != prev["page_end"] + 1:
        return False, 0, []
    header_only_prev = bool(prev.get("pending_merge") or prev.get("header_only"))
    prev_section = prev.get("section_ref", {}).get("section_ref_id")
    next_section = nxt.get("section_ref", {}).get("section_ref_id")
    if prev_section != next_section and not header_only_prev:
        return False, 0, []

    prev_page_height = doc[prev["page_end"] - 1].rect.height
    prev_bottom_gap = prev_page_height - prev["bbox"][3]
    next_top_gap = nxt["bbox"][1]
    score = 0
    reasons: list[str] = []
    merge_threshold = 4 if header_only_prev else 5

    prev_touches_bottom = prev_bottom_gap < 95
    next_starts_top = next_top_gap < 130
    if not (prev_touches_bottom and next_starts_top):
        return False, 0, []

    if prev_touches_bottom:
        score += 2
        reasons.append("previous_table_touches_page_bottom")
    if next_starts_top:
        score += 2
        reasons.append("next_table_starts_near_page_top")
    if header_only_prev:
        score += 2
        reasons.append("header_only_pending_merge")
    if header_only_prev and prev_section != next_section:
        score += 1
        reasons.append("cross_section_page_top_continuation")
    if prev["col_count"] == nxt["col_count"]:
        score += 2
        reasons.append("same_column_count")
    elif header_only_prev:
        score += 1
        reasons.append("header_data_column_mismatch_allowed")
    elif abs(prev["col_count"] - nxt["col_count"]) <= 1:
        score += 1
        reasons.append("similar_column_count")
    if similar_width(prev, nxt):
        score += 1
        reasons.append("similar_table_width")
    if nxt.get("synthetic_continuation"):
        score += 2
        reasons.append("synthetic_top_of_page_continuation")
    if nxt["raw_rows"] and any(is_data_row(row) for row in nxt["raw_rows"]):
        score += 2
        reasons.append("next_has_data_rows")
    elif nxt["raw_rows"] and not looks_like_header_row(nxt["raw_rows"][0]):
        score += 1
        reasons.append("next_first_row_looks_like_data")
    if (
        not header_only_prev
        and nxt["raw_rows"]
        and looks_like_header_row(nxt["raw_rows"][0])
        and not any(is_data_row(row) for row in nxt["raw_rows"])
    ):
        score -= 2
        reasons.append("next_has_header_like_first_row")

    if header_only_prev:
        # Cross-page merge for header-only tables: subsection stays with the header half.
        nxt["subsection_ref"] = prev.get("subsection_ref")
        if prev.get("section_ref"):
            nxt["section_ref"] = prev.get("section_ref")
    return score >= merge_threshold, score, reasons


def merge_connected_tables(tables: list[dict[str, Any]], doc: fitz.Document) -> list[dict[str, Any]]:
    tables = sorted(tables, key=lambda table: (table["page_start"], table["bbox"][1], table["bbox"][0]))
    groups: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(tables):
        group = [tables[idx]]
        idx += 1
        while idx < len(tables):
            should_merge, score, reasons = should_merge_tables(group[-1], tables[idx], doc)
            if not should_merge:
                break
            group[-1]["continued_to"] = tables[idx]["table_id"]
            tables[idx]["continued_from"] = group[-1]["table_id"]
            tables[idx]["merge_score"] = score
            tables[idx]["merge_reasons"] = reasons
            group.append(tables[idx])
            idx += 1
        groups.append(group)

    merged: list[dict[str, Any]] = []
    for group_idx, group in enumerate(groups, 1):
        merge_group_id = f"table_group_{group_idx:03d}"
        for table in group:
            table["merge_group_id"] = merge_group_id
        if len(group) == 1:
            merged.append(group[0])
            continue

        rows: list[list[str | None]] = []
        for table_idx, table in enumerate(group):
            table_rows = table["raw_rows"]
            if table_idx > 0 and table_rows and looks_like_header_row(table_rows[0]):
                table_rows = table_rows[1:]
            rows.extend(table_rows)
        score, reasons = table_complexity(rows, [
            min(table["bbox"][0] for table in group),
            group[0]["bbox"][1],
            max(table["bbox"][2] for table in group),
            group[-1]["bbox"][3],
        ])
        header_table = group[0]
        merged.append(
            {
                "table_id": merge_group_id,
                "asset_type": "table_group",
                "page_start": header_table["page_start"],
                "page_end": group[-1]["page_end"],
                "source_table_ids": [table["table_id"] for table in group],
                "bbox_by_page": [
                    {"page": int(table.get("page_start") or 0), "bbox": [round(x, 2) for x in table["bbox"]]}
                    for table in group
                ],
                "row_count": len(rows),
                "col_count": max((len(row) for row in rows), default=0),
                "raw_rows": rows,
                "raw_text": "\n".join(row_text(row) for row in rows),
                "complexity_score": score,
                "complexity_reasons": reasons + ["merged_across_pages"],
                "complexity": "complex" if score >= 5 else "simple",
                "section_ref": header_table["section_ref"],
                "subsection_ref": header_table["subsection_ref"],
                "subsection_attribution": "header_page",
                "merge_reasons": [table.get("merge_reasons", []) for table in group[1:]],
                "pending_merge": False,
                "header_only": False,
            }
        )
    return merged


def extract_images(doc: fitz.Document, section_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heading_positions = collect_heading_positions(doc, section_refs)
    images: list[dict[str, Any]] = []
    for page_idx in range(doc.page_count):
        page_no = page_idx + 1
        page = doc[page_idx]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 1:
                continue
            bbox = [round(x, 2) for x in block.get("bbox", [0, 0, 0, 0])]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < 80 or height < 60:
                continue
            section = find_section_for_page(section_refs, page_no)
            if section is None:
                continue
            subsection = find_subsection_for_asset(section, heading_positions, page_no, bbox[1])
            images.append(
                {
                    "image_id": f"image_{len(images) + 1:03d}",
                    "asset_type": "image",
                    "page": page_no,
                    "bbox": bbox,
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "section_ref": section_metadata(section),
                    "subsection_ref": subsection_metadata(subsection),
                }
            )
    return images


def compact_section_refs(section_refs: list[dict[str, Any]], tables: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_source_ids: set[str] = set()
    for table in tables:
        for source_id in table.get("source_table_ids") or []:
            grouped_source_ids.add(source_id)

    table_refs_by_section: dict[str, list[str]] = {}
    table_refs_by_subsection: dict[str, list[str]] = {}
    for table in tables:
        if table["table_id"] in grouped_source_ids:
            continue
        section_id = (table.get("section_ref") or {}).get("section_ref_id")
        subsection_id = (table.get("subsection_ref") or {}).get("subsection_ref_id")
        if section_id:
            table_refs_by_section.setdefault(section_id, []).append(table["table_id"])
        if subsection_id:
            table_refs_by_subsection.setdefault(subsection_id, []).append(table["table_id"])

    image_refs_by_section: dict[str, list[str]] = {}
    image_refs_by_subsection: dict[str, list[str]] = {}
    for image in images:
        section_id = (image.get("section_ref") or {}).get("section_ref_id")
        subsection_id = (image.get("subsection_ref") or {}).get("subsection_ref_id")
        if section_id:
            image_refs_by_section.setdefault(section_id, []).append(image["image_id"])
        if subsection_id:
            image_refs_by_subsection.setdefault(subsection_id, []).append(image["image_id"])

    compact = []
    for section in section_refs:
        compact_subsections = []
        for sub in section["subsections"]:
            compact_subsections.append(
                {
                    "subsection_ref_id": sub["subsection_ref_id"],
                    "title": sub["title"],
                    "path": sub["path"],
                    "level": sub["level"],
                    "page": sub["page"],
                    "char_count": sub["char_count"],
                    "table_refs": table_refs_by_subsection.get(sub["subsection_ref_id"], []),
                    "image_refs": image_refs_by_subsection.get(sub["subsection_ref_id"], []),
                }
            )
        section_table_refs = list(table_refs_by_section.get(section["section_ref_id"], []))
        for sub in compact_subsections:
            for table_id in sub.get("table_refs", []):
                if table_id not in section_table_refs:
                    section_table_refs.append(table_id)
        compact.append(
            {
                "section_ref_id": section["section_ref_id"],
                "part": section["part"],
                "item": section["item"],
                "section_title": section["section_title"],
                "start_pdf_page": section["start_pdf_page"],
                "end_pdf_page": section["end_pdf_page"],
                "char_count": section["char_count"],
                "table_refs": section_table_refs,
                "image_refs": image_refs_by_section.get(section["section_ref_id"], []),
                "subsections": compact_subsections,
            }
        )
    return compact


def subsection_lookup(section_refs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for section in section_refs:
        for sub in section["subsections"]:
            lookup[sub["subsection_ref_id"]] = sub
    return lookup


def line_index_by_offset(lines: list[dict[str, Any]], target: dict[str, Any]) -> int | None:
    target_offset = target.get("offset")
    for idx, line in enumerate(lines):
        if line.get("offset") == target_offset:
            return idx
    return None


def pick_table_anchor_line(
    caption_lines: list[dict[str, Any]],
    table_lines: list[dict[str, Any]],
    y0: float,
) -> dict[str, Any] | None:
    """Pick the narrative line immediately above the detected bbox (skip unit-header rows)."""
    near = [line for line in caption_lines if 0 < y0 - float(line["y0"]) <= 220]
    if near:
        ranked = sorted(near, key=lambda line: float(line["y0"]), reverse=True)
        for line in ranked:
            text = clean_text(line["line"])
            if len(text) < 12:
                continue
            if re.search(r"\(in millions|\(in billions|except percentages and per share", text, re.I):
                continue
            return line
        return ranked[0]
    return table_lines[0] if table_lines else None


def annotate_table_text_anchors(
    doc: fitz.Document,
    section_refs: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> None:
    """Map each table bbox to a substring anchor in its subsection text (for chunk markers)."""
    pages = collect_pages(doc)
    _, records = build_line_records(pages)
    sub_by_id = subsection_lookup(section_refs)

    for table in tables:
        subsection = table.get("subsection_ref") or {}
        sub_id = subsection.get("subsection_ref_id")
        sub = sub_by_id.get(sub_id or "")
        if not sub:
            continue

        page = int(table.get("page_start") or 0)
        bbox = table.get("bbox")
        if not bbox and table.get("bbox_by_page"):
            for crop in table["bbox_by_page"]:
                if int(crop.get("page") or 0) == page:
                    bbox = crop.get("bbox")
                    break
        bbox = bbox or [0, 0, 0, 0]
        y0 = float(bbox[1])
        sub_start = int(sub["start_offset"])
        sub_end = int(sub["end_offset"])

        page_lines = [
            record
            for record in records
            if record["page"] == page and sub_start <= record["offset"] < sub_end and record.get("y0") is not None
        ]
        if not page_lines:
            continue

        caption_lines = [line for line in page_lines if float(line["y0"]) < y0 - 2]
        table_lines = [line for line in page_lines if float(line["y0"]) >= y0 - 8]
        anchor_line = pick_table_anchor_line(caption_lines, table_lines, y0)
        if not anchor_line:
            continue

        insert_after = clean_text(anchor_line["line"])
        if caption_lines and len(insert_after) < 15:
            idx = line_index_by_offset(caption_lines, anchor_line)
            if idx is not None and idx > 0:
                insert_after = clean_text(caption_lines[idx - 1]["line"] + " " + insert_after)
        if len(insert_after) < 8 and caption_lines:
            insert_after = clean_text(caption_lines[-1]["line"])
        table["text_anchor"] = {
            "subsection_ref_id": sub_id,
            "global_offset": anchor_line["offset"],
            "local_offset": anchor_line["offset"] - sub_start,
            "insert_after_text": insert_after,
            "anchor_kind": "bbox_line",
            "page": page,
            "bbox_y0": y0,
        }


def merge_existing_vlm_parse(tables: list[dict[str, Any]], existing_assets_path: Path | None) -> None:
    if not existing_assets_path or not existing_assets_path.exists():
        return
    existing = json.loads(existing_assets_path.read_text(encoding="utf-8"))
    by_id = {table["table_id"]: table for table in existing.get("tables", [])}
    for table in tables:
        prior = by_id.get(table["table_id"]) or {}
        vlm = prior.get("vlm_parse")
        if vlm and vlm.get("status") == "success" and vlm.get("markdown"):
            table["vlm_parse"] = vlm


def annotate_table_anchors_only(pdf_path: Path, sections_path: Path, assets_path: Path) -> dict[str, Any]:
    """Fast path: refresh text_anchor metadata on existing assets without re-running find_tables."""
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    doc = fitz.open(pdf_path)
    section_refs = load_section_refs(sections_path)
    annotate_page_ranges(section_refs, doc.page_count)
    annotate_table_text_anchors(doc, section_refs, assets.get("tables", []))
    doc.close()
    assets_path.write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")
    anchored = sum(1 for table in assets.get("tables", []) if table.get("text_anchor"))
    return {"assets_path": str(assets_path), "tables_with_text_anchor": anchored}


def build_asset_payload(pdf_path: Path, sections_path: Path, existing_assets_path: Path | None = None) -> dict[str, Any]:
    import json

    from subsection_table_filter import filter_subsections_in_tables, relink_tables_to_subsections
    from toc_guided_section_probe import build_line_records, collect_pages

    doc = fitz.open(pdf_path)
    section_refs = load_section_refs(sections_path)
    annotate_page_ranges(section_refs, doc.page_count)

    pages = collect_pages(doc)
    full_text, records = build_line_records(pages)
    sections_payload = json.loads(sections_path.read_text(encoding="utf-8"))

    from table_pipeline import run_table_pipeline

    word_cache = PageWordCache()

    # Bbox-only pass for subsection filter — use a separate cache so warming
    # find_tables() for every page does not break tab.extract() on the pipeline pass.
    bbox_stubs = collect_table_bbox_stubs(doc, table_find_cache=PageTableFindCache())
    sections_payload, filter_stats = filter_subsections_in_tables(
        sections_payload, full_text, records, bbox_stubs
    )
    sections_path.write_text(json.dumps(sections_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    section_refs = load_section_refs(sections_path)
    annotate_page_ranges(section_refs, doc.page_count)
    heading_positions = collect_heading_positions(doc, section_refs)
    table_find_cache = PageTableFindCache()
    tables, pipeline_stats = run_table_pipeline(
        doc,
        section_refs,
        heading_positions,
        table_find_cache=table_find_cache,
        word_cache=word_cache,
    )
    merge_existing_vlm_parse(tables, existing_assets_path)
    relink_tables_to_subsections(tables, section_refs, heading_positions)
    annotate_table_text_anchors(doc, section_refs, tables)
    images = extract_images(doc, section_refs)
    sections = compact_section_refs(section_refs, tables, images)
    doc.close()

    source_file = sections_payload.get("source_file") or pdf_path.name

    return {
        "source_file": source_file,
        "sections_source_file": sections_path.name,
        "method": "table_pipeline_v2",
        "counts": {
            "sections": len(sections),
            "subsections": sum(len(section["subsections"]) for section in sections),
            "raw_regions_detected": pipeline_stats["raw_regions"],
            "after_same_page_link": pipeline_stats["after_same_page_link"],
            "after_dedupe": pipeline_stats["after_dedupe"],
            "tables_after_pipeline": pipeline_stats["final_tables"],
            "images_detected": len(images),
            "table_groups": pipeline_stats["table_groups"],
            "subsections_removed_in_table_filter": filter_stats["removed_subsections"],
        },
        "sections": sections,
        "tables": tables,
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach tables and images to menu-guided 10-K sections.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--annotate-anchors-only",
        action="store_true",
        help="Only refresh table text_anchor fields on an existing assets.json (no find_tables).",
    )
    parser.add_argument(
        "--merge-vlm-from",
        type=Path,
        default=None,
        help="Preserve vlm_parse from this existing assets.json when rebuilding tables.",
    )
    args = parser.parse_args()

    if args.annotate_anchors_only:
        if args.output == DEFAULT_OUTPUT and not args.output.exists():
            parser.error("--annotate-anchors-only requires --output pointing to existing assets.json")
        print(json.dumps(annotate_table_anchors_only(args.pdf, args.sections, args.output), indent=2))
        return

    merge_from = args.merge_vlm_from or (args.output if args.output.exists() else None)
    payload = build_asset_payload(args.pdf, args.sections, existing_assets_path=merge_from)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()

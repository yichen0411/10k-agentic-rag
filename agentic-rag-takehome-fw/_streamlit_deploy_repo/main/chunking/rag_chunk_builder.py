from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHUNKING_DIR = Path(__file__).resolve().parent
DEFAULT_SECTIONS = CHUNKING_DIR / "AAPL_FY2025_menu_guided_sections.json"
DEFAULT_ASSETS = CHUNKING_DIR / "AAPL_FY2025_section_assets.json"
DEFAULT_OUTPUT = CHUNKING_DIR / "AAPL_FY2025_rag_chunks.json"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def token_count(text: str) -> int:
    # Good enough for chunk sizing; embedding tokenizers can differ slightly.
    return len(re.findall(r"\S+", text))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_asset_lookup(assets: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    table_refs: dict[str, list[str]] = {}
    image_refs: dict[str, list[str]] = {}
    tables_by_id = {table["table_id"]: table for table in assets.get("tables", [])}

    for section in assets.get("sections", []):
        section_id = section["section_ref_id"]
        table_refs[section_id] = section.get("table_refs", [])
        image_refs[section_id] = section.get("image_refs", [])
        for sub in section.get("subsections", []):
            sub_id = sub["subsection_ref_id"]
            table_refs[sub_id] = sub.get("table_refs", [])
            image_refs[sub_id] = sub.get("image_refs", [])
    return table_refs, image_refs, tables_by_id


def section_label(part: str | None, item: str | None, title: str | None) -> str:
    return " > ".join(section_header_path(part, item, title))


_ITEM_NUM_RE = re.compile(r"^Item\s+(\d+)([A-Z]?)$", re.I)


def parse_item_parts(item: str | None) -> tuple[str | None, str | None]:
    """Base number and letter suffix: Item 1A -> ('1', 'A'); Item 1 -> ('1', None)."""
    if not item:
        return None, None
    match = _ITEM_NUM_RE.match(item.strip())
    if not match:
        return None, None
    return match.group(1), match.group(2) or None


def item_family_label(item: str | None) -> str | None:
    num, _ = parse_item_parts(item)
    if num:
        return f"Item {num}"
    return item


def item_label(item: str | None, title: str | None) -> str | None:
    if not item and not title:
        return None
    return ". ".join(part for part in [item, title] if part)


def section_header_path(part: str | None, item: str | None, title: str | None) -> list[str]:
    """
    Tree hierarchy for SEC Items:
      Part I > Item 1 > Business          (Item 1 body)
      Part I > Item 1 > Item 1A > Risk Factors   (1A/1B/1C nest under Item N, not after Item 2)
      Part I > Item 2 > Properties
    """
    num, suffix = parse_item_parts(item)
    title = clean_text(title or "")
    parts: list[str] = []
    if part:
        parts.append(part)
    if num:
        parts.append(f"Item {num}")
        if suffix:
            parts.append(f"Item {num}{suffix}")
            if title:
                parts.append(title)
        elif title:
            parts.append(title)
        elif item:
            parts.append(item)
    elif item:
        parts.append(item)
        if title:
            parts.append(title)
    return parts


def section_header_path_for_section(section: dict[str, Any]) -> list[str]:
    return section_header_path(section.get("part"), section.get("item"), section.get("toc_title") or section.get("section_title"))


def header_text(section_title: str, path: list[str] | None = None, part: str | None = None, item: str | None = None) -> str:
    parts = [section_label(part, item, section_title) if part or item else section_title]
    if path:
        parts.extend(path)
    deduped = []
    for part in parts:
        if part and part not in deduped:
            deduped.append(part)
    return " > ".join(deduped)


def strip_repeated_heading(text: str, heading: str) -> str:
    text = clean_text(text)
    heading = clean_text(heading)
    if heading and text.lower().startswith(heading.lower()):
        return clean_text(text[len(heading) :])
    return text


def strip_pdf_noise(text: str) -> str:
    text = re.sub(r"\[\[PAGE\s+\d+\]\]", " ", text)
    text = re.sub(r"Apple Inc\. \| 2025 Form 10-K \| \d+", " ", text)
    text = re.sub(r"Apple Inc\.", " ", text)
    text = re.sub(r"Item\s+\d+[A-Z]?\.\s+[^.]{0,140}?(?=(The|This|In|For|Beginning|Company|$))", " ", text)
    return clean_text(text)


def looks_like_table_residue(text: str) -> bool:
    words = re.findall(r"\S+", text)
    if len(words) < 5:
        return False
    numeric_words = sum(1 for word in words if re.search(r"\d", word))
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    if numeric_words / max(len(words), 1) > 0.30 and sentence_count == 0:
        return True
    return numeric_words / max(len(words), 1) > 0.22 and sentence_count < 3


def sentence_split_text(text: str, chunk_size: int = 430, chunk_overlap: int = 40) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        count = token_count(sentence)
        if count > chunk_size:
            if current:
                chunks.append(clean_text(" ".join(current)))
                current = []
                current_tokens = 0
            words = sentence.split()
            for start in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[start : start + chunk_size]))
            continue
        if current and current_tokens + count > chunk_size:
            chunks.append(clean_text(" ".join(current)))
            overlap = current[-2:] if chunk_overlap else []
            current = overlap + [sentence]
            current_tokens = token_count(" ".join(current))
        else:
            current.append(sentence)
            current_tokens += count
    if current:
        chunks.append(clean_text(" ".join(current)))

    final_chunks: list[str] = []
    for chunk in chunks:
        if token_count(chunk) <= 520:
            final_chunks.append(chunk)
            continue
        words = chunk.split()
        for start in range(0, len(words), 430):
            final_chunks.append(" ".join(words[start : start + 430]))
    return final_chunks


def chunk_with_header(body: str, header: str, min_tokens: int = 100, max_tokens: int = 500) -> list[str]:
    body = clean_text(body)
    if not body:
        return []
    header_prefix = f"{header}\n\n" if header else ""
    total = token_count(header_prefix + body)
    if total <= max_tokens:
        return [header_prefix + body]
    pieces = sentence_split_text(body, chunk_size=max_tokens - token_count(header_prefix) - 20)
    merged: list[str] = []
    for piece in pieces:
        if merged and token_count(header_prefix + piece) < min_tokens:
            merged[-1] = clean_text(merged[-1] + " " + piece)
        else:
            merged.append(piece)
    output: list[str] = []
    header_tokens = token_count(header_prefix)
    max_body_tokens = max(max_tokens - header_tokens, 80)
    for piece in merged:
        if token_count(header_prefix + piece) <= max_tokens:
            output.append(header_prefix + piece)
            continue
        words = piece.split()
        for start in range(0, len(words), max_body_tokens):
            output.append(header_prefix + " ".join(words[start : start + max_body_tokens]))
    return [chunk for chunk in output if clean_text(chunk)]


def iter_sections(section_data: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for part in section_data.get("parts", []):
        for item in part.get("items", []):
            sections.append({**item, "part": part["part"]})
    return sections


def row_values_for_matching(row: list[str | None]) -> list[str]:
    return [clean_text(cell) for cell in row if clean_text(cell or "")]


def row_has_numbers(row: list[str | None]) -> bool:
    return sum(1 for cell in row if cell and re.search(r"\d", cell)) >= 2


def row_pattern_usable(cells: list[str]) -> bool:
    if len(cells) < 3:
        return False
    numeric_cells = sum(1 for cell in cells if re.search(r"\d", cell))
    if numeric_cells >= 2:
        return True
    return any(re.search(r"20\d{2}|september|change|total|fair value|revenue|income|expenses|margin", c, re.I) for c in cells)


def table_row_patterns(table: dict[str, Any]) -> list[list[str]]:
    patterns: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add_cells(cells: list[str]) -> None:
        cells = [clean_text(re.sub(r"\*+", "", cell)) for cell in cells if clean_text(cell or "")]
        if not row_pattern_usable(cells):
            return
        key = tuple(cells[:4])
        if key in seen:
            return
        seen.add(key)
        patterns.append(cells)

    for row in table.get("raw_rows", []):
        add_cells(row_values_for_matching(row))

    raw_text = table.get("raw_text") or ""
    for line in raw_text.splitlines():
        add_cells([part.strip() for part in line.split("|")])

    markdown = ((table.get("vlm_parse") or {}).get("markdown") or "")
    for line in markdown.splitlines():
        if not line.strip().startswith("|") or re.match(r"^\|[\s:|-]+\|", line.strip()):
            continue
        add_cells([part.strip() for part in line.strip("|").split("|")])

    return patterns


def strip_table_rows(text: str, table_ids: list[str] | None, tables_by_id: dict[str, dict[str, Any]]) -> str:
    """Remove table-looking rows from subsection text while preserving narrative text."""
    stripped = clean_text(text)
    for table_id in table_ids or []:
        table = tables_by_id.get(table_id)
        if not table:
            continue
        for cells in table_row_patterns(table):
            pattern = r"\s+".join(re.escape(cell) for cell in cells)
            stripped = re.sub(pattern, " ", stripped, flags=re.I)
            if cells:
                label = re.escape(cells[0])
                number = next((re.escape(cell) for cell in cells[1:] if re.search(r"\d", cell)), None)
                if number:
                    stripped = re.sub(rf"{label}\s+{number}(?:\s+[\d,$().%-]+){{0,8}}", " ", stripped, flags=re.I)
    return clean_text(stripped)


TABLE_HEADER_PATTERNS = [
    r"\bSEGMENT RESULTS OF OPERATIONS\b",
    r"\b(?:CONSOLIDATED\s+)?(?:STATEMENTS?|SCHEDULES?) OF (?:INCOME|OPERATIONS|EARNINGS|COMPREHENSIVE INCOME|CASH FLOWS|FINANCIAL POSITION)\b",
    r"\b(?:INCOME STATEMENTS?|BALANCE SHEETS?|CASH FLOWS STATEMENTS?)\b",
    r"\b(?:The components of .{5,120}? were as follows)\.?\s*(?:\(?In millions[^)]*\)?)?",
    r"\bThe following table presents .{5,160}?\.?\s*(?:\(?In millions[^)]*\)?)?",
    r"\b(?:\(In millions[^)]*\)|In millions, except percentages[^)]*)\)\s*Year Ended\b",
    r"\b(?:\(In millions[^)]*\)|In millions, except percentages[^)]*)\)",
    r"\b(?:\(In billions[^)]*\)|In billions except[^)]*)\)",
    r"\b(?:\(In millions[^)]*\)|In millions\))\s*(?:Fair Value Level|Assets|Liabilities|Maturities \(calendar year\))\b",
]


NARRATIVE_RESUME_PATTERNS = [
    r"We\s+[A-Za-z]",
    r"Our\s+[A-Za-z]",
    r"The Company\s+[A-Za-z]",
    r"Following\s+(?:are|is)\s+[A-Za-z]",
    r"(?:In fiscal|As of|No instances|Substantially|Gross unrecognized|Income taxes|Other reconciling|See Note|See accompanying|Microsoft |Activision )",
    r"[A-Z][A-Za-z&',() -]{2,80}\s+(?:net sales|gross margin|operating income|operating expenses|provision|effective tax rate|cash|assets|liabilities|revenue|sales|income|expenses|margin)\s+(?:increased|decreased|was|were|remained|represented)",
]


def find_narrative_resume_index(tail: str) -> int | None:
    best: int | None = None
    for pattern in NARRATIVE_RESUME_PATTERNS:
        match = re.search(pattern, tail)
        if match and (best is None or match.start() < best):
            best = match.start()
    return best


TABLE_MARKER_RE = re.compile(r"\[\[TABLE:([^\]]+)\]\]")


def table_marker(table_id: str) -> str:
    return f"[[TABLE:{table_id}]]"


def strip_table_markers(text: str) -> str:
    return clean_text(TABLE_MARKER_RE.sub(" ", text or ""))


def find_header_strip_start(text: str) -> int | None:
    earliest: int | None = None
    for pattern in TABLE_HEADER_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    return earliest


def find_flattened_table_start(body: str, table: dict[str, Any] | None) -> int | None:
    """Find where this table's flattened PDF text begins using label + numeric cell pairs from assets."""
    if not table:
        return None
    earliest: int | None = None
    for row in table.get("raw_rows") or []:
        cells = [clean_text(cell) for cell in row if clean_text(cell or "")]
        if len(cells) < 2:
            continue
        label = cells[0]
        if len(label) < 3 or not re.search(r"[A-Za-z]", label):
            continue
        num_cell = next((cell for cell in cells[1:] if re.search(r"\d", cell or "")), None)
        if not num_cell:
            continue
        num = re.sub(r"[^\d,.()-]", "", num_cell)
        if not num:
            continue
        match = re.search(rf"\b{re.escape(label)}\b[\s\S]{{0,32}}{re.escape(num)}", body, flags=re.I)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    return earliest


def insert_table_markers(body: str, anchors: list[dict[str, Any]]) -> str:
    text = body
    for anchor in sorted(anchors, key=lambda item: item["char_offset"], reverse=True):
        pos = max(0, min(int(anchor["char_offset"]), len(text)))
        text = clean_text(text[:pos] + f" {anchor['marker']} " + text[pos:])
    return text


def strip_known_table_headers(text: str) -> str:
    """Drop flattened financial tables that start with a recognizable statement header."""
    cleaned = clean_text(text)
    earliest = find_header_strip_start(cleaned)
    if earliest is None or earliest == 0:
        return cleaned
    prefix = cleaned[:earliest].rstrip(" ,;:-")
    if prefix and not prefix.endswith("."):
        prefix += "."
    return clean_text(prefix)


def strip_numeric_table_runs(text: str) -> str:
    """Remove flattened table runs that PyMuPDF left inside narrative text."""
    if not re.search(r"\b(?:following tables?|were as follows|was as follows|as follows)\b", text, re.I):
        return text

    def clean_match(match: re.Match[str]) -> str:
        prefix = match.group("prefix").rstrip(": ,;") + "."
        tail = match.group("tail")
        resume = find_narrative_resume_index(tail)
        if resume is not None:
            return clean_text(prefix + " " + tail[resume:])
        return prefix

    return clean_text(
        re.sub(
            r"(?P<prefix>[^.]{0,320}\b(?:following tables?|were as follows|was as follows|as follows)[^:]{0,320}):(?P<tail>.*)",
            clean_match,
            text,
            flags=re.I,
        )
    )


def strip_page_table_noise(text: str) -> str:
    """Remove page markers and trailing numeric-only tails attached to table flattening."""
    text = re.sub(r"\b\d{1,3}\s+PART\s+[IVXLC]+\s+Item\s+\d+[A-Z]?\b", " ", text, flags=re.I)
    return clean_text(text)


def strip_flattened_financial_tables(text: str, table_ids: list[str] | None, tables_by_id: dict[str, dict[str, Any]]) -> str:
    text = strip_known_table_headers(text)
    text = strip_numeric_table_runs(text)
    text = strip_table_rows(text, table_ids, tables_by_id)
    text = strip_page_table_noise(text)
    return clean_text(text)


def find_header_row_start(body: str, table: dict[str, Any] | None) -> int | None:
    if not table:
        return None
    body_lower = body.lower()
    for row in table.get("raw_rows") or []:
        cells = [clean_text(cell) for cell in row if clean_text(cell or "")]
        if len(cells) < 3:
            continue
        if any(re.search(r"\d", cell or "") for cell in cells):
            continue
        for width in (min(5, len(cells)), 3, 2):
            needle = " ".join(cells[:width]).lower()
            idx = body_lower.find(needle)
            if idx >= 0:
                return idx
    return None


def table_strip_start(body: str, table: dict[str, Any] | None) -> int | None:
    """Earliest char offset where flattened table content for this asset begins."""
    row_start = find_flattened_table_start(body, table)
    header_row_start = find_header_row_start(body, table)
    header_start = find_header_strip_start(body)
    starts = [pos for pos in (row_start, header_row_start, header_start) if pos is not None]
    return min(starts) if starts else None


def clean_body_for_text_chunk(
    body: str,
    heading: str,
    table_ids: list[str],
    tables_by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    body = strip_repeated_heading(body, heading)
    body = strip_pdf_noise(body)
    strip_starts: dict[str, int] = {}
    for table_id in table_ids or []:
        strip_at = table_strip_start(body, tables_by_id.get(table_id))
        if strip_at is not None:
            strip_starts[table_id] = strip_at
    if len(strip_starts) == 1:
        earliest = next(iter(strip_starts.values()))
        prefix = body[:earliest].rstrip(" ,;:-")
        prefix = re.sub(r"\(\s*$", "", prefix).rstrip(" ,;:-")
        if prefix and not prefix.endswith("."):
            prefix += "."
        body = clean_text(prefix)
        if looks_like_table_residue(body):
            body = ""
    elif len(strip_starts) > 1:
        # Multiple tables in one subsection: keep inter-table narrative, strip rows later.
        pass
    body = strip_flattened_financial_tables(body, table_ids, tables_by_id)
    body = strip_pdf_noise(body)
    if looks_like_table_residue(body):
        if not strip_starts:
            return "", [], []
        body = ""

    anchors: list[dict[str, Any]] = []
    chunk_refs: list[str] = []
    ordered_ids = sorted(strip_starts, key=lambda table_id: strip_starts[table_id])
    if ordered_ids:
        if len(ordered_ids) == 1:
            insert_at = len(body)
            for table_id in ordered_ids:
                table = tables_by_id.get(table_id) or {}
                anchor_kind = (table.get("text_anchor") or {}).get("anchor_kind") or "strip_boundary"
                anchors.append(
                    {
                        "table_id": table_id,
                        "char_offset": insert_at,
                        "marker": table_marker(table_id),
                        "anchor_kind": anchor_kind,
                    }
                )
                chunk_refs.append(table_id)
            body = insert_table_markers(body, anchors)
        else:
            parts: list[str] = []
            cursor = 0
            for table_id in ordered_ids:
                strip_at = strip_starts[table_id]
                prefix = body[cursor:strip_at].rstrip(" ,;:-")
                if prefix and not prefix.endswith("."):
                    prefix += "."
                if prefix:
                    parts.append(prefix)
                marker = table_marker(table_id)
                parts.append(marker)
                table = tables_by_id.get(table_id) or {}
                anchor_kind = (table.get("text_anchor") or {}).get("anchor_kind") or "strip_boundary"
                offset = len(clean_text(" ".join(parts))) - len(marker)
                anchors.append(
                    {
                        "table_id": table_id,
                        "char_offset": max(0, offset),
                        "marker": marker,
                        "anchor_kind": anchor_kind,
                    }
                )
                chunk_refs.append(table_id)
                next_strips = [strip_starts[tid] for tid in ordered_ids if strip_starts[tid] > strip_at]
                cursor = min(next_strips) if next_strips else len(body)
            tail = clean_text(body[cursor:])
            if tail:
                parts.append(tail)
            body = clean_text(" ".join(parts))

    return clean_text(body), anchors, chunk_refs


def anchors_for_chunk_text(text: str, anchors: list[dict[str, Any]], chunk_refs: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    if not anchors:
        return [], []
    piece_anchors: list[dict[str, Any]] = []
    piece_refs: list[str] = []
    for anchor in anchors:
        match = re.search(re.escape(anchor["marker"]), text)
        if not match:
            continue
        piece_anchors.append({**anchor, "char_offset": match.start()})
        piece_refs.append(anchor["table_id"])
    return piece_anchors, piece_refs


def append_text_chunk(
    chunks: list[dict[str, Any]],
    *,
    chunk_index: int,
    section_data: dict[str, Any],
    section_ref_id: str,
    section_title: str,
    section_header_path: list[str],
    sub_ref_id: str | None,
    text_unit_id: str,
    text_unit_kind: str,
    header_path: list[str],
    text: str,
    table_refs: list[str],
    table_anchors: list[dict[str, Any]],
    image_refs: list[str],
    split_index: int,
    split_count: int,
) -> int:
    piece_anchors, piece_refs = anchors_for_chunk_text(text, table_anchors, table_refs)
    chunks.append(
        {
            "chunk_id": f"text_{chunk_index:05d}",
            "chunk_type": "text",
            "source_file": section_data["source_file"],
            "section_ref_id": section_ref_id,
            "subsection_ref_id": sub_ref_id,
            "text_unit_id": text_unit_id,
            "text_unit_kind": text_unit_kind,
            "section_title": section_title,
            "header_path": header_path,
            "text": text,
            "token_count": token_count(text),
            "table_refs": piece_refs or table_refs,
            "table_anchors": piece_anchors,
            "image_refs": image_refs,
            "split_index": split_index,
            "split_count": split_count,
        }
    )
    return chunk_index + 1


def annotate_inference_links(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add metadata for context expansion without crossing section boundaries."""
    by_section: dict[str, list[dict[str, Any]]] = {}
    by_text_unit: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        section_ref_id = chunk["section_ref_id"]
        text_unit_id = chunk["text_unit_id"]
        by_section.setdefault(section_ref_id, []).append(chunk)
        by_text_unit.setdefault(text_unit_id, []).append(chunk)

    for group in by_text_unit.values():
        for idx, chunk in enumerate(group):
            chunk["same_text_unit_prev_chunk_id"] = group[idx - 1]["chunk_id"] if idx > 0 else None
            chunk["same_text_unit_next_chunk_id"] = group[idx + 1]["chunk_id"] if idx + 1 < len(group) else None
            chunk["is_split_continuation"] = idx > 0

    for _section_ref_id, group in by_section.items():
        for idx, chunk in enumerate(group):
            chunk["same_section_prev_chunk_id"] = group[idx - 1]["chunk_id"] if idx > 0 else None
            chunk["same_section_next_chunk_id"] = group[idx + 1]["chunk_id"] if idx + 1 < len(group) else None
            chunk["section_preamble_refs"] = []
            chunk["neighbor_expansion_scope"] = "same_section"
            chunk["cross_section_expansion_allowed"] = False
    return chunks


def build_text_chunks(section_data: dict[str, Any], assets: dict[str, Any]) -> list[dict[str, Any]]:
    table_refs, image_refs, tables_by_id = build_asset_lookup(assets)
    chunks: list[dict[str, Any]] = []
    chunk_index = 1

    for section in iter_sections(section_data):
        section_ref_id = f"{section['part']}::{section['item']}"
        section_title = section["toc_title"]
        section_header_path = section_header_path_for_section(section)
        preamble = clean_text(section.get("preamble_text", ""))
        if not preamble and not section.get("subsection_chunks"):
            # Back-compat: sections processed before attach_subsections filled preamble.
            preamble = clean_text(section.get("text", ""))

        subsection_units = list(section.get("subsection_chunks") or [])
        if preamble and subsection_units:
            first = dict(subsection_units[0])
            first["text"] = clean_text(f"{preamble} {first.get('text', '')}")
            subsection_units = [first, *subsection_units[1:]]
        elif preamble and not subsection_units:
            subsection_units = [{"title": section_title, "path": [], "text": preamble}]

        for sub_idx, sub in enumerate(subsection_units, 1):
            path = sub.get("path", [sub["title"]])
            header = header_text(section_title, path, part=section.get("part"), item=section.get("item"))
            sub_ref_id = sub.get("subsection_ref_id") or f"{section['part']}::{section['item']}::sub_{sub_idx:03d}"
            refs = table_refs.get(sub_ref_id, [])
            body, body_anchors, body_refs = clean_body_for_text_chunk(sub.get("text", ""), sub["title"], refs, tables_by_id)
            pieces = chunk_with_header(body, header)
            for piece_idx, text in enumerate(pieces, 1):
                chunk_index = append_text_chunk(
                    chunks,
                    chunk_index=chunk_index,
                    section_data=section_data,
                    section_ref_id=section_ref_id,
                    section_title=section_title,
                    section_header_path=section_header_path,
                    sub_ref_id=sub_ref_id,
                    text_unit_id=sub_ref_id,
                    text_unit_kind="subsection",
                    header_path=[*section_header_path, *path],
                    text=text,
                    table_refs=body_refs or refs,
                    table_anchors=body_anchors,
                    image_refs=image_refs.get(sub_ref_id, []),
                    split_index=piece_idx,
                    split_count=len(pieces),
                )
    return annotate_inference_links(chunks)


def table_to_text(table: dict[str, Any]) -> str:
    section = table.get("section_ref") or {}
    subsection = table.get("subsection_ref") or {}
    path = subsection.get("path") or []
    header = header_text(section.get("section_title", ""), path, part=section.get("part"), item=section.get("item"))
    rows = table.get("raw_rows", [])
    row_lines = [" | ".join(cell or "" for cell in row) for row in rows]
    table_body = "\n".join(row_lines)
    return clean_text(f"{header}\n\nTable on page {table.get('page_start')}. Rows:\n{table_body}")


def build_table_chunks(assets: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for idx, table in enumerate(assets.get("tables", []), 1):
        section = table.get("section_ref") or {}
        subsection = table.get("subsection_ref") or {}
        chunks.append(
            {
                "chunk_id": f"table_{idx:05d}",
                "chunk_type": "table",
                "source_file": assets["source_file"],
                "table_id": table["table_id"],
                "section_title": section.get("section_title"),
                "header_path": [*section_header_path(section.get("part"), section.get("item"), section.get("section_title")), *(subsection.get("path") or [])],
                "text": table_to_text(table),
                "token_count": token_count(table_to_text(table)),
                "page_start": table.get("page_start"),
                "page_end": table.get("page_end"),
                "bbox": table.get("bbox") or table.get("bbox_by_page"),
            }
        )
    return chunks


def build_image_chunks(assets: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for idx, image in enumerate(assets.get("images", []), 1):
        section = image.get("section_ref") or {}
        subsection = image.get("subsection_ref") or {}
        header = header_text(section.get("section_title", ""), subsection.get("path") or [], part=section.get("part"), item=section.get("item"))
        text = clean_text(f"{header}\n\nImage/figure on page {image.get('page')} with bbox {image.get('bbox')}.")
        chunks.append(
            {
                "chunk_id": f"image_{idx:05d}",
                "chunk_type": "image",
                "source_file": assets["source_file"],
                "image_id": image["image_id"],
                "section_title": section.get("section_title"),
                "header_path": [*section_header_path(section.get("part"), section.get("item"), section.get("section_title")), *(subsection.get("path") or [])],
                "text": text,
                "token_count": token_count(text),
                "page": image.get("page"),
                "bbox": image.get("bbox"),
            }
        )
    return chunks


def drop_empty_headers(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for chunk in chunks:
        chunk["header_path"] = [part for part in chunk.get("header_path", []) if part]
    return chunks


def build_rag_payload(sections_path: Path, assets_path: Path) -> dict[str, Any]:
    section_data = load_json(sections_path)
    assets = load_json(assets_path)
    text_chunks = build_text_chunks(section_data, assets)
    chunks = drop_empty_headers(text_chunks)
    return {
        "source_file": section_data["source_file"],
        "method": "text_only_rag_chunks_with_header_paths_table_refs",
        "counts": {
            "chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "table_chunks": 0,
            "image_chunks": 0,
            "chunks_over_500_tokens": sum(1 for chunk in chunks if chunk["token_count"] > 500),
            "chunks_under_100_tokens": sum(1 for chunk in chunks if chunk["token_count"] < 100),
            "chunks_with_table_refs": sum(1 for chunk in chunks if chunk.get("table_refs")),
            "chunks_with_table_anchors": sum(1 for chunk in chunks if chunk.get("table_anchors")),
            "chunks_with_image_refs": sum(1 for chunk in chunks if chunk.get("image_refs")),
        },
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG-friendly chunks from section and asset extraction outputs.")
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_rag_payload(args.sections, args.assets)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()

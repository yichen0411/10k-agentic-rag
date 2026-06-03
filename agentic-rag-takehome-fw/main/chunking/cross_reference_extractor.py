"""Extract intra-filing cross-references from 10-K text chunks.

Detects cue phrases (e.g. "refer to", "see", "discussed in") followed by
structured targets (Note N, Item N, Part X, Exhibit N, etc.) and returns
normalized metadata for each match.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Cue phrases — verbs / prepositional phrases that introduce a cross-ref
# ---------------------------------------------------------------------------
CROSS_REF_CUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("refer_to", re.compile(r"\brefer(?:s|red|ring)?\s+to\b", re.I)),
    ("see", re.compile(r"\bsee(?:\s+also)?\b", re.I)),
    ("discussed_in", re.compile(r"\b(?:as\s+)?discussed\s+in\b", re.I)),
    ("described_in", re.compile(r"\b(?:as\s+)?described\s+in\b", re.I)),
    ("noted_in", re.compile(r"\b(?:as\s+)?noted\s+in\b", re.I)),
    ("detailed_in", re.compile(r"\b(?:as\s+)?detailed\s+in\b", re.I)),
    ("outlined_in", re.compile(r"\b(?:as\s+)?outlined\s+in\b", re.I)),
    ("explained_in", re.compile(r"\b(?:as\s+)?explained\s+in\b", re.I)),
    ("presented_in", re.compile(r"\bpresented\s+in\b", re.I)),
    ("set_forth_in", re.compile(r"\bset\s+forth\s+in\b", re.I)),
    ("included_in", re.compile(r"\bincluded\s+in\b", re.I)),
    ("found_in", re.compile(r"\bfound\s+in\b", re.I)),
    ("contained_in", re.compile(r"\bcontained\s+in\b", re.I)),
    ("read_in_conjunction_with", re.compile(r"\bread\s+in\s+conjunction\s+with\b", re.I)),
    ("incorporated_by_reference", re.compile(r"\bincorporated\s+(?:herein\s+)?by\s+reference\b", re.I)),
    ("pursuant_to", re.compile(r"\bpursuant\s+to\b", re.I)),
    ("under_the_heading", re.compile(r"\bunder\s+the\s+heading\b", re.I)),
    ("in_accordance_with", re.compile(r"\bin\s+accordance\s+with\b", re.I)),
    ("for_more_information", re.compile(r"\bfor\s+more\s+information\s+(?:on|about|regarding)\b", re.I)),
    ("described_under", re.compile(r"\b(?:as\s+)?described\s+under\b", re.I)),
    ("discussed_under", re.compile(r"\b(?:as\s+)?discussed\s+under\b", re.I)),
]

# Combined cue matcher (first match wins at a position).
_ALL_CUES = re.compile(
    r"(?:"
    r"\brefer(?:s|red|ring)?\s+to\b"
    r"|\bsee(?:\s+also)?\b"
    r"|\b(?:as\s+)?discussed\s+in\b"
    r"|\b(?:as\s+)?described\s+in\b"
    r"|\b(?:as\s+)?noted\s+in\b"
    r"|\b(?:as\s+)?detailed\s+in\b"
    r"|\b(?:as\s+)?outlined\s+in\b"
    r"|\b(?:as\s+)?explained\s+in\b"
    r"|\bpresented\s+in\b"
    r"|\bset\s+forth\s+in\b"
    r"|\bincluded\s+in\b"
    r"|\bfound\s+in\b"
    r"|\bcontained\s+in\b"
    r"|\bread\s+in\s+conjunction\s+with\b"
    r"|\bincorporated\s+(?:herein\s+)?by\s+reference\b"
    r"|\bpursuant\s+to\b"
    r"|\bunder\s+the\s+heading\b"
    r"|\bin\s+accordance\s+with\b"
    r"|\bfor\s+more\s+information\s+(?:on|about|regarding)\b"
    r"|\b(?:as\s+)?described\s+under\b"
    r"|\b(?:as\s+)?discussed\s+under\b"
    r")",
    re.I,
)

# ---------------------------------------------------------------------------
# Target patterns — what follows the cue phrase
# ---------------------------------------------------------------------------
_PART = r"Part\s+(?:[IVXLC]+|\d+)"
_ITEM = r"Item\s+\d+[A-Z]?"
_NOTE_NUM = r"Note\s+\d+"
_NOTE_TITLE = rf"{_NOTE_NUM}\s*(?:[–—\-:]\s*[^.;,\(]+)?"
_NOTES_BLOCK = r"(?:the\s+)?Notes?\s+to\s+(?:the\s+)?(?:Consolidated\s+)?Financial\s+Statements?"
_EXHIBIT = r"Exhibit\s+\d+(?:\.\d+)?"
_SCHEDULE = r"Schedule\s+(?:[IVXLC]+|\d+)"
_APPENDIX = r"Appendix\s+[A-Z0-9]+"
_SECTION = r"Section\s+\d+(?:\([a-z]\))?"
_TABLE = r"(?:the\s+)?(?:following\s+)?[Tt]able(?:\s+\d+)?"
_FORM = r"(?:this\s+)?(?:Form\s+10-[KQ]|Annual\s+Report\s+on\s+Form\s+10-K)"
_PROXY = r"(?:the\s+)?(?:\d{4}\s+)?(?:Definitive\s+)?Proxy\s+Statement"
_HEADING_QUOTED = r'["\u201c\u201d]([^"\u201c\u201d]{3,120})["\u201c\u201d]'

# Ordered: more specific targets first.
TARGET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "note_with_location",
        re.compile(
            rf"(?P<label>{_NOTE_TITLE})\s+of\s+(?:the\s+)?{_NOTES_BLOCK}"
            rf"(?:\s*\(\s*(?P<part>{_PART})\s*,\s*(?P<item>{_ITEM})[^)]*\))?",
            re.I,
        ),
    ),
    (
        "note_simple",
        re.compile(
            rf"(?P<label>{_NOTE_TITLE})(?:\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements)?",
            re.I,
        ),
    ),
    (
        "notes_block",
        re.compile(rf"(?P<label>{_NOTES_BLOCK})", re.I),
    ),
    (
        "proxy_statement",
        re.compile(rf"(?P<label>{_PROXY})", re.I),
    ),
    (
        "part_item_with_title",
        re.compile(
            rf"(?P<part>{_PART})\s*,\s*(?P<item>{_ITEM})(?:\s*,\s*[\"\\u201c\\u201d](?P<title>[^\"\\u201c\\u201d]+)[\"\\u201c\\u201d])?",
            re.I,
        ),
    ),
    (
        "item_with_form",
        re.compile(
            rf"(?P<item>{_ITEM})\s+of\s+(?:this\s+)?(?:Form\s+10-[KQ]|Annual\s+Report)",
            re.I,
        ),
    ),
    (
        "part_item",
        re.compile(rf"(?P<part>{_PART})\s*,\s*(?P<item>{_ITEM})", re.I),
    ),
    (
        "prior_report",
        re.compile(
            r'(?P<label>[^"\u201c\u201d]{10,160}?(?:Annual\s+Report\s+on\s+Form\s+10-K|Form\s+10-K)[^"\u201c\u201d]{0,80})',
            re.I,
        ),
    ),
    (
        "item_only",
        re.compile(rf"(?P<item>{_ITEM})", re.I),
    ),
    (
        "part_only",
        re.compile(rf"(?P<part>{_PART})", re.I),
    ),
    (
        "exhibit",
        re.compile(rf"(?P<label>{_EXHIBIT})", re.I),
    ),
    (
        "schedule",
        re.compile(rf"(?P<label>{_SCHEDULE})", re.I),
    ),
    (
        "appendix",
        re.compile(rf"(?P<label>{_APPENDIX})", re.I),
    ),
    (
        "section",
        re.compile(rf"(?P<label>{_SECTION})", re.I),
    ),
    (
        "table",
        re.compile(rf"(?P<label>{_TABLE})", re.I),
    ),
    (
        "form",
        re.compile(rf"(?P<label>{_FORM})", re.I),
    ),
    (
        "heading",
        re.compile(
            rf"(?:under\s+the\s+heading\s+)?(?P<label>{_HEADING_QUOTED})",
            re.I,
        ),
    ),
]

# Targets that validate location-style cues (included in / found in / contained in).
_LOCATION_CUES = frozenset(
    {
        "included_in",
        "found_in",
        "contained_in",
        "presented_in",
        "set_forth_in",
        "read_in_conjunction_with",
    }
)

# Structural target types accepted for location-style cues.
_LOCATION_TARGET_TYPES = frozenset(
    {
        "note_with_location",
        "note_simple",
        "notes_block",
        "part_item_with_title",
        "part_item",
        "item_with_form",
        "item_only",
        "part_only",
        "form",
        "proxy_statement",
        "prior_report",
        "heading",
    }
)

# Cues where target may appear *before* the cue (incorporated by reference).
_REVERSE_CUE_TYPES = frozenset({"incorporated_by_reference"})

# Per-cue allowed target types (None = all targets allowed).
_CUE_ALLOWED_TARGETS: dict[str, frozenset[str] | None] = {
    "refer_to": frozenset(
        {
            "note_with_location",
            "note_simple",
            "notes_block",
            "part_item_with_title",
            "part_item",
            "item_with_form",
            "item_only",
            "exhibit",
            "schedule",
            "section",
            "table",
            "proxy_statement",
            "form",
            "prior_report",
        }
    ),
    "see": frozenset(
        {
            "note_with_location",
            "note_simple",
            "notes_block",
            "part_item_with_title",
            "part_item",
            "item_with_form",
            "item_only",
            "exhibit",
            "section",
            "table",
            "heading",
        }
    ),
    "under_the_heading": frozenset({"heading"}),
    "pursuant_to": frozenset({"section", "item_only", "part_item", "part_item_with_title"}),
    "in_accordance_with": frozenset({"section", "part_item_with_title", "part_item", "item_only"}),
    "incorporated_by_reference": frozenset({"proxy_statement", "form", "exhibit", "prior_report", "part_item_with_title"}),
}

_MAX_TARGET_WINDOW = 220


def _cue_type_at(text: str, pos: int) -> str | None:
    for cue_type, pattern in CROSS_REF_CUE_PATTERNS:
        m = pattern.search(text, pos, min(len(text), pos + 40))
        if m and m.start() == pos:
            return cue_type
    return None


def _normalize_label(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip(" ,;."))


def _build_target_record(
    target_type: str,
    match: re.Match[str],
    *,
    cue_type: str,
    raw_span: str,
    char_start: int,
    char_end: int,
) -> dict[str, Any]:
    groups = match.groupdict()
    label = groups.get("label") or groups.get("item") or groups.get("part") or match.group(0)
    record: dict[str, Any] = {
        "cue_type": cue_type,
        "target_type": target_type,
        "target_label": _normalize_label(label),
        "raw_span": _normalize_label(raw_span),
        "char_start": char_start,
        "char_end": char_end,
    }
    if groups.get("part"):
        record["part"] = _normalize_label(groups["part"])
    if groups.get("item"):
        record["item"] = _normalize_label(groups["item"])
    if groups.get("title"):
        record["title"] = _normalize_label(groups["title"])
    return record


def _extract_target_after_cue(text: str, cue_end: int, cue_type: str) -> dict[str, Any] | None:
    window = text[cue_end : cue_end + _MAX_TARGET_WINDOW]
    window = re.sub(r"^\s*(?:the|our|Company(?:'s)?|Registrant(?:'s)?)\s+", "", window, count=1, flags=re.I)

    for target_type, pattern in TARGET_PATTERNS:
        m = pattern.search(window)
        if not m:
            continue
        if cue_type in _LOCATION_CUES and target_type not in _LOCATION_TARGET_TYPES:
            continue
        allowed = _CUE_ALLOWED_TARGETS.get(cue_type)
        if allowed is not None and target_type not in allowed:
            continue
        if cue_type == "refer_to":
            # "refer to the Company" / "refer to website URLs" are not filing cross-refs.
            skip_prefixes = ("the Company", "website", "our ", "each ", "those ")
            frag = window[: m.start()].lower()
            if any(skip in frag for skip in skip_prefixes):
                continue
        abs_start = cue_end + m.start()
        abs_end = cue_end + m.end()
        raw_span = text[max(0, abs_start - 40) : min(len(text), abs_end + 40)]
        return _build_target_record(
            target_type,
            m,
            cue_type=cue_type,
            raw_span=raw_span,
            char_start=abs_start,
            char_end=abs_end,
        )
    return None


def _extract_target_before_cue(text: str, cue_start: int, cue_type: str) -> dict[str, Any] | None:
    window_start = max(0, cue_start - _MAX_TARGET_WINDOW)
    window = text[window_start:cue_start]
    for target_type, pattern in TARGET_PATTERNS:
        matches = list(pattern.finditer(window))
        if not matches:
            continue
        m = matches[-1]
        if target_type not in {"proxy_statement", "form", "exhibit", "prior_report"}:
            continue
        abs_start = window_start + m.start()
        abs_end = window_start + m.end()
        raw_span = text[max(0, abs_start - 20) : min(len(text), cue_start + 40)]
        return _build_target_record(
            target_type,
            m,
            cue_type=cue_type,
            raw_span=raw_span,
            char_start=abs_start,
            char_end=abs_end,
        )
    return None


def extract_cross_references(text: str) -> list[dict[str, Any]]:
    """Return structured cross-reference records found in *text*."""
    if not text or not text.strip():
        return []

    refs: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int, str]] = set()

    for cue_match in _ALL_CUES.finditer(text):
        cue_start = cue_match.start()
        cue_end = cue_match.end()
        cue_type = _cue_type_at(text, cue_start) or "unknown"

        target: dict[str, Any] | None
        if cue_type in _REVERSE_CUE_TYPES:
            target = _extract_target_before_cue(text, cue_start, cue_type)
            if target is None:
                target = _extract_target_after_cue(text, cue_end, cue_type)
        else:
            target = _extract_target_after_cue(text, cue_end, cue_type)

        if target is None:
            continue

        # "under the heading" is handled by its own cue; skip heading-only follow-ups from other cues.
        if cue_type != "under_the_heading" and target["target_type"] == "heading":
            if cue_type not in {"see", "discussed_in", "described_in", "noted_in"}:
                continue

        dedupe_key = (target["char_start"], target["char_end"], target["target_label"].lower())
        if dedupe_key in seen_spans:
            continue
        seen_spans.add(dedupe_key)
        refs.append(target)

    return refs


def extract_cross_references_from_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract cross-refs from a chunk dict (uses body text after header path)."""
    text = chunk.get("text") or ""
    body = text.split("\n\n", 1)[-1] if "\n\n" in text else text
    header_len = len(text) - len(body)
    refs = extract_cross_references(body)
    for ref in refs:
        ref["char_start"] += header_len
        ref["char_end"] += header_len
    return refs


def annotate_chunks_with_cross_refs(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for chunk in chunks:
        chunk["cross_refs"] = extract_cross_references_from_chunk(chunk)
    return chunks


def summarize_cross_refs(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    all_refs = [ref for chunk in chunks for ref in chunk.get("cross_refs") or []]
    by_cue: dict[str, int] = {}
    by_target: dict[str, int] = {}
    for ref in all_refs:
        by_cue[ref["cue_type"]] = by_cue.get(ref["cue_type"], 0) + 1
        by_target[ref["target_type"]] = by_target.get(ref["target_type"], 0) + 1
    chunks_with_refs = sum(1 for chunk in chunks if chunk.get("cross_refs"))
    return {
        "total_refs": len(all_refs),
        "chunks_with_refs": chunks_with_refs,
        "total_chunks": len(chunks),
        "by_cue_type": dict(sorted(by_cue.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_target_type": dict(sorted(by_target.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

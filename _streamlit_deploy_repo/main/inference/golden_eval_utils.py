"""Shared helpers for MSFT golden dataset generation and evaluation."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

YEAR_RE = re.compile(r"\b(?:FY)?20\d{2}\b")
NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|percent|%))?", re.I)
CAP_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9&-]+(?:\s+[A-Z][A-Za-z0-9&-]+){0,4}\b")


def normalize_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("::")[-1] if "::" in chunk_id else chunk_id


def parse_json_response(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[") if "[" in text else text.find("{")
        end = text.rfind("]") if start == text.find("[") and start != -1 else text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])


def load_text_chunks_from_db(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, content, section, metadata_json FROM chunks ORDER BY id"
    ).fetchall()
    conn.close()
    chunks: list[dict[str, Any]] = []
    for row in rows:
        meta = json.loads(row["metadata_json"])
        chunks.append(
            {
                "chunk_id": row["chunk_id"],
                "short_id": normalize_chunk_id(row["chunk_id"]),
                "content": row["content"],
                "section": row["section"],
                "metadata": meta,
                "header_path": meta.get("header_path") or [],
                "table_refs": meta.get("table_refs") or [],
            }
        )
    return chunks


def is_good_text_candidate(chunk: dict[str, Any], *, min_chars: int = 350, min_words: int = 60) -> bool:
    content = chunk.get("content") or ""
    if len(content) < min_chars:
        return False
    if content.count("[[TABLE:") > 4:
        return False
    if len(content.split()) < min_words:
        return False
    return True


def keyword_signals(text: str) -> dict[str, list[str]]:
    years = sorted(set(YEAR_RE.findall(text)))[:6]
    numbers = sorted(set(NUMBER_RE.findall(text)), key=len, reverse=True)[:8]
    phrases = []
    for match in CAP_PHRASE_RE.findall(text):
        if len(match) > 4 and match not in {"Microsoft", "Company", "Part II", "Part I"}:
            phrases.append(match)
        if len(phrases) >= 8:
            break
    return {"years": years, "numbers": numbers, "phrases": phrases[:8]}


def rank_position(hits: list[dict[str, Any]], matcher) -> int | None:
    for idx, hit in enumerate(hits, 1):
        if matcher(hit):
            return idx
    return None


RECALL_KS_DEFAULT = (1, 3, 5, 10, 15)


def recall_at_k_from_rank(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def recall_at_k_dict(rank: int | None, ks: tuple[int, ...] = RECALL_KS_DEFAULT) -> dict[str, bool]:
    return {str(k): recall_at_k_from_rank(rank, k) for k in ks}


def hit_rate_at_k(hits: list[dict[str, Any]], k: int, matcher) -> bool:
    for hit in hits[:k]:
        if matcher(hit):
            return True
    return False


QUESTION_TYPES = ("text_semantic", "text_keyword", "table")
TEXT_QUESTION_TYPES = ("text_semantic", "text_keyword")


def select_text_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in questions if row.get("question_type") in TEXT_QUESTION_TYPES]


def select_table_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in questions if row.get("question_type") == "table"]


def table_ground_truth_ids(row: dict[str, Any], table_lookup: dict[str, dict[str, Any]] | None = None) -> set[str]:
    ids: set[str] = set()
    for raw_id in [row.get("ground_truth_chunk_id"), *(row.get("expected_table_ids") or [])]:
        if not raw_id:
            continue
        tid = normalize_chunk_id(str(raw_id))
        ids.add(tid)
        if not table_lookup:
            continue
        table = table_lookup.get(tid) or table_lookup.get(raw_id)
        if not table:
            continue
        canonical = normalize_chunk_id(table.get("table_id") or tid)
        ids.add(canonical)
        for source_id in table.get("source_table_ids") or []:
            ids.add(normalize_chunk_id(source_id))
            ids.add(normalize_chunk_id(f"{source_id}_merged"))
    return ids


def select_balanced_questions(
    questions: list[dict[str, Any]],
    n: int,
    *,
    types: tuple[str, ...] = QUESTION_TYPES,
) -> list[dict[str, Any]]:
    """Pick up to *n* questions with every requested type represented."""
    if n <= 0 or n >= len(questions):
        return list(questions)

    by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in types}
    for row in questions:
        qtype = row.get("question_type")
        if qtype in by_type:
            by_type[qtype].append(row)

    missing = [t for t in types if not by_type[t]]
    if missing:
        raise RuntimeError(f"Dataset missing questions for types: {', '.join(missing)}")

    base, remainder = divmod(n, len(types))
    counts = {t: base + (1 if idx < remainder else 0) for idx, t in enumerate(types)}
    selected: list[dict[str, Any]] = []
    for qtype in types:
        selected.extend(by_type[qtype][: counts[qtype]])
    return selected


def matches_ground_truth(hit: dict[str, Any], row: dict[str, Any]) -> bool:
    gt = normalize_chunk_id(row["ground_truth_chunk_id"])
    cid = normalize_chunk_id(hit.get("candidate_id") or hit.get("chunk_id") or "")
    if row.get("question_type") == "table":
        gt_ids = {gt}
        for tid in row.get("expected_table_ids") or []:
            gt_ids.add(normalize_chunk_id(tid))
        meta = hit.get("metadata") or {}
        refs = [normalize_chunk_id(r) for r in meta.get("table_refs") or []]
        return cid in gt_ids or bool(gt_ids & set(refs))
    return cid == gt

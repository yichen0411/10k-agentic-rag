#!/usr/bin/env python3
"""Generate table-only golden eval questions from MSFT FY2025 indexed tables."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REDO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from golden_eval_utils import load_text_chunks_from_db, normalize_chunk_id, parse_json_response
from text_vector_rag_inference import DEFAULT_CHAT_MODEL, call_anthropic, load_env_file, load_table_lookup, table_markdown

DEFAULT_WORKSPACE = ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
DEFAULT_DB = DEFAULT_WORKSPACE / "index" / "vectors.db"
DEFAULT_ASSETS = DEFAULT_WORKSPACE / "assets.json"
DEFAULT_TABLE_DB = DEFAULT_WORKSPACE / "index" / "table_vectors.db"
DEFAULT_EXISTING = REDO_ROOT / "common" / "msft_fy2025_golden_eval_50.json"
DEFAULT_OUTPUT = REDO_ROOT / "common" / "msft_fy2025_golden_eval_table20.json"


def load_table_chunks_from_db(table_db_path: Path) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(table_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT chunk_id, content, metadata_json FROM chunks").fetchall()
    conn.close()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = json.loads(row["metadata_json"])
        table_id = meta.get("table_id") or normalize_chunk_id(row["chunk_id"])
        out[table_id] = {
            "chunk_id": row["chunk_id"],
            "short_id": table_id,
            "content": row["content"],
            "metadata": meta,
        }
    return out


def used_table_ids(existing_path: Path) -> set[str]:
    if not existing_path.exists():
        return set()
    payload = json.loads(existing_path.read_text(encoding="utf-8"))
    used: set[str] = set()
    for row in payload.get("questions", []):
        if row.get("question_type") != "table":
            continue
        used.add(normalize_chunk_id(row.get("ground_truth_chunk_id") or ""))
        for tid in row.get("expected_table_ids") or []:
            used.add(normalize_chunk_id(tid))
    return used


def resolve_table_asset(table_lookup: dict[str, dict[str, Any]], table_id: str) -> dict[str, Any] | None:
    if table_id in table_lookup:
        return table_lookup[table_id]
    for table in table_lookup.values():
        if table_id in (table.get("source_table_ids") or []):
            return table
    return None


def table_section(table: dict[str, Any]) -> str:
    section = table.get("section_ref") or {}
    subsection = table.get("subsection_ref") or {}
    parts = [section.get("section_title"), *(subsection.get("path") or [])]
    header = " > ".join(part for part in parts if part)
    return header.split(" > ")[0] if header else "unknown"


def candidate_tables(
    table_lookup: dict[str, dict[str, Any]],
    table_vector_lookup: dict[str, dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    exclude: set[str],
) -> list[dict[str, Any]]:
    seen_canonical: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for table_id, table in table_lookup.items():
        canonical = normalize_chunk_id(table.get("table_id") or table_id)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        if canonical in exclude:
            continue
        for source_id in table.get("source_table_ids") or []:
            if normalize_chunk_id(source_id) in exclude or normalize_chunk_id(f"{source_id}_merged") in exclude:
                continue
        vector_row = table_vector_lookup.get(table_id) or table_vector_lookup.get(canonical)
        markdown = table_markdown(table)
        content = (vector_row or {}).get("content") or markdown or (table.get("vlm_parse") or {}).get("summary") or ""
        if len(content.split()) < 15:
            continue
        source_chunk = None
        for chunk in chunk_lookup.values():
            refs = {normalize_chunk_id(r) for r in chunk.get("table_refs") or []}
            if canonical in refs or table_id in refs:
                source_chunk = chunk["short_id"]
                break
        candidates.append(
            {
                "table_id": canonical,
                "table": table,
                "content": content[:6000],
                "markdown": markdown[:4000],
                "section": table_section(table),
                "header": " > ".join(
                    part
                    for part in [
                        (table.get("section_ref") or {}).get("section_title"),
                        *(((table.get("subsection_ref") or {}).get("path")) or []),
                    ]
                    if part
                ),
                "source_chunk": source_chunk,
            }
        )
    return candidates


def stratified_pick(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_section: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        by_section.setdefault(item["section"], []).append(item)
    for pool in by_section.values():
        rng.shuffle(pool)
    sections = sorted(by_section, key=lambda s: len(by_section[s]), reverse=True)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(picked) < count and sections:
        progressed = False
        for section in list(sections):
            pool = [c for c in by_section[section] if c["table_id"] not in seen]
            if not pool:
                continue
            item = pool[0]
            picked.append(item)
            seen.add(item["table_id"])
            progressed = True
            if len(picked) >= count:
                break
        if not progressed:
            break
    if len(picked) < count:
        rest = [c for c in candidates if c["table_id"] not in seen]
        rng.shuffle(rest)
        picked.extend(rest[: count - len(picked)])
    return picked[:count]


def generate_table_qa(
    item: dict[str, Any],
    *,
    anthropic_key: str,
    chat_model: str,
) -> dict[str, Any]:
    prompt = f"""Generate one table-grounded evaluation question for a financial 10-K RAG benchmark.

Return JSON only:
{{
  "question": "...",
  "expected_answer": "...",
  "gold_answer_numeric": 123.45
}}

Rules:
- Ask exactly ONE question requiring a specific value from the table (amount, percentage, count, or named line item).
- Use natural language; do not copy long phrases from the table header.
- expected_answer must include the exact figure(s) from the table (with units like million/billion/% as shown).
- gold_answer_numeric: parse the primary numeric answer as a float (e.g. $32,488 million -> 32488000000; 21% -> 21.0; negative values stay negative).
- Prefer fiscal year 2025 column when the table has multiple years.
- Do not ask compound multi-part questions.

Table id: {item["table_id"]}
Header: {item["header"]}
Table:
{item["markdown"][:3500]}
"""
    response = call_anthropic(
        prompt,
        api_key=anthropic_key,
        model=chat_model,
        system="You generate high-quality table-grounded RAG evaluation datasets. Return JSON only.",
        max_tokens=400,
    )
    data = parse_json_response(response)
    numeric = data.get("gold_answer_numeric")
    try:
        numeric = float(numeric) if numeric is not None else None
    except (TypeError, ValueError):
        numeric = None
    return {
        "question_type": "table",
        "ground_truth_chunk_id": item["table_id"],
        "ground_truth_text_chunk_id": item["source_chunk"],
        "ground_truth_chunk_content": item["content"][:6000],
        "question": str(data.get("question", "")).strip(),
        "expected_answer": str(data.get("expected_answer", "")).strip(),
        "paraphrase_of_chunk": False,
        "gold_answer_numeric": numeric,
        "expected_table_ids": [item["table_id"]],
    }


def build_dataset(
    db_path: Path,
    assets_path: Path,
    table_db_path: Path,
    existing_path: Path,
    *,
    count: int,
    seed: int,
    chat_model: str,
) -> dict[str, Any]:
    load_env_file()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    chunks = load_text_chunks_from_db(db_path)
    chunk_lookup = {c["short_id"]: c for c in chunks}
    table_lookup = load_table_lookup(assets_path)
    table_vector_lookup = load_table_chunks_from_db(table_db_path)
    exclude = used_table_ids(existing_path)
    pool = candidate_tables(table_lookup, table_vector_lookup, chunk_lookup, exclude)
    if len(pool) < count:
        raise RuntimeError(f"Only {len(pool)} candidate tables after exclusions; need {count}.")
    sample = stratified_pick(pool, count, seed)
    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(sample, 1):
        print(f"generating table question {idx}/{len(sample)}: {item['table_id']}", flush=True)
        row = generate_table_qa(item, anthropic_key=anthropic_key, chat_model=chat_model)
        row["question_id"] = f"tq{idx:03d}"
        questions.append(row)
    return {
        "source_file": "MSFT_FY2025_10-K.pdf",
        "file_id": "1779921176-msft-fy2025-10-k-8d505c867d",
        "description": f"{count}-question MSFT FY2025 table-only golden eval set.",
        "generation": {
            "seed": seed,
            "table_n": count,
            "text_db_path": str(db_path),
            "assets_path": str(assets_path),
            "table_db_path": str(table_db_path),
            "excluded_from": str(existing_path),
            "chat_model": chat_model,
        },
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MSFT FY2025 table-only golden eval questions.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--table-db", type=Path, default=DEFAULT_TABLE_DB)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3025)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args()

    payload = build_dataset(
        args.db,
        args.assets,
        args.table_db,
        args.existing,
        count=args.count,
        seed=args.seed,
        chat_model=args.chat_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n": len(payload["questions"]), "output": str(args.output)}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

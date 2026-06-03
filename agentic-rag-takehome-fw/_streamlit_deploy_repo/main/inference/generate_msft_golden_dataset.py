#!/usr/bin/env python3
"""Generate MSFT FY2025 golden evaluation dataset from indexed SQLite chunks."""

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
MAIN_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from golden_eval_utils import (
    is_good_text_candidate,
    keyword_signals,
    load_text_chunks_from_db,
    normalize_chunk_id,
    parse_json_response,
)
from text_vector_rag_inference import DEFAULT_CHAT_MODEL, call_anthropic, load_env_file, load_table_lookup, table_markdown

DEFAULT_WORKSPACE = ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
DEFAULT_DB = DEFAULT_WORKSPACE / "index" / "vectors.db"
DEFAULT_ASSETS = DEFAULT_WORKSPACE / "assets.json"
DEFAULT_TABLE_DB = DEFAULT_WORKSPACE / "index" / "table_vectors.db"
DEFAULT_TABLE_QUESTIONS = MAIN_ROOT / "common" / "msft_fy2025_parsed_table_test_questions.json"
DEFAULT_OUTPUT = MAIN_ROOT / "common" / "msft_fy2025_golden_eval_50.json"


def stratified_sample(chunks: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_section: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        if not is_good_text_candidate(chunk):
            continue
        by_section.setdefault(chunk.get("section") or "unknown", []).append(chunk)
    sections = sorted(by_section)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(picked) < count and sections:
        progressed = False
        for section in list(sections):
            pool = [c for c in by_section[section] if c["short_id"] not in seen]
            if not pool:
                continue
            chunk = rng.choice(pool)
            picked.append(chunk)
            seen.add(chunk["short_id"])
            progressed = True
            if len(picked) >= count:
                break
        if not progressed:
            break
    if len(picked) < count:
        rest = [c for c in chunks if is_good_text_candidate(c) and c["short_id"] not in seen]
        rng.shuffle(rest)
        picked.extend(rest[: count - len(picked)])
    return picked[:count]


def generate_text_questions_batch(
    chunks: list[dict[str, Any]],
    *,
    question_type: str,
    anthropic_key: str,
    chat_model: str,
) -> list[dict[str, Any]]:
    blocks = []
    for chunk in chunks:
        header = " > ".join(chunk.get("header_path") or [])
        body = (chunk.get("content") or "")[:2200]
        signals = keyword_signals(body)
        blocks.append(
            f"""Chunk {chunk['short_id']}
Section header: {header}
Keyword hints: years={signals['years']}, numbers={signals['numbers'][:5]}, phrases={signals['phrases'][:5]}
Text:
{body}
"""
        )
    joined = "\n---\n".join(blocks)
    if question_type == "text_semantic":
        style_rules = """
- Write questions whose wording is clearly different from the chunk text (avoid copying phrases longer than 3 words).
- Each question must be answerable only from its chunk.
- expected_answer should be concise (1-3 sentences) and grounded in the chunk.
- Set paraphrase_of_chunk to true.
"""
    else:
        style_rules = """
- Write keyword-heavy questions that include specific numbers, fiscal years, percentages, or proper nouns from the chunk.
- expected_answer must include the exact figures/terms needed to verify correctness.
- Set paraphrase_of_chunk to false.
"""
    prompt = f"""Generate {len(chunks)} evaluation questions for a financial 10-K RAG benchmark.

Return JSON only as an array:
[
  {{
    "chunk_id": "text_00001",
    "question": "...",
    "expected_answer": "...",
    "paraphrase_of_chunk": true
  }}
]

Rules:
{style_rules}
- Ask exactly ONE question. Never combine two asks (avoid ", and what/how...", or "X and Y?" requiring two separate facts).
- Use chunk_id exactly as given (short id like text_00001).
- One question per chunk.

Chunks:
{joined}
"""
    response = call_anthropic(
        prompt,
        api_key=anthropic_key,
        model=chat_model,
        system="You generate high-quality RAG evaluation datasets. Return JSON only.",
        max_tokens=1800,
    )
    rows = parse_json_response(response)
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected JSON array from generator, got: {type(rows)}")
    by_id = {chunk["short_id"]: chunk for chunk in chunks}
    out: list[dict[str, Any]] = []
    for row in rows:
        cid = normalize_chunk_id(str(row.get("chunk_id", "")))
        chunk = by_id.get(cid)
        if not chunk:
            continue
        out.append(
            {
                "question_type": question_type,
                "ground_truth_chunk_id": cid,
                "ground_truth_chunk_content": chunk["content"][:4000],
                "question": str(row.get("question", "")).strip(),
                "expected_answer": str(row.get("expected_answer", "")).strip(),
                "paraphrase_of_chunk": bool(row.get("paraphrase_of_chunk", question_type == "text_semantic")),
            }
        )
    return out


def load_table_chunks_from_db(table_db_path: Path) -> dict[str, dict[str, Any]]:
    if not table_db_path.exists():
        return {}
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


def resolve_table_asset(table_lookup: dict[str, dict[str, Any]], table_id: str) -> dict[str, Any] | None:
    if table_id in table_lookup:
        return table_lookup[table_id]
    for table in table_lookup.values():
        if table_id in (table.get("source_table_ids") or []):
            return table
    return None


def convert_table_questions(
    table_questions_path: Path,
    assets_path: Path,
    table_db_path: Path,
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = json.loads(table_questions_path.read_text(encoding="utf-8"))
    table_lookup = load_table_lookup(assets_path)
    table_vector_lookup = load_table_chunks_from_db(table_db_path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("questions", []):
        table_id = (item.get("expected_table_ids") or [None])[0]
        if not table_id:
            continue
        vector_row = table_vector_lookup.get(table_id)
        table = resolve_table_asset(table_lookup, table_id)
        markdown = table_markdown(table) if table else ""
        vlm = (table or {}).get("vlm_parse") or {}
        content = (
            (vector_row or {}).get("content")
            or markdown
            or vlm.get("summary")
            or (table or {}).get("raw_text")
            or ""
        )
        if not content.strip():
            continue
        source_chunk = None
        for chunk in chunk_lookup.values():
            refs = chunk.get("table_refs") or []
            if table_id in refs or (table and table.get("table_id") in refs):
                source_chunk = chunk["short_id"]
                break
        rows.append(
            {
                "question_type": "table",
                "ground_truth_chunk_id": table_id,
                "ground_truth_text_chunk_id": source_chunk,
                "ground_truth_chunk_content": content[:6000],
                "question": item["question"],
                "expected_answer": item.get("gold_answer") or "",
                "paraphrase_of_chunk": False,
                "gold_answer_numeric": item.get("gold_answer_numeric"),
                "expected_table_ids": item.get("expected_table_ids") or [table_id],
            }
        )
    return rows


def build_dataset(
    db_path: Path,
    assets_path: Path,
    table_db_path: Path,
    table_questions_path: Path,
    *,
    semantic_n: int,
    keyword_n: int,
    table_n: int,
    seed: int,
    batch_size: int,
    chat_model: str,
) -> dict[str, Any]:
    load_env_file()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for question generation.")

    chunks = load_text_chunks_from_db(db_path)
    chunk_lookup = {c["short_id"]: c for c in chunks}
    sample = stratified_sample(chunks, semantic_n + keyword_n, seed=seed)
    semantic_chunks = sample[:semantic_n]
    keyword_chunks = sample[semantic_n : semantic_n + keyword_n]

    text_rows: list[dict[str, Any]] = []
    for label, pool in [("text_semantic", semantic_chunks), ("text_keyword", keyword_chunks)]:
        for start in range(0, len(pool), batch_size):
            batch = pool[start : start + batch_size]
            print(f"generating {label} batch {start // batch_size + 1} ({len(batch)} chunks)", flush=True)
            text_rows.extend(
                generate_text_questions_batch(
                    batch,
                    question_type=label,
                    anthropic_key=anthropic_key,
                    chat_model=chat_model,
                )
            )

    table_rows = convert_table_questions(table_questions_path, assets_path, table_db_path, chunk_lookup)[:table_n]
    if len(text_rows) < semantic_n + keyword_n:
        raise RuntimeError(f"Generated only {len(text_rows)} text questions; expected {semantic_n + keyword_n}.")
    if len(table_rows) < table_n:
        raise RuntimeError(f"Only {len(table_rows)} table questions available; expected {table_n}.")

    questions: list[dict[str, Any]] = []
    seq = 1
    for row in text_rows[: semantic_n + keyword_n]:
        questions.append(
            {
                "question_id": f"q{seq:03d}",
                **row,
            }
        )
        seq += 1
    for row in table_rows[:table_n]:
        questions.append(
            {
                "question_id": f"q{seq:03d}",
                **row,
            }
        )
        seq += 1

    return {
        "source_file": "MSFT_FY2025_10-K.pdf",
        "file_id": "1779921176-msft-fy2025-10-k-8d505c867d",
        "description": "50-question MSFT FY2025 golden eval set: 20 semantic text + 20 keyword text + 10 table.",
        "generation": {
            "seed": seed,
            "semantic_n": semantic_n,
            "keyword_n": keyword_n,
            "table_n": table_n,
            "text_db_path": str(db_path),
            "assets_path": str(assets_path),
            "chat_model": chat_model,
        },
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MSFT FY2025 golden evaluation dataset.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--table-db", type=Path, default=DEFAULT_TABLE_DB)
    parser.add_argument("--table-questions", type=Path, default=DEFAULT_TABLE_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--semantic-n", type=int, default=20)
    parser.add_argument("--keyword-n", type=int, default=20)
    parser.add_argument("--table-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args()

    payload = build_dataset(
        args.db,
        args.assets,
        args.table_db,
        args.table_questions,
        semantic_n=args.semantic_n,
        keyword_n=args.keyword_n,
        table_n=args.table_n,
        seed=args.seed,
        batch_size=args.batch_size,
        chat_model=args.chat_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n": len(payload["questions"]), "output": str(args.output)}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

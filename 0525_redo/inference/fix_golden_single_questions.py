#!/usr/bin/env python3
"""Rewrite compound golden-dataset questions into single-focus questions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from golden_eval_utils import load_text_chunks_from_db, normalize_chunk_id, parse_json_response
from text_vector_rag_inference import DEFAULT_CHAT_MODEL, call_anthropic, load_env_file

DEFAULT_DATASET = ROOT / "0525_redo" / "common" / "msft_fy2025_golden_eval_50.json"
DEFAULT_DB = (
    ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d" / "index" / "vectors.db"
)

COMPOUND_RE = re.compile(
    r",\s*and\s+(what|how|which|who|when|where|why|by how much|what was|what percent|what percentage|on what)",
    re.I,
)
WHO_AND_WHAT_RE = re.compile(r"\bwho\b.+\band what\b", re.I)


def is_compound_question(question: str) -> bool:
    if COMPOUND_RE.search(question):
        return True
    if WHO_AND_WHAT_RE.search(question):
        return True
    if re.search(r", and what was her\b", question, re.I):
        return True
    return False


def rewrite_question(row: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]], *, api_key: str, chat_model: str) -> dict[str, str]:
    gt_id = normalize_chunk_id(row["ground_truth_chunk_id"])
    chunk = chunk_lookup.get(gt_id)
    if not chunk and row.get("question_type") == "table":
        chunk = {"content": row.get("ground_truth_chunk_content") or "", "header_path": []}
    if not chunk:
        raise RuntimeError(f"Missing chunk for {row['question_id']} gt={gt_id}")

    header = " > ".join(chunk.get("header_path") or [])
    body = (chunk.get("content") or row.get("ground_truth_chunk_content") or "")[:2800]
    prompt = f"""Rewrite this evaluation question into ONE single-focus question.

Rules:
- Ask exactly one thing. No compound questions.
- Do NOT use patterns like ", and what...", ", and how...", "X and Y" asking two facts.
- The new question must be fully answerable from the provided chunk only.
- Keep question_type intent: {"paraphrase wording" if row.get("paraphrase_of_chunk") else "include concrete keywords/numbers from chunk where natural"}.
- expected_answer must answer only that one question, grounded in the chunk (1-2 sentences).

Return JSON only:
{{
  "question": "...",
  "expected_answer": "..."
}}

Original question:
{row["question"]}

Original expected_answer:
{row.get("expected_answer", "")}

Chunk id: {gt_id}
Header: {header}
Chunk:
{body}
"""
    response = call_anthropic(
        prompt,
        api_key=api_key,
        model=chat_model,
        system="You rewrite RAG eval questions to be single-focus. Return JSON only.",
        max_tokens=400,
    )
    data = parse_json_response(response)
    question = str(data.get("question", "")).strip()
    answer = str(data.get("expected_answer", "")).strip()
    if not question or not answer:
        raise RuntimeError(f"Empty rewrite for {row['question_id']}")
    if is_compound_question(question):
        raise RuntimeError(f"Rewrite still compound for {row['question_id']}: {question}")
    return {"question": question, "expected_answer": answer}


def fix_dataset(dataset_path: Path, db_path: Path, *, chat_model: str, dry_run: bool) -> dict[str, Any]:
    load_env_file()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    chunks = load_text_chunks_from_db(db_path)
    chunk_lookup = {c["short_id"]: c for c in chunks}

    changed: list[dict[str, Any]] = []
    for row in payload["questions"]:
        if not is_compound_question(row["question"]):
            continue
        print(f"fix {row['question_id']}: {row['question'][:90]}...", flush=True)
        rewrite = rewrite_question(row, chunk_lookup, api_key=api_key, chat_model=chat_model)
        changed.append(
            {
                "question_id": row["question_id"],
                "old_question": row["question"],
                "new_question": rewrite["question"],
                "old_expected_answer": row.get("expected_answer"),
                "new_expected_answer": rewrite["expected_answer"],
            }
        )
        if not dry_run:
            row["question"] = rewrite["question"]
            row["expected_answer"] = rewrite["expected_answer"]

    if not dry_run:
        meta = payload.setdefault("generation", {})
        meta["single_question_fix"] = True
        meta["single_question_fix_count"] = len(changed)
        dataset_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"changed": len(changed), "items": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix compound questions in golden dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = fix_dataset(args.dataset, args.db, chat_model=args.chat_model, dry_run=args.dry_run)
    print(json.dumps({"changed": result["changed"]}, indent=2))


if __name__ == "__main__":
    main()

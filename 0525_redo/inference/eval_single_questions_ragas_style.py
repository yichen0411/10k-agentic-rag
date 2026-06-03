#!/usr/bin/env python3
"""Run RAG inference on sampled single-chunk questions and score RAGAS-style metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REDO_ROOT = Path(__file__).resolve().parents[1]
CHUNKING_DIR = REDO_ROOT / "chunking"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_vector_rag_inference import (
    DEFAULT_ASSETS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_DB,
    call_anthropic,
    load_env_file,
    run_pipeline,
    table_markdown,
)

DEFAULT_QUESTIONS = CHUNKING_DIR / "AAPL_FY2025_text_vector_hit_eval_100_combined.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "AAPL_FY2025_mixed_35_ragas_style_eval.json"
JUDGE_CHUNK_TEXT_MAX = 800
JUDGE_TABLE_TEXT_MAX = 1200
JUDGE_TOTAL_CONTEXT_MAX = 14000


def load_chunk_lookup(db_path: Path) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT chunk_id, content, metadata_json FROM chunks").fetchall()
    conn.close()
    return {
        row["chunk_id"]: {
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"]),
        }
        for row in rows
    }


def load_assets(assets_path: Path) -> dict[str, Any]:
    return json.loads(assets_path.read_text(encoding="utf-8"))


def load_table_lookup(assets_path: Path) -> dict[str, dict[str, Any]]:
    assets = load_assets(assets_path)
    return {table["table_id"]: table for table in assets.get("tables", []) if table.get("table_id")}


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _table_body(table: dict[str, Any], table_lookup: dict[str, dict[str, Any]] | None) -> str:
    if table.get("markdown"):
        return table["markdown"]
    if table.get("raw_text"):
        return table["raw_text"]
    table_id = table.get("table_id")
    if table_lookup and table_id in table_lookup:
        full = table_lookup[table_id]
        return table_markdown(full) or full.get("raw_text") or table_to_reference_text(full)
    return table.get("summary") or ""


def format_judge_contexts(
    contexts: list[dict[str, Any]],
    table_contexts: list[dict[str, Any]],
    *,
    chunk_lookup: dict[str, str] | None = None,
    table_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    blocks: list[str] = []
    budget = JUDGE_TOTAL_CONTEXT_MAX

    for idx, ctx in enumerate(contexts, 1):
        header = " > ".join(ctx.get("header_path") or [])
        chunk_id = ctx.get("chunk_id") or ""
        body = ctx.get("content") or (chunk_lookup or {}).get(chunk_id) or ""
        if body:
            body = _truncate(body, JUDGE_CHUNK_TEXT_MAX)
            block = f"Context {idx}: {chunk_id}\nHeader: {header}\nText:\n{body}"
        else:
            block = f"Context {idx}: {chunk_id}\nHeader: {header}"
        if len(block) > budget:
            block = _truncate(block, budget)
        if not block.strip():
            continue
        blocks.append(block)
        budget -= len(block) + 2
        if budget <= 0:
            break

    for idx, table in enumerate(table_contexts, 1):
        if budget <= 0:
            break
        header = " > ".join(table.get("header_path") or [])
        table_id = table.get("table_id") or ""
        body = _truncate(_table_body(table, table_lookup), JUDGE_TABLE_TEXT_MAX)
        if body:
            block = f"Table Context {idx}: {table_id}\nHeader: {header}\nText:\n{body}"
        else:
            block = f"Table Context {idx}: {table_id}\nHeader: {header}"
        if len(block) > budget:
            block = _truncate(block, budget)
        blocks.append(block)
        budget -= len(block) + 2

    return "\n\n".join(blocks)


def judge_ragas_style(
    question: str,
    answer: str,
    contexts: list[dict[str, Any]],
    table_contexts: list[dict[str, Any]],
    reference_id: str,
    reference_header: str,
    reference_text: str,
    anthropic_key: str,
    chat_model: str,
    *,
    chunk_lookup: dict[str, str] | None = None,
    table_lookup: dict[str, dict[str, Any]] | None = None,
    question_type: str | None = None,
    expected_answer: str | None = None,
    gold_answer_numeric: float | int | None = None,
) -> dict[str, Any]:
    joined_contexts = format_judge_contexts(
        contexts,
        table_contexts,
        chunk_lookup=chunk_lookup,
        table_lookup=table_lookup,
    )
    table_scoring = ""
    if question_type == "table":
        gold_num = ""
        if gold_answer_numeric is not None:
            gold_num = f"\nGold numeric reference: {gold_answer_numeric}"
        table_scoring = f"""
This is a TABLE question. For reference_coverage and faithfulness, prioritize numeric correctness.
Expected answer: {expected_answer or ""}{gold_num}

Treat as equivalent when comparing numbers: different units ($47M vs 47,000,000), rounding ($281.7B vs $281,724M),
commas/currency symbols, and parentheses for negatives. Penalize wrong metric, year, segment, sign, or order-of-magnitude errors.
"""
    prompt = f"""Evaluate this RAG answer using RAGAS-style criteria.
{table_scoring}
Question:
{question}

Answer:
{answer}

Retrieved context (text chunks and tables sent to the answer model; per-item text may be truncated):
{joined_contexts}

Gold/source reference:
id: {reference_id}
header: {reference_header}
text:
{reference_text[:2600]}

Return JSON only with:
{{
  "faithfulness": 1-5,
  "answer_relevancy": 1-5,
  "context_precision": 1-5,
  "reference_coverage": 1-5,
  "verdict": "pass" | "borderline" | "fail",
  "notes": "short explanation"
}}

Scoring:
- faithfulness: answer is supported by retrieved context, no hallucination.
- answer_relevancy: answer directly addresses the question.
- context_precision: retrieved/expanded context is mostly relevant, not noisy.
- reference_coverage: answer captures the key information from the gold/source chunk needed for the question.
"""
    response = call_anthropic(
        prompt,
        api_key=anthropic_key,
        model=chat_model,
        system="You are a strict RAG evaluator. Return JSON only.",
        max_tokens=500,
    )
    data = parse_json_response(response)
    return {
        "faithfulness": float(data.get("faithfulness", 0)),
        "answer_relevancy": float(data.get("answer_relevancy", 0)),
        "context_precision": float(data.get("context_precision", 0)),
        "reference_coverage": float(data.get("reference_coverage", 0)),
        "verdict": data.get("verdict", ""),
        "notes": data.get("notes", ""),
    }


def table_to_reference_text(table: dict[str, Any]) -> str:
    raw_text = table.get("raw_text") or ""
    if raw_text:
        return raw_text
    rows = table.get("raw_rows") or []
    return "\n".join(" | ".join(cell or "" for cell in row) for row in rows)


def generate_table_questions(
    assets_path: Path,
    chunk_lookup: dict[str, dict[str, Any]],
    count: int,
    seed: int,
    anthropic_key: str,
    chat_model: str,
) -> list[dict[str, Any]]:
    table_lookup = load_table_lookup(assets_path)
    candidate_tables: list[dict[str, Any]] = []
    seen = set()
    for chunk_id, chunk in chunk_lookup.items():
        for table_id in chunk["metadata"].get("table_refs", []) or []:
            table = table_lookup.get(table_id)
            if not table or table_id in seen:
                continue
            text = table_to_reference_text(table)
            if len(text.split()) < 12:
                continue
            candidate_tables.append({"table_id": table_id, "source_chunk_id": chunk_id, "table": table, "text": text})
            seen.add(table_id)
    rng = random.Random(seed)
    sample = rng.sample(candidate_tables, min(count, len(candidate_tables)))

    questions = []
    for idx, item in enumerate(sample, 1):
        table = item["table"]
        section = table.get("section_ref") or {}
        subsection = table.get("subsection_ref") or {}
        header = " > ".join(part for part in [section.get("section_title"), *(subsection.get("path") or [])] if part)
        prompt = f"""Generate one question answerable from this table.

Rules:
- The question should require a table value or comparison from the table.
- It should be natural language.
- Return JSON only: {{"question": "..."}}

Table id: {item["table_id"]}
Header: {header}
Table:
{item["text"][:2500]}
"""
        try:
            response = call_anthropic(
                prompt,
                api_key=anthropic_key,
                model=chat_model,
                system="You generate table-grounded evaluation questions. Return JSON only.",
                max_tokens=180,
            )
            data = parse_json_response(response)
            question = str(data.get("question", "")).strip()
        except Exception:
            question = f"According to the table in {header}, what information and key values does the table report?"
        questions.append(
            {
                "question_type": "table",
                "question": question,
                "target_table_id": item["table_id"],
                "target_chunk_id": item["source_chunk_id"],
                "reference_header": header,
                "reference_text": item["text"],
            }
        )
        print(f"generated table question {idx}/{len(sample)}: {item['table_id']}", flush=True)
    return questions


def evaluate(
    questions_path: Path,
    db_path: Path,
    assets_path: Path,
    output_path: Path,
    n: int,
    table_n: int,
    seed: int,
    chat_model: str,
) -> dict[str, Any]:
    load_env_file()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    source = json.loads(questions_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    chunk_lookup = load_chunk_lookup(db_path)
    table_lookup = load_table_lookup(assets_path)
    content_lookup = {cid: item["content"] for cid, item in chunk_lookup.items()}
    text_sample = rng.sample(source["results"], min(n, len(source["results"])))
    text_questions = [
        {
            "question_type": "text",
            "question": row["question"],
            "target_chunk_id": row["target_chunk_id"],
            "reference_header": " > ".join(chunk_lookup[row["target_chunk_id"]]["metadata"].get("header_path") or []),
            "reference_text": chunk_lookup[row["target_chunk_id"]]["content"],
        }
        for row in text_sample
    ]
    table_questions = generate_table_questions(assets_path, chunk_lookup, table_n, seed + 999, anthropic_key, chat_model)
    sample = text_questions + table_questions

    results = []
    for idx, row in enumerate(sample, 1):
        question = row["question"]
        target_id = row.get("target_chunk_id")
        target_table_id = row.get("target_table_id")
        print(f"[{idx}/{len(sample)}] {row['question_type']} {target_id or target_table_id}", flush=True)
        inference = run_pipeline(
            question,
            db_path=db_path,
            vector_top_k=10,
            rerank_top_n=3,
            max_context_chunks=14,
            chat_model=chat_model,
            embed_model="nomic-ai/nomic-embed-text-v1.5",
            assets_path=assets_path,
        )
        expanded_ids = {ctx["chunk_id"] for ctx in inference["expanded_context"]}
        reranked_ids = {ctx["chunk_id"] for ctx in inference["reranked_top"]}
        vector_ids = {ctx["chunk_id"] for ctx in inference["vector_hits"]}
        table_ids = {table["table_id"] for table in inference.get("table_contexts", [])}
        judge = judge_ragas_style(
            question=question,
            answer=inference["answer"],
            contexts=inference["expanded_context"],
            table_contexts=inference.get("table_contexts", []),
            reference_id=target_table_id or target_id,
            reference_header=row["reference_header"],
            reference_text=row["reference_text"],
            anthropic_key=anthropic_key,
            chat_model=chat_model,
            chunk_lookup=content_lookup,
            table_lookup=table_lookup,
        )
        result = {
            "question_type": row["question_type"],
            "question": question,
            "target_chunk_id": target_id,
            "target_table_id": target_table_id,
            "target_in_vector_top10": target_id in vector_ids if target_id else None,
            "target_in_rerank_top3": target_id in reranked_ids if target_id else None,
            "target_in_expanded_context": target_id in expanded_ids if target_id else None,
            "target_table_in_context": target_table_id in table_ids if target_table_id else None,
            "latency": inference.get("latency", {}),
            "answer": inference["answer"],
            "reranked_top": inference["reranked_top"],
            "expanded_context": inference["expanded_context"],
            "table_contexts": inference.get("table_contexts", []),
            **judge,
        }
        results.append(result)
        partial = {"summary": summarize(results, seed, db_path, chat_model), "results": results, "partial": len(results) < len(sample)}
        output_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {"summary": summarize(results, seed, db_path, chat_model), "results": results}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def summarize(results: list[dict[str, Any]], seed: int, db_path: Path, chat_model: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": len(results),
        "seed": seed,
        "db_path": str(db_path),
        "chat_model": chat_model,
    }
    if not results:
        return summary
    for key in ["target_in_vector_top10", "target_in_rerank_top3", "target_in_expanded_context"]:
        eligible = [r for r in results if r.get(key) is not None]
        summary[key] = sum(1 for r in eligible if r[key]) / len(eligible) if eligible else None
    eligible_tables = [r for r in results if r.get("target_table_in_context") is not None]
    summary["target_table_in_context"] = (
        sum(1 for r in eligible_tables if r["target_table_in_context"]) / len(eligible_tables) if eligible_tables else None
    )
    for key in ["faithfulness", "answer_relevancy", "context_precision", "reference_coverage"]:
        summary[f"mean_{key}"] = sum(float(r[key]) for r in results) / len(results)
    latencies = [r.get("latency", {}).get("total_sec") for r in results if r.get("latency", {}).get("total_sec") is not None]
    if latencies:
        summary["mean_latency_sec"] = sum(latencies) / len(latencies)
        summary["max_latency_sec"] = max(latencies)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_type.setdefault(row.get("question_type", "unknown"), []).append(row)
    summary["by_type"] = {}
    for question_type, rows in by_type.items():
        summary["by_type"][question_type] = {
            "n": len(rows),
            "mean_faithfulness": sum(float(r["faithfulness"]) for r in rows) / len(rows),
            "mean_answer_relevancy": sum(float(r["answer_relevancy"]) for r in rows) / len(rows),
            "mean_reference_coverage": sum(float(r["reference_coverage"]) for r in rows) / len(rows),
            "mean_latency_sec": sum(r.get("latency", {}).get("total_sec", 0) for r in rows) / len(rows),
        }
    summary["pass_rate"] = sum(1 for r in results if r.get("verdict") == "pass") / len(results)
    summary["borderline_rate"] = sum(1 for r in results if r.get("verdict") == "borderline") / len(results)
    summary["fail_rate"] = sum(1 for r in results if r.get("verdict") == "fail") / len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference + RAGAS-style eval on sampled single-chunk questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--table-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args()

    payload = evaluate(args.questions, args.db, args.assets, args.output, args.n, args.table_n, args.seed, args.chat_model)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

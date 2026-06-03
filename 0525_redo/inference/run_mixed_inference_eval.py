#!/usr/bin/env python3
"""Run mixed table/text inference eval and write latency-rich results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REDO_ROOT = Path(__file__).resolve().parents[1]
CHUNKING_DIR = REDO_ROOT / "chunking"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CHUNKING_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNKING_DIR))

from text_vector_rag_inference import (
    DEFAULT_EMBED_MODEL,
    load_env_file,
    resolve_answer_model,
    resolve_rerank_model,
    run_pipeline,
)

DEFAULT_QUESTIONS = REDO_ROOT / "common" / "msft_fy2025_mixed_15_inference_test.json"
DEFAULT_WORKSPACE = ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "msft_fy2025_mixed_15_inference_results.json"
DEFAULT_LOG = Path(__file__).resolve().parent / "msft_fy2025_mixed_15_inference_results.jsonl"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def extract_nums(text: str) -> list[float]:
    s = (text or "").lower().replace(",", "")
    out: list[float] = []
    for m in re.finditer(r"\(?-?\d+(?:\.\d+)?\)?%?", s):
        t = m.group()
        neg = t.startswith("(") and t.endswith(")")
        t = t.strip("()%")
        if t:
            out.append(-float(t) if neg else float(t))
    return out


def numeric_match(pred: str, gold_num: float | None, rel_tol: float = 0.02) -> bool | None:
    if gold_num is None:
        return None
    pred_nums = extract_nums(pred)
    candidates = list(pred_nums)
    if "million" in pred.lower():
        candidates.extend(n * 1_000_000 for n in pred_nums if abs(n) < 1e7)
    if "billion" in pred.lower():
        candidates.extend(n * 1_000_000_000 for n in pred_nums if abs(n) < 1e5)
    for pn in candidates:
        if gold_num == 0 and abs(pn) < 1e-6:
            return True
        if gold_num != 0 and abs(pn - gold_num) / abs(gold_num) <= rel_tol:
            return True
    return False


def text_match(pred: str, gold: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    tokens = [tok for tok in re.findall(r"[a-z0-9][a-z0-9-]+", g) if len(tok) > 3]
    if not tokens:
        return g in p
    hits = sum(1 for tok in tokens if tok in p)
    return hits / len(tokens) >= 0.55


def short_answer(answer: str, limit: int = 220) -> str:
    answer = re.sub(r"\s+", " ", (answer or "").strip())
    if len(answer) <= limit:
        return answer
    return answer[: limit - 3] + "..."


def build_pipeline_path(inference: dict[str, Any]) -> dict[str, Any]:
    text_rerank = inference.get("text_reranked_top") or inference.get("reranked_top") or []
    return {
        "text_vector_top10": [
            {
                "rank": hit.get("rank"),
                "candidate_id": hit.get("candidate_id"),
                "score": hit.get("score"),
                "header_path": hit.get("header_path"),
            }
            for hit in inference.get("vector_hits", [])
        ],
        "text_bm25_top10": [
            {
                "rank": hit.get("rank"),
                "candidate_id": hit.get("candidate_id"),
                "score": hit.get("score"),
                "header_path": hit.get("header_path"),
            }
            for hit in inference.get("bm25_hits", [])
        ],
        "text_retrieval_merged": [
            {
                "rank": hit.get("rank"),
                "candidate_id": hit.get("candidate_id"),
                "vector_score": hit.get("vector_score"),
                "bm25_score": hit.get("bm25_score"),
                "retrieval_sources": hit.get("retrieval_sources") or [],
                "header_path": hit.get("header_path"),
            }
            for hit in inference.get("text_retrieval_hits", [])
        ],
        "table_vector_top5": [
            {
                "rank": hit.get("rank"),
                "table_id": hit.get("table_id"),
                "score": hit.get("score"),
                "passed_threshold": hit.get("passed_threshold"),
                "header_path": hit.get("header_path"),
                "section_ref": hit.get("section_ref"),
                "summary": hit.get("summary"),
            }
            for hit in inference.get("table_vector_hits", [])
        ],
        "table_vector_filtered": [
            {
                "rank": hit.get("rank"),
                "table_id": hit.get("table_id"),
                "score": hit.get("score"),
                "header_path": hit.get("header_path"),
                "section_ref": hit.get("section_ref"),
                "summary": hit.get("summary"),
            }
            for hit in inference.get("table_vector_hits_filtered", [])
        ],
        "text_rerank_top3": [
            {
                "candidate_id": hit.get("candidate_id"),
                "vector_rank": hit.get("vector_rank") or hit.get("rank"),
                "header_path": hit.get("header_path"),
                "rerank_reason": hit.get("rerank_reason", ""),
            }
            for hit in text_rerank
        ],
        "expanded_text_chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "context_role": chunk.get("context_role"),
                "table_refs": chunk.get("table_refs") or [],
                "header_path": chunk.get("header_path") or [],
            }
            for chunk in inference.get("expanded_context", [])
        ],
        "table_contexts": [
            {
                "table_id": table.get("table_id"),
                "source_kind": table.get("source_kind"),
                "page_start": table.get("page_start"),
                "header_path": table.get("header_path") or [],
                "section_path": table.get("section_path"),
                "summary": table.get("summary"),
                "has_markdown": table.get("has_markdown"),
            }
            for table in inference.get("table_contexts", [])
        ],
    }


def evaluate_row(row: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    qtype = row.get("question_type", "table")
    answer = inference.get("answer") or ""
    expected_tables = set(row.get("expected_table_ids") or [])
    expected_chunks = set(row.get("expected_chunk_ids") or [])

    table_vec_ids = {hit.get("table_id") for hit in inference.get("table_vector_hits", [])}
    filtered_ids = {hit.get("table_id") for hit in inference.get("table_vector_hits_filtered", [])}
    ctx_table_ids = {t.get("table_id") for t in inference.get("table_contexts", [])}
    expanded_ids = {c.get("chunk_id") for c in inference.get("expanded_context", [])}
    rerank_ids = {hit.get("chunk_id") or hit.get("candidate_id") for hit in inference.get("text_reranked_top", [])}

    if qtype == "table":
        correct = numeric_match(answer, row.get("gold_answer_numeric"))
        if correct is False:
            correct = text_match(answer, row.get("gold_answer", ""))
    else:
        correct = text_match(answer, row.get("gold_answer", ""))

    latency = inference.get("latency") or {}
    return {
        "id": row["id"],
        "question_type": qtype,
        "question": row["question"],
        "gold_answer": row.get("gold_answer"),
        "answer_short": short_answer(answer),
        "answer_correct": correct,
        "expected_table_ids": sorted(expected_tables),
        "expected_chunk_ids": sorted(expected_chunks),
        "table_vector_hit": bool(expected_tables & table_vec_ids) if expected_tables else None,
        "table_threshold_pass": bool(expected_tables & filtered_ids) if expected_tables else None,
        "table_in_context": bool(expected_tables & ctx_table_ids) if expected_tables else None,
        "text_rerank_hit": bool(expected_chunks & rerank_ids) if expected_chunks else None,
        "text_in_expanded_context": bool(expected_chunks & expanded_ids) if expected_chunks else None,
        "table_context_ids": [t.get("table_id") for t in inference.get("table_contexts", [])],
        "text_rerank_ids": list(rerank_ids),
        "pipeline_path": build_pipeline_path(inference),
        "latency_sec": latency,
        "latency_total_sec": latency.get("total_sec"),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def bucket(qtype: str) -> list[dict[str, Any]]:
        return [r for r in results if r.get("question_type") == qtype]

    def avg_latency(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [r.get("latency_sec", {}).get(key) for r in rows if r.get("latency_sec", {}).get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    overall_lat = [r.get("latency_total_sec") for r in results if r.get("latency_total_sec") is not None]
    table_rows = bucket("table")
    text_rows = bucket("text")
    table_vec = [r for r in table_rows if r.get("table_vector_hit") is not None]
    table_ctx = [r for r in table_rows if r.get("table_in_context") is not None]
    return {
        "n": len(results),
        "n_table": len(table_rows),
        "n_text": len(text_rows),
        "answer_correct_rate": sum(1 for r in results if r.get("answer_correct")) / len(results) if results else None,
        "table_answer_correct_rate": sum(1 for r in table_rows if r.get("answer_correct")) / len(table_rows) if table_rows else None,
        "text_answer_correct_rate": sum(1 for r in text_rows if r.get("answer_correct")) / len(text_rows) if text_rows else None,
        "table_vector_hit_rate": sum(1 for r in table_vec if r["table_vector_hit"]) / len(table_vec) if table_vec else None,
        "table_in_context_rate": sum(1 for r in table_ctx if r["table_in_context"]) / len(table_ctx) if table_ctx else None,
        "latency_sec": {
            "overall_mean_total": round(sum(overall_lat) / len(overall_lat), 3) if overall_lat else None,
            "table_mean_total": avg_latency(table_rows, "total_sec"),
            "text_mean_total": avg_latency(text_rows, "total_sec"),
            "mean_vector_search": avg_latency(results, "vector_search_sec"),
            "mean_rerank": avg_latency(results, "rerank_sec"),
            "mean_context_expansion": avg_latency(results, "context_expansion_sec"),
            "mean_answer": avg_latency(results, "answer_sec"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixed table/text inference eval.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--table-threshold", type=float, default=0.65)
    args = parser.parse_args()

    load_env_file()
    answer_model = resolve_answer_model()
    rerank_model = resolve_rerank_model()
    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    workspace = args.workspace
    db_path = workspace / "index" / "vectors.db"
    table_db_path = workspace / "index" / "table_vectors.db"
    assets_path = workspace / "assets.json"
    paths = {
        "questions_path": str(args.questions.resolve()),
        "workspace": str(workspace.resolve()),
        "text_db_path": str(db_path.resolve()),
        "table_db_path": str(table_db_path.resolve()),
        "assets_path": str(assets_path.resolve()),
        "output_path": str(args.output.resolve()),
        "log_path": str(args.log.resolve()),
    }

    if args.log.exists():
        args.log.unlink()

    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "run_start",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "paths": paths,
                    "answer_model": answer_model,
                    "rerank_model": rerank_model,
                    "embed_model": DEFAULT_EMBED_MODEL,
                    "n_questions": len(payload["questions"]),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    results: list[dict[str, Any]] = []
    n_total = len(payload["questions"])
    for idx, row in enumerate(payload["questions"], 1):
        print(f"[{idx}/{n_total}] {row['id']} ({row.get('question_type', 'table')}): {row['question'][:80]}...", flush=True)
        inference = run_pipeline(
            row["question"],
            db_path=db_path,
            vector_top_k=10,
            rerank_top_n=3,
            max_context_chunks=14,
            chat_model=answer_model,
            rerank_model=rerank_model,
            embed_model=DEFAULT_EMBED_MODEL,
            assets_path=assets_path,
            table_db_path=table_db_path,
            table_vector_top_k=5,
            table_similarity_threshold=args.table_threshold,
        )
        result = evaluate_row(row, inference)
        results.append(result)
        print(
            f"  -> correct={result['answer_correct']} total={result['latency_total_sec']}s "
            f"(vec={result['latency_sec'].get('vector_search_sec')} rerank={result['latency_sec'].get('rerank_sec')} "
            f"answer={result['latency_sec'].get('answer_sec')})",
            flush=True,
        )
        record = {"event": "question_result", "ts": datetime.now(timezone.utc).isoformat(), "paths": paths, **result}
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    final = {
        "summary": summarize(results),
        "settings": {
            "pipeline": "dual_path_text_rerank_plus_table_threshold",
            "table_similarity_threshold": args.table_threshold,
            "answer_model": answer_model,
            "rerank_model": rerank_model,
            "embed_model": DEFAULT_EMBED_MODEL,
        },
        "paths": paths,
        "results": results,
    }
    args.output.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "run_end", "ts": datetime.now(timezone.utc).isoformat(), "summary": final["summary"], "paths": paths}, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": final["summary"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()

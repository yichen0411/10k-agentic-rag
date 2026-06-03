#!/usr/bin/env python3
"""Run the MSFT parsed-table test set through text+table RAG and score retrieval hits."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
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

from text_vector_rag_inference import DEFAULT_EMBED_MODEL, load_env_file, resolve_chat_model, run_pipeline

DEFAULT_QUESTIONS = REDO_ROOT / "common" / "msft_fy2025_parsed_table_test_questions.json"
DEFAULT_WORKSPACE = ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "msft_fy2025_table_test_eval.json"
DEFAULT_LOG = Path(__file__).resolve().parent / "msft_fy2025_table_test_eval.jsonl"


def normalize_number(text: str) -> float | None:
    if text is None:
        return None
    cleaned = str(text).lower().replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = float(match.group())
    return -value if negative else value


def numeric_close(pred: str, gold: float | None, rel_tol: float = 0.02) -> bool | None:
    if gold is None:
        return None
    pred_num = normalize_number(pred)
    if pred_num is None:
        return False
    if gold == 0:
        return abs(pred_num) < 1e-6
    return abs(pred_num - gold) / abs(gold) <= rel_tol


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
        "table_vector_top5": [
            {
                "rank": hit.get("rank"),
                "table_id": hit.get("table_id"),
                "score": hit.get("score"),
                "passed_threshold": hit.get("passed_threshold"),
                "header_path": hit.get("header_path"),
            }
            for hit in inference.get("table_vector_hits", [])
        ],
        "table_vector_filtered": [
            {
                "rank": hit.get("rank"),
                "table_id": hit.get("table_id"),
                "score": hit.get("score"),
                "header_path": hit.get("header_path"),
            }
            for hit in inference.get("table_vector_hits_filtered", [])
        ],
        "text_rerank_top3": [
            {
                "candidate_id": hit.get("candidate_id"),
                "vector_rank": hit.get("vector_rank") or hit.get("rank"),
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
                "has_markdown": table.get("has_markdown"),
            }
            for table in inference.get("table_contexts", [])
        ],
    }


def diagnose_failure(
    *,
    expected_tables: set[str],
    vector_hit: bool | None,
    threshold_hit: bool | None,
    context_hit: bool | None,
    numeric_ok: bool | None,
    error: str | None,
) -> tuple[str, str]:
    if error:
        return "error", error
    if expected_tables and vector_hit is False:
        return "table_vector_miss", f"Expected table(s) {sorted(expected_tables)} not in table_vector_top5"
    if expected_tables and threshold_hit is False:
        return "table_threshold_miss", f"Expected table(s) {sorted(expected_tables)} below similarity threshold"
    if expected_tables and context_hit is False:
        return "table_context_miss", f"Expected table(s) {sorted(expected_tables)} not attached in final table_contexts"
    if numeric_ok is False:
        return "answer_wrong", "Answer numeric value did not match gold within tolerance"
    return "ok", ""


def append_log(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(
    questions_path: Path,
    workspace: Path,
    output_path: Path,
    log_path: Path,
    chat_model: str | None,
) -> dict[str, Any]:
    load_env_file()
    provider, resolved_model = resolve_chat_model()
    chat_model = chat_model or resolved_model
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    db_path = workspace / "index" / "vectors.db"
    table_db_path = workspace / "index" / "table_vectors.db"
    assets_path = workspace / "assets.json"
    paths = {
        "questions_path": str(questions_path.resolve()),
        "workspace": str(workspace.resolve()),
        "text_db_path": str(db_path.resolve()),
        "table_db_path": str(table_db_path.resolve()),
        "assets_path": str(assets_path.resolve()),
        "output_path": str(output_path.resolve()),
        "log_path": str(log_path.resolve()),
    }
    if not db_path.exists():
        raise FileNotFoundError(f"Missing text vector index: {db_path}. Re-process the file with embeddings enabled.")
    if not table_db_path.exists():
        raise FileNotFoundError(
            f"Missing table vector index: {table_db_path}. Run build_table_vector_db.py and copy to table_vectors.db."
        )

    if log_path.exists():
        log_path.unlink()

    run_meta = {
        "event": "run_start",
        "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "paths": paths,
        "chat_provider": provider,
        "chat_model": chat_model,
        "embed_model": DEFAULT_EMBED_MODEL,
    }
    append_log(log_path, run_meta)

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(payload["questions"], 1):
        question = row["question"]
        expected_tables = set(row.get("expected_table_ids") or [])
        started = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        print(f"[{idx}/{len(payload['questions'])}] {row['id']}: {question}", flush=True)
        record: dict[str, Any] = {
            "event": "question_result",
            "ts": started,
            "idx": idx,
            "id": row["id"],
            "question": question,
            "paths": paths,
            "expected_table_ids": sorted(expected_tables),
            "gold_answer": row.get("gold_answer"),
        }
        try:
            inference = run_pipeline(
                question,
                db_path=db_path,
                vector_top_k=10,
                rerank_top_n=3,
                max_context_chunks=14,
                chat_model=chat_model,
                embed_model=DEFAULT_EMBED_MODEL,
                assets_path=assets_path,
                table_db_path=table_db_path,
                table_vector_top_k=5,
            )
            table_ids = {table["table_id"] for table in inference.get("table_contexts", [])}
            table_hits = {table["table_id"] for table in inference.get("table_vector_hits", [])}
            filtered_hits = {table["table_id"] for table in inference.get("table_vector_hits_filtered", [])}
            vector_hit = bool(expected_tables & table_hits) if expected_tables else None
            threshold_hit = bool(expected_tables & filtered_hits) if expected_tables else None
            context_hit = bool(expected_tables & table_ids) if expected_tables else None
            numeric_ok = numeric_close(inference.get("answer", ""), row.get("gold_answer_numeric"))
            failure_stage, failure_reason = diagnose_failure(
                expected_tables=expected_tables,
                vector_hit=vector_hit,
                threshold_hit=threshold_hit,
                context_hit=context_hit,
                numeric_ok=numeric_ok,
                error=None,
            )
            pipeline_path = build_pipeline_path(inference)
            result = {
                "id": row["id"],
                "question": question,
                "gold_answer": row.get("gold_answer"),
                "expected_table_ids": sorted(expected_tables),
                "answer": inference.get("answer"),
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "expected_table_in_table_vector_top5": vector_hit,
                "expected_table_in_threshold_pass": threshold_hit,
                "expected_table_in_table_context": context_hit,
                "numeric_match": numeric_ok,
                "pipeline_path": pipeline_path,
                "latency": inference.get("latency", {}),
                "settings": inference.get("settings", {}),
            }
            results.append(result)
            record.update(result)
            record["status"] = "ok" if failure_stage == "ok" else failure_stage
            print(
                f"  -> {failure_stage} | latency={inference.get('latency', {}).get('total_sec')}s | "
                f"table_vec={vector_hit} threshold={threshold_hit} ctx={context_hit} numeric={numeric_ok}",
                flush=True,
            )
        except Exception as exc:
            failure_stage, failure_reason = diagnose_failure(
                expected_tables=expected_tables,
                vector_hit=None,
                threshold_hit=None,
                context_hit=None,
                numeric_ok=None,
                error=str(exc),
            )
            result = {
                "id": row["id"],
                "question": question,
                "gold_answer": row.get("gold_answer"),
                "expected_table_ids": sorted(expected_tables),
                "answer": None,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "traceback": traceback.format_exc(),
                "latency": {},
            }
            results.append(result)
            record.update(result)
            record["status"] = "error"
            print(f"  -> ERROR: {exc}", flush=True)
        append_log(log_path, record)
        output_path.write_text(
            json.dumps({"summary": summarize(results), "paths": paths, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    final = {
        "summary": summarize(results),
        "paths": paths,
        "chat_provider": provider,
        "chat_model": chat_model,
        "questions_path": str(questions_path),
        "workspace": str(workspace),
        "output_path": str(output_path),
        "log_path": str(log_path),
        "results": results,
    }
    output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(
        log_path,
        {
            "event": "run_end",
            "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "summary": final["summary"],
            "output_path": str(output_path.resolve()),
        },
    )
    return final


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    ctx_hits = [r for r in results if r.get("expected_table_in_table_context") is not None]
    vec_hits = [r for r in results if r.get("expected_table_in_table_vector_top5") is not None]
    threshold_hits = [r for r in results if r.get("expected_table_in_threshold_pass") is not None]
    numeric = [r for r in results if r.get("numeric_match") is not None]
    failures: dict[str, int] = {}
    for row in results:
        stage = row.get("failure_stage") or "unknown"
        failures[stage] = failures.get(stage, 0) + 1
    latencies = [r.get("latency", {}).get("total_sec") for r in results if r.get("latency", {}).get("total_sec") is not None]
    return {
        "n": len(results),
        "table_context_hit_rate": sum(1 for r in ctx_hits if r["expected_table_in_table_context"]) / len(ctx_hits) if ctx_hits else None,
        "table_vector_hit_rate": sum(1 for r in vec_hits if r["expected_table_in_table_vector_top5"]) / len(vec_hits) if vec_hits else None,
        "table_threshold_pass_rate": sum(1 for r in threshold_hits if r["expected_table_in_threshold_pass"]) / len(threshold_hits) if threshold_hits else None,
        "numeric_match_rate": sum(1 for r in numeric if r["numeric_match"]) / len(numeric) if numeric else None,
        "failure_counts": failures,
        "latency_sec": {
            "mean_total": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max_total": round(max(latencies), 3) if latencies else None,
            "min_total": round(min(latencies), 3) if latencies else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MSFT parsed-table test questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--chat-model", default=None)
    args = parser.parse_args()
    result = evaluate(args.questions, args.workspace, args.output, args.log, args.chat_model)
    print(json.dumps({"summary": result["summary"], "paths": result["paths"], "log_path": result["log_path"]}, indent=2))


if __name__ == "__main__":
    main()

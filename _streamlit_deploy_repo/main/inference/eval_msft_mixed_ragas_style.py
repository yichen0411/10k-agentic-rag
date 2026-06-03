#!/usr/bin/env python3
"""RAGAS-style eval for MSFT mixed table/text questions with split rerank/answer models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_single_questions_ragas_style import judge_ragas_style, summarize as summarize_ragas
from text_vector_rag_inference import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    assemble_dual_path_context,
    filter_table_hits_by_threshold,
    load_chunks,
    load_env_file,
    load_table_lookup,
    rerank_text_chunks,
    answer_query,
    resolve_answer_model,
    resolve_rerank_model,
    run_pipeline,
    table_markdown,
)

DEFAULT_QUESTIONS = MAIN_ROOT / "common" / "msft_fy2025_mixed_15_inference_test.json"
DEFAULT_WORKSPACE = ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
DEFAULT_REPLAY_LOG = Path(__file__).resolve().parent / "msft_fy2025_mixed_15_inference_results.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "msft_fy2025_mixed_5_ragas_style_eval.json"
DEFAULT_JUDGE_MODEL = DEFAULT_CHAT_MODEL


def normalize_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("::")[-1] if "::" in chunk_id else chunk_id


def chunk_lookup_by_short_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        out[chunk["chunk_id"]] = chunk
        out[normalize_chunk_id(chunk["chunk_id"])] = chunk
    return out


def reference_for_question(question: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]], table_lookup: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    gold = question.get("gold_answer") or ""
    if question.get("expected_table_ids"):
        table_id = question["expected_table_ids"][0]
        table = table_lookup.get(table_id, {})
        section = table.get("section_ref") or {}
        subsection = table.get("subsection_ref") or {}
        header = " > ".join(part for part in [section.get("section_title"), *(subsection.get("path") or [])] if part)
        markdown = table_markdown(table)
        text = markdown or table.get("raw_text") or gold
        return table_id, header, f"Gold answer: {gold}\n\nTable ({table_id}):\n{text}"

    if question.get("expected_chunk_ids"):
        chunk_id = question["expected_chunk_ids"][0]
        chunk = chunk_lookup.get(chunk_id) or chunk_lookup.get(f"source.pdf::{chunk_id}")
        if chunk:
            header = " > ".join(chunk["metadata"].get("header_path") or [])
            return chunk["chunk_id"], header, f"Gold answer: {gold}\n\n{chunk['content']}"
        return chunk_id, "", f"Gold answer: {gold}"

    return question["id"], "", gold


def rebuild_text_hits(old_row: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for hit in old_row["pipeline_path"]["text_vector_top10"]:
        cid = hit["candidate_id"]
        chunk = chunk_lookup.get(cid)
        hits.append(
            {
                "rank": hit["rank"],
                "candidate_id": cid,
                "chunk_id": cid,
                "score": hit["score"],
                "header_path": hit.get("header_path") or [],
                "content": chunk["content"] if chunk else "",
                "section": chunk.get("section") if chunk else None,
                "metadata": chunk.get("metadata", {}) if chunk else {},
                "candidate_type": "text",
            }
        )
    return hits


def rebuild_table_hits(old_row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": hit["rank"],
            "table_id": hit["table_id"],
            "score": hit["score"],
            "header_path": hit.get("header_path") or [],
            "summary": hit.get("summary") or "",
            "candidate_type": "table",
        }
        for hit in old_row["pipeline_path"]["table_vector_top5"]
    ]


def run_replay_inference(
    question: str,
    old_row: dict[str, Any],
    *,
    chunks: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    answer_model: str,
    rerank_model: str,
    table_threshold: float,
) -> dict[str, Any]:
    text_hits = rebuild_text_hits(old_row, chunk_lookup)
    table_hits = rebuild_table_hits(old_row)
    filtered_table_hits = filter_table_hits_by_threshold(table_hits, table_threshold)

    timings: dict[str, float] = {"load_sec": 0.0}
    timings["vector_search_sec"] = old_row["latency_sec"].get("vector_search_sec", 0)

    t0 = time.perf_counter()
    text_anchors = rerank_text_chunks(
        question,
        text_hits,
        rerank_model=rerank_model,
        top_n=3,
        fireworks_key=os.environ.get("FIREWORKS_API_KEY"),
    )
    timings["rerank_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    expanded, table_contexts = assemble_dual_path_context(
        text_anchors,
        filtered_table_hits,
        chunks,
        table_lookup,
        max_context_chunks=14,
    )
    timings["context_expansion_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    answer = answer_query(question, expanded, table_contexts, anthropic_key="", chat_model=answer_model)
    timings["answer_sec"] = round(time.perf_counter() - t0, 3)
    timings["total_sec"] = round(sum(timings.values()), 3)

    reranked_top = [
        {
            "chunk_id": hit.get("chunk_id") or hit.get("candidate_id"),
            "candidate_id": hit.get("candidate_id"),
            "header_path": hit.get("header_path") or [],
            "rerank_score": hit.get("rerank_score"),
            "rerank_reason": hit.get("rerank_reason", ""),
        }
        for hit in text_anchors
    ]
    return {
        "answer": answer,
        "vector_hits": text_hits,
        "table_vector_hits": table_hits,
        "table_vector_hits_filtered": filtered_table_hits,
        "reranked_top": reranked_top,
        "expanded_context": expanded,
        "table_contexts": table_contexts,
        "latency": timings,
        "settings": {
            "pipeline": "replay_cached_vector_hits",
            "answer_model": answer_model,
            "rerank_model": rerank_model,
            "table_similarity_threshold": table_threshold,
        },
    }


def evaluate(
    questions_path: Path,
    workspace: Path,
    output_path: Path,
    *,
    limit: int,
    replay_log: Path | None,
    answer_model: str,
    rerank_model: str,
    judge_model: str,
    table_threshold: float,
) -> dict[str, Any]:
    load_env_file()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload["questions"][:limit]
    db_path = workspace / "index" / "vectors.db"
    table_db_path = workspace / "index" / "table_vectors.db"
    assets_path = workspace / "assets.json"

    chunks = load_chunks(db_path)
    chunk_lookup = chunk_lookup_by_short_id(chunks)
    table_lookup = load_table_lookup(assets_path)

    replay_rows: dict[str, dict[str, Any]] = {}
    if replay_log and replay_log.exists():
        for line in replay_log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event") == "question_result":
                replay_rows[row["id"]] = row

    results: list[dict[str, Any]] = []
    for idx, question in enumerate(questions, 1):
        qid = question["id"]
        print(f"[{idx}/{len(questions)}] {qid} ({question.get('question_type', 'table')})", flush=True)
        if replay_log:
            old_row = replay_rows.get(qid)
            if not old_row:
                raise RuntimeError(f"Missing replay row for {qid} in {replay_log}")
            inference = run_replay_inference(
                question["question"],
                old_row,
                chunks=chunks,
                table_lookup=table_lookup,
                chunk_lookup=chunk_lookup,
                answer_model=answer_model,
                rerank_model=rerank_model,
                table_threshold=table_threshold,
            )
        else:
            inference = run_pipeline(
                question["question"],
                db_path=db_path,
                vector_top_k=10,
                rerank_top_n=3,
                max_context_chunks=14,
                chat_model=answer_model,
                embed_model=DEFAULT_EMBED_MODEL,
                assets_path=assets_path,
                table_db_path=table_db_path,
                table_vector_top_k=5,
                table_similarity_threshold=table_threshold,
                rerank_model=rerank_model,
            )

        reference_id, reference_header, reference_text = reference_for_question(question, chunk_lookup, table_lookup)
        expanded_ids = {ctx["chunk_id"] for ctx in inference["expanded_context"]}
        reranked_ids = {ctx["chunk_id"] for ctx in inference["reranked_top"]}
        vector_ids = {ctx["candidate_id"] for ctx in inference["vector_hits"]}
        table_ids = {table["table_id"] for table in inference.get("table_contexts", [])}
        expected_chunks = {normalize_chunk_id(cid) for cid in (question.get("expected_chunk_ids") or [])}
        expected_tables = set(question.get("expected_table_ids") or [])

        judge = judge_ragas_style(
            question=question["question"],
            answer=inference["answer"],
            contexts=inference["expanded_context"],
            table_contexts=inference.get("table_contexts", []),
            reference_id=reference_id,
            reference_header=reference_header,
            reference_text=reference_text,
            anthropic_key=anthropic_key,
            chat_model=judge_model,
            chunk_lookup={cid: chunk["content"] for cid, chunk in chunk_lookup.items()},
            table_lookup=table_lookup,
        )
        result = {
            "id": qid,
            "question_type": question.get("question_type", "table"),
            "question": question["question"],
            "gold_answer": question.get("gold_answer"),
            "reference_id": reference_id,
            "reference_header": reference_header,
            "expected_table_ids": sorted(expected_tables),
            "expected_chunk_ids": question.get("expected_chunk_ids") or [],
            "target_in_vector_top10": bool(expected_chunks & {normalize_chunk_id(cid) for cid in vector_ids}) if expected_chunks else None,
            "target_in_rerank_top3": bool(expected_chunks & {normalize_chunk_id(cid) for cid in reranked_ids}) if expected_chunks else None,
            "target_in_expanded_context": bool(expected_chunks & {normalize_chunk_id(cid) for cid in expanded_ids}) if expected_chunks else None,
            "target_table_in_context": bool(expected_tables & table_ids) if expected_tables else None,
            "table_vector_hit": bool(expected_tables & {hit.get("table_id") for hit in inference.get("table_vector_hits", [])}) if expected_tables else None,
            "latency": inference.get("latency", {}),
            "answer": inference["answer"],
            "reranked_top": inference["reranked_top"],
            "expanded_context": inference["expanded_context"],
            "table_contexts": inference.get("table_contexts", []),
            "settings": inference.get("settings", {}),
            **judge,
        }
        results.append(result)
        print(
            f"  verdict={judge.get('verdict')} faith={judge.get('faithfulness')} rel={judge.get('answer_relevancy')} "
            f"total={result['latency'].get('total_sec')}s rerank={result['latency'].get('rerank_sec')}s answer={result['latency'].get('answer_sec')}s",
            flush=True,
        )
        partial = {
            "summary": build_summary(results, workspace, answer_model, rerank_model, judge_model, replay_log),
            "results": results,
            "partial": len(results) < len(questions),
        }
        output_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    final = {
        "summary": build_summary(results, workspace, answer_model, rerank_model, judge_model, replay_log),
        "results": results,
    }
    output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    return final


def build_summary(
    results: list[dict[str, Any]],
    workspace: Path,
    answer_model: str,
    rerank_model: str,
    judge_model: str,
    replay_log: Path | None,
) -> dict[str, Any]:
    base = summarize_ragas(results, seed=0, db_path=workspace / "index" / "vectors.db", chat_model=answer_model)
    base["workspace"] = str(workspace)
    base["answer_model"] = answer_model
    base["rerank_model"] = rerank_model
    base["judge_model"] = judge_model
    base["replay_log"] = str(replay_log) if replay_log else None
    if results:
        base["mean_rerank_sec"] = sum(r.get("latency", {}).get("rerank_sec", 0) for r in results) / len(results)
        base["mean_answer_sec"] = sum(r.get("latency", {}).get("answer_sec", 0) for r in results) / len(results)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="MSFT mixed RAGAS-style eval with Haiku rerank + Sonnet answer.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-log", type=Path, default=DEFAULT_REPLAY_LOG)
    parser.add_argument("--no-replay", action="store_true", help="Run full pipeline (requires Fireworks embeddings).")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--answer-model", default="")
    parser.add_argument("--rerank-model", default="")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--table-threshold", type=float, default=0.65)
    args = parser.parse_args()

    load_env_file()
    answer_model = args.answer_model or resolve_answer_model()
    rerank_model = args.rerank_model or resolve_rerank_model()
    replay_log = None if args.no_replay else args.replay_log

    payload = evaluate(
        args.questions,
        args.workspace,
        args.output,
        limit=args.limit,
        replay_log=replay_log,
        answer_model=answer_model,
        rerank_model=rerank_model,
        judge_model=args.judge_model,
        table_threshold=args.table_threshold,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

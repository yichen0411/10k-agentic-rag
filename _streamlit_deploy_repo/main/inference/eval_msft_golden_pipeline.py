#!/usr/bin/env python3
"""Three-layer MSFT golden dataset evaluation with LangSmith logging."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = Path(__file__).resolve().parent
CHUNKING_DIR = MAIN_ROOT / "chunking"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
if str(CHUNKING_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNKING_DIR))

from bm25_retrieval import bm25_search, merge_vector_and_bm25_hits
from build_text_vector_db import DEFAULT_EMBED_MODEL, embed_texts_fireworks
from eval_single_questions_ragas_style import judge_ragas_style
from golden_eval_utils import (
    RECALL_KS_DEFAULT,
    matches_ground_truth,
    normalize_chunk_id,
    parse_json_response,
    rank_position,
    recall_at_k_dict,
    select_balanced_questions,
    select_table_questions,
    select_text_questions,
    table_ground_truth_ids,
)
from langsmith_eval_logging import upload_eval_results_to_langsmith
from text_vector_rag_inference import (
    DEFAULT_CHAT_MODEL,
    call_anthropic,
    load_chunks,
    load_env_file,
    load_table_lookup,
    rerank_text_chunks,
    resolve_answer_model,
    resolve_rerank_model,
    run_pipeline,
    vector_search_table_summaries_with_embedding,
    vector_search_with_embedding,
)

DEFAULT_DATASET = MAIN_ROOT / "common" / "msft_fy2025_golden_eval_50.json"
DEFAULT_WORKSPACE = ROOT / "data" / "chunk_studio" / "1779921176-msft-fy2025-10-k-8d505c867d"
DEFAULT_OUTPUT = INFERENCE_DIR / "msft_fy2025_golden_eval_results.json"
DEFAULT_JUDGE_MODEL = DEFAULT_CHAT_MODEL
RECALL_KS = RECALL_KS_DEFAULT


def judge_llm_score(
    row: dict[str, Any],
    answer: str,
    *,
    anthropic_key: str,
    chat_model: str,
) -> dict[str, Any]:
    qtype = row.get("question_type", "text")
    if qtype == "table":
        gold_numeric = row.get("gold_answer_numeric")
        gold_numeric_note = f"\nGold numeric reference (if helpful): {gold_numeric}" if gold_numeric is not None else ""
        criteria = f"""
You are judging a TABLE question answer for a financial 10-K RAG system.
Focus on whether the core numeric answer is correct.

Scoring (llm_judge_score 1-5):
- 5: The key number(s) or percentage match the expected answer (allow equivalent units/formatting).
- 4: Correct number with minor omission (e.g., missing "million" label but value is right).
- 3: Close but materially off (wrong row/year/segment, or >1% numeric error after unit normalization).
- 1-2: Wrong value, wrong sign, or hallucinated figure.

Treat these as EQUIVALENT (do not penalize):
- $47,000,000 vs $47 million vs 47M vs 47000000
- $281.7 billion vs $281,724 million vs 281724 (when context is millions)
- 21% vs 21 percent; $(4,901) million vs -$4.901 billion vs -4901 (millions)
- Rounding: $281.724 billion vs $281.7 billion; $13.64 vs $13.6 if clearly the same figure
- Commas, currency symbols, parentheses for negatives

Still penalize: wrong fiscal year, wrong segment/metric, wrong sign, off by orders of magnitude.
Ignore extra explanation if the requested number is correct.{gold_numeric_note}
"""
    elif qtype == "text_keyword":
        criteria = """
You are judging a KEYWORD-HEAVY text question.
- Score 5 if the answer includes the exact figures, names, or years required by expected_answer.
- Penalize missing critical numbers or misquoted proper nouns.
"""
    else:
        criteria = """
You are judging a SEMANTIC text question where wording differs from the source chunk.
- Score 5 if the answer captures the same facts as expected_answer even with different phrasing.
- Penalize hallucinations or missing the core fact.
"""
    prompt = f"""Score the RAG answer from 1 to 5.

Question:
{row['question']}

Expected answer (gold):
{row['expected_answer']}

Model answer:
{answer}

{criteria}

Return JSON only:
{{
  "llm_judge_score": 1-5,
  "verdict": "pass" | "borderline" | "fail",
  "notes": "short explanation"
}}
"""
    response = call_anthropic(
        prompt,
        api_key=anthropic_key,
        model=chat_model,
        system="You are a strict financial RAG evaluator. Return JSON only.",
        max_tokens=400,
    )
    data = parse_json_response(response)
    return {
        "llm_judge_score": float(data.get("llm_judge_score", 0)),
        "verdict": data.get("verdict", ""),
        "notes": data.get("notes", ""),
    }


def evaluate_retrieval_layer(
    row: dict[str, Any],
    *,
    query: str,
    chunks: list[dict[str, Any]],
    query_embedding: list[float],
    table_chunks: list[dict[str, Any]],
    vector_top_k: int,
    bm25_top_k: int,
    table_lookup: dict[str, dict[str, Any]] | None = None,
    recall_ks: tuple[int, ...] = RECALL_KS,
) -> dict[str, Any]:
    matcher: Callable[[dict[str, Any]], bool] = lambda hit: matches_ground_truth(hit, row)
    eval_top_k = max(*recall_ks, vector_top_k, bm25_top_k)

    vector_hits = vector_search_with_embedding(query_embedding, chunks, top_k=eval_top_k)
    bm25_hits = bm25_search(query, chunks, top_k=eval_top_k)
    merged_hits = merge_vector_and_bm25_hits(vector_hits, bm25_hits)

    per_path: dict[str, Any] = {}
    for name, hits in {"vector": vector_hits, "bm25": bm25_hits}.items():
        rank = rank_position(hits, matcher)
        per_path[name] = {
            "recall_at_k": recall_at_k_dict(rank, recall_ks),
            "rank": rank,
            "retrieved_k": len(hits),
            "top_ids": [normalize_chunk_id(h.get("candidate_id", "")) for h in hits[: max(recall_ks)]],
        }

    merged_rank = rank_position(merged_hits, matcher)
    per_path["merged"] = {
        "recall_at_k": recall_at_k_dict(merged_rank, recall_ks),
        "rank": merged_rank,
        "pool_size": len(merged_hits),
        "pool_hit": merged_rank is not None,
        "pool_ids": [normalize_chunk_id(h.get("candidate_id", "")) for h in merged_hits],
    }

    fusion_gain_at_k: dict[str, float] = {}
    for k in recall_ks:
        key = str(k)
        merged_hit = per_path["merged"]["recall_at_k"][key]
        best_single = max(
            float(per_path["vector"]["recall_at_k"][key]),
            float(per_path["bm25"]["recall_at_k"][key]),
        )
        fusion_gain_at_k[key] = float(merged_hit) - best_single

    table_vector_rank = None
    table_vector_recall: dict[str, bool] | None = None
    if row.get("question_type") == "table":
        table_hits = vector_search_table_summaries_with_embedding(
            query_embedding, table_chunks, top_k=eval_top_k
        )
        gt_table_ids = table_ground_truth_ids(row, table_lookup)

        def table_match(hit: dict[str, Any]) -> bool:
            meta = hit.get("metadata") or {}
            tid = normalize_chunk_id(meta.get("table_id") or hit.get("candidate_id") or "")
            return tid in gt_table_ids

        table_vector_rank = rank_position(table_hits, table_match)
        table_vector_recall = recall_at_k_dict(table_vector_rank, recall_ks)

    return {
        "eval_top_k": eval_top_k,
        "recall_ks": list(recall_ks),
        "per_path": per_path,
        "fusion_gain_at_k": fusion_gain_at_k,
        "fusion_gain_pool": fusion_gain_at_k.get(str(max(recall_ks)), 0.0),
        "table_vector": {
            "rank": table_vector_rank,
            "recall_at_k": table_vector_recall,
            "retrieved_k": eval_top_k if table_vector_recall else None,
        },
        "merged_pool_size": len(merged_hits),
        "merged_hits": merged_hits,
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
    }


def evaluate_rerank_layer(
    row: dict[str, Any],
    *,
    query: str,
    merged_hits: list[dict[str, Any]],
    rerank_model: str,
    rerank_top_n: int,
    fireworks_key: str,
) -> dict[str, Any]:
    matcher: Callable[[dict[str, Any]], bool] = lambda hit: matches_ground_truth(hit, row)
    pre_rank = rank_position(merged_hits, matcher)
    reranked = rerank_text_chunks(
        query,
        merged_hits,
        rerank_model=rerank_model,
        top_n=rerank_top_n,
        fireworks_key=fireworks_key,
    )
    post_rank = rank_position(reranked, matcher)
    in_merged = pre_rank is not None
    in_rerank_top3 = post_rank is not None and post_rank <= rerank_top_n
    kicked_out = bool(in_merged and not in_rerank_top3)

    table_ref_chunks = 0
    for hit in reranked[:rerank_top_n]:
        meta = hit.get("metadata") or {}
        if meta.get("table_refs"):
            table_ref_chunks += 1
    pure_text_chunks = rerank_top_n - table_ref_chunks

    position_improvement = None
    if pre_rank is not None and post_rank is not None:
        position_improvement = pre_rank - post_rank

    return {
        "pre_rerank_rank": pre_rank,
        "post_rerank_rank": post_rank,
        "position_improvement": position_improvement,
        "kicked_out_by_rerank": kicked_out,
        "in_merged_pool": in_merged,
        "in_rerank_top3": in_rerank_top3,
        "rerank_top3_table_ref_chunks": table_ref_chunks,
        "rerank_top3_pure_text_chunks": pure_text_chunks,
        "rerank_top3_table_ref_ratio": round(table_ref_chunks / rerank_top_n, 3) if rerank_top_n else 0.0,
        "reranked": reranked,
    }


def ground_truth_header(row: dict[str, Any], chunk_by_id: dict[str, dict[str, Any]]) -> str:
    gt_id = row.get("ground_truth_chunk_id") or ""
    gt_chunk = chunk_by_id.get(gt_id)
    if gt_chunk:
        return " > ".join(gt_chunk.get("metadata", {}).get("header_path") or [])
    gt_text = row.get("ground_truth_chunk_content") or ""
    if " > " in gt_text:
        return gt_text.split("\n", 1)[0].strip()
    return ""


def evaluate_answer_layer(
    row: dict[str, Any],
    inference: dict[str, Any],
    *,
    anthropic_key: str,
    judge_model: str,
    chunk_lookup: dict[str, str] | None = None,
    table_lookup: dict[str, dict[str, Any]] | None = None,
    chunk_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reference_header = ground_truth_header(row, chunk_by_id or {})
    ragas = judge_ragas_style(
        question=row["question"],
        answer=inference["answer"],
        contexts=inference.get("expanded_context") or [],
        table_contexts=inference.get("table_contexts") or [],
        reference_id=row["ground_truth_chunk_id"],
        reference_header=reference_header,
        reference_text=row.get("ground_truth_chunk_content") or "",
        anthropic_key=anthropic_key,
        chat_model=judge_model,
        chunk_lookup=chunk_lookup,
        table_lookup=table_lookup,
        question_type=row.get("question_type"),
        expected_answer=row.get("expected_answer"),
        gold_answer_numeric=row.get("gold_answer_numeric"),
    )
    llm = judge_llm_score(row, inference["answer"], anthropic_key=anthropic_key, chat_model=judge_model)
    return {**ragas, **llm}


def summarize_results(results: list[dict[str, Any]], *, rerank_top_n: int = 3) -> dict[str, Any]:
    def mean_key(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
        vals = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = (cur or {}).get(key)
            if isinstance(cur, (int, float)) and cur is not None:
                vals.append(float(cur))
        return round(sum(vals) / len(vals), 4) if vals else None

    def rate(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
        vals = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = (cur or {}).get(key)
            if isinstance(cur, bool):
                vals.append(1.0 if cur else 0.0)
        return round(sum(vals) / len(vals), 4) if vals else None

    def recall_summary(rows: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float | None]:
        return {f"recall_at_{k}": rate(rows, (*path, "recall_at_k", str(k))) for k in RECALL_KS}

    def best_of_vector_bm25(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for k in RECALL_KS:
            key = str(k)
            vals: list[float] = []
            for row in rows:
                retrieval = (row.get("metrics") or {}).get("retrieval") or {}
                per_path = retrieval.get("per_path") or {}
                vector_hit = ((per_path.get("vector") or {}).get("recall_at_k") or {}).get(key)
                bm25_hit = ((per_path.get("bm25") or {}).get("recall_at_k") or {}).get(key)
                if isinstance(vector_hit, bool) or isinstance(bm25_hit, bool):
                    vals.append(1.0 if (vector_hit or bm25_hit) else 0.0)
            out[f"recall_at_{k}"] = round(sum(vals) / len(vals), 4) if vals else None
        return out

    def rerank_recall_summary(rows: list[dict[str, Any]], rerank_top_n: int) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for k in RECALL_KS:
            cap = min(k, rerank_top_n)
            vals: list[float] = []
            for row in rows:
                rank = ((row.get("metrics") or {}).get("rerank") or {}).get("post_rerank_rank")
                vals.append(1.0 if rank is not None and rank <= cap else 0.0)
            out[f"recall_at_{k}"] = round(sum(vals) / len(vals), 4) if vals else None
        return out

    def rerank_block(rows: list[dict[str, Any]], rerank_top_n: int) -> dict[str, Any]:
        return {
            "in_merged_pool_rate": rate(rows, ("metrics", "rerank", "in_merged_pool")),
            "in_rerank_top3_rate": rate(rows, ("metrics", "rerank", "in_rerank_top3")),
            "kick_out_rate": rate(rows, ("metrics", "rerank", "kicked_out_by_rerank")),
            "position_improvement_mean": mean_key(rows, ("metrics", "rerank", "position_improvement")),
            "post_rerank_recall": rerank_recall_summary(rows, rerank_top_n),
        }

    def ragas_block(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        return {
            "faithfulness": mean_key(rows, ("metrics", "judge", "faithfulness")),
            "answer_relevancy": mean_key(rows, ("metrics", "judge", "answer_relevancy")),
            "context_precision": mean_key(rows, ("metrics", "judge", "context_precision")),
            "reference_coverage": mean_key(rows, ("metrics", "judge", "reference_coverage")),
            "llm_judge_score": mean_key(rows, ("metrics", "judge", "llm_judge_score")),
        }

    summary: dict[str, Any] = {"n": len(results)}

    text_rows = [r for r in results if r.get("question_type") != "table"]
    table_rows = [r for r in results if r.get("question_type") == "table"]

    retrieval_base = ("metrics", "retrieval", "per_path")
    table_base = ("metrics", "retrieval", "table_vector")

    summary["retrieval"] = {
        "recall_ks": list(RECALL_KS),
        "eval_top_k": (results[0].get("metrics") or {}).get("retrieval", {}).get("eval_top_k") if results else None,
        "text_questions": {
            "n": len(text_rows),
            "vector": recall_summary(text_rows, (*retrieval_base, "vector")),
            "bm25": recall_summary(text_rows, (*retrieval_base, "bm25")),
            "merged": recall_summary(text_rows, (*retrieval_base, "merged")),
            "best_of_vector_bm25": best_of_vector_bm25(text_rows),
        },
        "table_questions": {
            "n": len(table_rows),
            "table_vector": recall_summary(table_rows, table_base),
        },
        "fusion_gain_at_k": {
            f"at_{k}": mean_key(results, ("metrics", "retrieval", "fusion_gain_at_k", str(k)))
            for k in RECALL_KS
        },
    }

    # Flat aliases for quick scanning / LangSmith compatibility.
    for k in RECALL_KS:
        summary[f"text_merged_recall_at_{k}"] = rate(
            text_rows, (*retrieval_base, "merged", "recall_at_k", str(k))
        )
        summary[f"table_vector_recall_at_{k}"] = rate(
            table_rows, (*table_base, "recall_at_k", str(k))
        )

    summary["rerank"] = {
        "rerank_top_n": rerank_top_n,
        "overall": rerank_block(results, rerank_top_n),
        "text_questions": rerank_block(text_rows, rerank_top_n) if text_rows else None,
    }
    summary["ragas"] = {
        "overall": ragas_block(results),
        "text_questions": ragas_block(text_rows) if text_rows else None,
    }

    # Legacy flat keys
    summary["rerank_kick_out_rate"] = summary["rerank"]["overall"]["kick_out_rate"]
    summary["rerank_position_improvement_mean"] = summary["rerank"]["overall"]["position_improvement_mean"]
    summary["mean_faithfulness"] = summary["ragas"]["overall"]["faithfulness"]
    summary["mean_answer_relevancy"] = summary["ragas"]["overall"]["answer_relevancy"]
    summary["mean_context_precision"] = summary["ragas"]["overall"]["context_precision"]
    summary["mean_reference_coverage"] = summary["ragas"]["overall"]["reference_coverage"]
    summary["mean_llm_judge_score"] = summary["ragas"]["overall"]["llm_judge_score"]

    by_type: dict[str, Any] = {}
    for qtype in ("text_semantic", "text_keyword", "table"):
        rows = [r for r in results if r.get("question_type") == qtype]
        if not rows:
            continue
        by_type[qtype] = {
            "n": len(rows),
            "retrieval": {
                "vector": recall_summary(rows, (*retrieval_base, "vector")),
                "bm25": recall_summary(rows, (*retrieval_base, "bm25")),
                "merged": recall_summary(rows, (*retrieval_base, "merged")),
                "best_of_vector_bm25": best_of_vector_bm25(rows) if qtype != "table" else None,
                "table_vector": recall_summary(rows, table_base) if qtype == "table" else None,
            },
            "rerank": rerank_block(rows, rerank_top_n),
            "ragas": ragas_block(rows),
        }
    summary["by_question_type"] = by_type

    latency_keys = (
        "load_sec",
        "vector_search_sec",
        "rerank_sec",
        "context_expansion_sec",
        "answer_sec",
        "judge_sec",
        "total_sec",
    )
    summary["latency"] = {f"mean_{key}": mean_key(results, ("latency", key)) for key in latency_keys}
    summary["latency_by_question_type"] = {
        qtype: {f"mean_{key}": mean_key(rows, ("latency", key)) for key in latency_keys}
        for qtype in ("text_semantic", "text_keyword", "table")
        if (rows := [r for r in results if r.get("question_type") == qtype])
    }
    return summary


def evaluate_dataset(
    dataset_path: Path,
    workspace: Path,
    output_path: Path,
    *,
    limit: int | None,
    balanced_limit: int | None,
    text_only: bool,
    table_only: bool,
    vector_top_k: int,
    bm25_top_k: int,
    rerank_top_n: int,
    table_threshold: float,
    judge_model: str,
    experiment_name: str,
    skip_langsmith: bool,
) -> dict[str, Any]:
    load_env_file()
    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not fireworks_key:
        raise RuntimeError("FIREWORKS_API_KEY is required.")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if balanced_limit:
        questions = select_balanced_questions(questions, balanced_limit)
    elif text_only:
        questions = select_text_questions(questions)
    elif table_only:
        questions = select_table_questions(questions)
    elif limit:
        questions = questions[:limit]

    db_path = workspace / "index" / "vectors.db"
    table_db_path = workspace / "index" / "table_vectors.db"
    assets_path = workspace / "assets.json"
    chunks = load_chunks(db_path)
    table_chunks = load_chunks(table_db_path) if table_db_path.exists() else []
    chunk_lookup = {chunk["chunk_id"]: chunk["content"] for chunk in chunks}
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    table_lookup = load_table_lookup(assets_path)
    rerank_model = resolve_rerank_model()
    answer_model = resolve_answer_model()

    results: list[dict[str, Any]] = []

    for idx, row in enumerate(questions, 1):
        qid = row["question_id"]
        query = row["question"]
        print(f"[{idx}/{len(questions)}] {qid} {row.get('question_type')}", flush=True)

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        query_embedding = embed_texts_fireworks([query], model=DEFAULT_EMBED_MODEL, api_key=fireworks_key)[0]
        embed_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        retrieval = evaluate_retrieval_layer(
            row,
            query=query,
            chunks=chunks,
            query_embedding=query_embedding,
            table_chunks=table_chunks,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            table_lookup=table_lookup,
        )
        retrieval_eval_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        rerank = evaluate_rerank_layer(
            row,
            query=query,
            merged_hits=retrieval["merged_hits"],
            rerank_model=rerank_model,
            rerank_top_n=rerank_top_n,
            fireworks_key=fireworks_key,
        )
        rerank_eval_sec = time.perf_counter() - t0

        inference = run_pipeline(
            query,
            db_path=db_path,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_n=rerank_top_n,
            max_context_chunks=14,
            chat_model=answer_model,
            embed_model=DEFAULT_EMBED_MODEL,
            assets_path=assets_path,
            table_db_path=table_db_path if table_db_path.exists() else None,
            table_similarity_threshold=table_threshold,
            rerank_model=rerank_model,
        )
        pipeline_latency = dict(inference.get("latency") or {})

        t0 = time.perf_counter()
        judge = evaluate_answer_layer(
            row,
            inference,
            anthropic_key=anthropic_key,
            judge_model=judge_model,
            chunk_lookup=chunk_lookup,
            table_lookup=table_lookup,
            chunk_by_id=chunk_by_id,
        )
        judge_sec = time.perf_counter() - t0

        total_sec = time.perf_counter() - t_total
        latency = {
            **{k: pipeline_latency.get(k, 0.0) for k in (
                "load_sec",
                "vector_search_sec",
                "rerank_sec",
                "context_expansion_sec",
                "answer_sec",
            )},
            "judge_sec": round(judge_sec, 3),
            "total_sec": round(total_sec, 3),
            "eval_embed_sec": round(embed_sec, 3),
            "eval_retrieval_sec": round(retrieval_eval_sec, 3),
            "eval_rerank_sec": round(rerank_eval_sec, 3),
        }

        expanded = inference.get("expanded_context") or []
        table_contexts = inference.get("table_contexts") or []
        result = {
            "question_id": qid,
            "question_type": row.get("question_type"),
            "question": query,
            "ground_truth_chunk_id": row.get("ground_truth_chunk_id"),
            "expected_answer": row.get("expected_answer"),
            "answer": inference.get("answer"),
            "latency": latency,
            "metrics": {
                "retrieval": {
                    "eval_top_k": retrieval["eval_top_k"],
                    "recall_ks": retrieval["recall_ks"],
                    "per_path": retrieval["per_path"],
                    "fusion_gain_at_k": retrieval["fusion_gain_at_k"],
                    "fusion_gain_pool": retrieval["fusion_gain_pool"],
                    "table_vector": retrieval["table_vector"],
                    "merged_pool_size": retrieval["merged_pool_size"],
                },
                "rerank": {k: v for k, v in rerank.items() if k != "reranked"},
                "judge": judge,
                "final_context": {
                    "text_chunks": len(expanded),
                    "table_contexts": len(table_contexts),
                    "table_to_text_ratio": round(
                        len(table_contexts) / max(len(expanded), 1), 3
                    ),
                },
            },
        }
        results.append(result)

        partial = {
            "summary": summarize_results(results, rerank_top_n=rerank_top_n),
            "results": results,
            "partial": len(results) < len(questions),
        }
        output_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize_results(results, rerank_top_n=rerank_top_n)
    output = {
        "dataset_path": str(dataset_path),
        "workspace": str(workspace),
        "settings": {
            "vector_top_k": vector_top_k,
            "bm25_top_k": bm25_top_k,
            "rerank_top_n": rerank_top_n,
            "table_threshold": table_threshold,
            "rerank_model": rerank_model,
            "answer_model": answer_model,
            "judge_model": judge_model,
            "balanced_limit": balanced_limit,
            "text_only": text_only,
            "table_only": table_only,
            "limit": limit,
        },
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if not skip_langsmith:
        try:
            output["langsmith"] = upload_eval_results_to_langsmith(
                dataset_path=dataset_path,
                results_path=output_path,
                experiment_prefix=experiment_name,
                description=f"MSFT golden eval ({len(results)} questions)",
            )
        except Exception as exc:
            output["langsmith"] = {"enabled": False, "reason": str(exc)}
    else:
        output["langsmith"] = {"enabled": False, "reason": "skipped_by_flag"}

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSFT golden dataset three-layer eval.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--balanced-limit",
        type=int,
        default=0,
        help="Sample N questions with all question types represented (overrides --limit).",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Evaluate only text_semantic + text_keyword questions (40 of 50).",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Evaluate only table questions (10 of 50).",
    )
    parser.add_argument("--vector-top-k", type=int, default=10)
    parser.add_argument("--bm25-top-k", type=int, default=10)
    parser.add_argument("--rerank-top-n", type=int, default=3)
    parser.add_argument("--table-threshold", type=float, default=0.65)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--experiment-name", default="msft-fy2025-golden-eval")
    parser.add_argument("--skip-langsmith", action="store_true")
    args = parser.parse_args()

    payload = evaluate_dataset(
        args.dataset,
        args.workspace,
        args.output,
        limit=args.limit or None,
        balanced_limit=args.balanced_limit or None,
        text_only=args.text_only,
        table_only=args.table_only,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_n=args.rerank_top_n,
        table_threshold=args.table_threshold,
        judge_model=args.judge_model,
        experiment_name=args.experiment_name,
        skip_langsmith=args.skip_langsmith,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

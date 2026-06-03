#!/usr/bin/env python3
"""Re-run judge layer only on saved golden eval results (reuse answers + retrieval metrics)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REDO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from eval_msft_golden_pipeline import (
    DEFAULT_DATASET,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_WORKSPACE,
    evaluate_answer_layer,
    summarize_results,
)
from golden_eval_utils import normalize_chunk_id
from text_vector_rag_inference import load_chunks, load_env_file, load_table_lookup

CHUNK_ID_RE = re.compile(
    r"(?:source\.pdf::)?((?:text_\d+|table_group_\d+|table_\d+(?:_merged)?))",
    re.IGNORECASE,
)


def _short_chunk_id(chunk_id: str) -> str:
    return normalize_chunk_id(chunk_id)


def reconstruct_inference(
    result: dict[str, Any],
    *,
    chunk_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild expanded_context / table_contexts from saved retrieval metrics + answer citations."""
    answer = result.get("answer") or ""
    final_ctx = (result.get("metrics") or {}).get("final_context") or {}
    text_n = int(final_ctx.get("text_chunks") or 10)
    table_n = int(final_ctx.get("table_contexts") or 0)

    cited: list[str] = []
    seen: set[str] = set()
    for match in CHUNK_ID_RE.finditer(answer):
        cid = _short_chunk_id(match.group(1))
        if cid not in seen:
            cited.append(cid)
            seen.add(cid)

    retrieval = (result.get("metrics") or {}).get("retrieval") or {}
    merged_ids = [
        _short_chunk_id(cid)
        for cid in ((retrieval.get("per_path") or {}).get("merged") or {}).get("top_ids") or []
    ]
    gt_id = _short_chunk_id(result.get("ground_truth_chunk_id") or "")

    ordered_text_ids: list[str] = []
    for cid in [gt_id, *cited, *merged_ids]:
        if not cid or cid.startswith("table"):
            continue
        if cid in ordered_text_ids:
            continue
        if cid in chunk_by_id or f"source.pdf::{cid}" in chunk_by_id:
            ordered_text_ids.append(cid)
        if len(ordered_text_ids) >= text_n:
            break

    expanded_context: list[dict[str, Any]] = []
    for cid in ordered_text_ids:
        chunk = chunk_by_id.get(cid) or chunk_by_id.get(f"source.pdf::{cid}")
        if not chunk:
            continue
        expanded_context.append(
            {
                "chunk_id": _short_chunk_id(chunk["chunk_id"]),
                "header_path": chunk.get("metadata", {}).get("header_path") or [],
            }
        )

    table_contexts: list[dict[str, Any]] = []
    table_candidates = [cid for cid in cited if cid.startswith("table")]
    for cid in table_candidates:
        if len(table_contexts) >= table_n:
            break
        table_contexts.append({"table_id": cid, "header_path": []})

    return {
        "answer": answer,
        "expanded_context": expanded_context,
        "table_contexts": table_contexts,
    }


def rejudge(
    input_path: Path,
    output_path: Path,
    *,
    dataset_path: Path,
    workspace: Path,
    judge_model: str,
    rerank_top_n: int,
) -> dict[str, Any]:
    load_env_file()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    golden = {
        row["question_id"]: row
        for row in json.loads(dataset_path.read_text(encoding="utf-8"))["questions"]
    }

    db_path = workspace / "index" / "vectors.db"
    assets_path = workspace / "assets.json"
    chunks = load_chunks(db_path)
    chunk_lookup = {chunk["chunk_id"]: chunk["content"] for chunk in chunks}
    chunk_by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_by_id[chunk["chunk_id"]] = chunk
        chunk_by_id[_short_chunk_id(chunk["chunk_id"])] = chunk
    table_lookup = load_table_lookup(assets_path)

    results: list[dict[str, Any]] = payload.get("results") or []
    settings = dict(payload.get("settings") or {})
    settings["judge_model"] = judge_model

    for idx, result in enumerate(results, 1):
        qid = result["question_id"]
        row = golden[qid]
        print(f"[{idx}/{len(results)}] rejudge {qid}", flush=True)

        inference = reconstruct_inference(result, chunk_by_id=chunk_by_id)
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

        latency = dict(result.get("latency") or {})
        old_judge = float(latency.get("judge_sec") or 0.0)
        latency["judge_sec"] = round(judge_sec, 3)
        latency["total_sec"] = round(float(latency.get("total_sec") or 0.0) - old_judge + judge_sec, 3)
        result["latency"] = latency
        result["metrics"]["judge"] = judge

        partial = {
            "dataset_path": payload.get("dataset_path") or str(dataset_path),
            "workspace": payload.get("workspace") or str(workspace),
            "settings": settings,
            "rejudge": {
                "source_results": str(input_path),
                "judge_model": judge_model,
                "full_context_in_judge": True,
                "reconstructed_context": True,
            },
            "summary": summarize_results(results[:idx], rerank_top_n=rerank_top_n),
            "results": results[:idx],
            "partial": idx < len(results),
        }
        output_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    output = {
        "dataset_path": payload.get("dataset_path") or str(dataset_path),
        "workspace": payload.get("workspace") or str(workspace),
        "settings": settings,
        "rejudge": {
            "source_results": str(input_path),
            "judge_model": judge_model,
            "full_context_in_judge": True,
            "reconstructed_context": True,
        },
        "summary": summarize_results(results, rerank_top_n=rerank_top_n),
        "results": results,
        "langsmith": {"enabled": False, "reason": "rejudge_only"},
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge saved golden eval results without re-running RAG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INFERENCE_DIR / "msft_fy2025_golden_eval_text40_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_DIR / "msft_fy2025_golden_eval_text40_results_v2.json",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--rerank-top-n", type=int, default=3)
    args = parser.parse_args()

    output = rejudge(
        args.input,
        args.output,
        dataset_path=args.dataset,
        workspace=args.workspace,
        judge_model=args.judge_model,
        rerank_top_n=args.rerank_top_n,
    )
    summary = output["summary"]
    print(
        json.dumps(
            {
                "n": summary["n"],
                "judge_model": args.judge_model,
                "mean_faithfulness": summary.get("mean_faithfulness"),
                "mean_context_precision": summary.get("mean_context_precision"),
                "mean_llm_judge_score": summary.get("mean_llm_judge_score"),
                "mean_judge_sec": summary.get("latency", {}).get("mean_judge_sec"),
                "mean_total_sec": summary.get("latency", {}).get("mean_total_sec"),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

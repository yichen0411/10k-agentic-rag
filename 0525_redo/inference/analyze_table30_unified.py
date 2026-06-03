#!/usr/bin/env python3
"""Unified table-30 analysis: retriever rank/score + final context source breakdown."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INFERENCE))
sys.path.insert(0, str(ROOT / "0525_redo" / "chunking"))

from build_text_vector_db import DEFAULT_EMBED_MODEL, embed_texts_fireworks
from cross_encoder_rerank import resolve_rerank_model
from golden_eval_utils import normalize_chunk_id, select_table_questions, table_ground_truth_ids
from text_vector_rag_inference import (
    DEFAULT_CHAT_MODEL,
    filter_table_hits_by_threshold,
    load_chunks,
    load_env_file,
    load_table_lookup,
    run_pipeline,
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def table_header(table: dict[str, Any] | None) -> str:
    if not table:
        return "unknown"
    section = table.get("section_ref") or {}
    subsection = table.get("subsection_ref") or {}
    parts = [section.get("section_title"), *(subsection.get("path") or [])]
    return " > ".join(p for p in parts if p) or "unknown"


def item_part(header: str) -> str:
    if not header or header == "unknown":
        return "unknown"
    return header.split(" > ")[0]


def expand_table_aliases(table_id: str, table_lookup: dict[str, dict[str, Any]]) -> set[str]:
    aliases = {normalize_chunk_id(table_id)}
    table = table_lookup.get(table_id)
    if not table:
        return aliases
    aliases.add(normalize_chunk_id(table.get("table_id") or table_id))
    for sid in table.get("source_table_ids") or []:
        aliases.add(normalize_chunk_id(sid))
        aliases.add(normalize_chunk_id(f"{sid}_merged"))
    return aliases


def gt_in_aliases(gt: set[str], table_id: str, table_lookup: dict[str, dict[str, Any]]) -> bool:
    aliases = expand_table_aliases(table_id, table_lookup)
    return bool(gt & aliases)


def load_table30_questions() -> list[dict[str, Any]]:
    golden50 = json.loads((ROOT / "0525_redo/common/msft_fy2025_golden_eval_50.json").read_text())
    table20 = json.loads((ROOT / "0525_redo/common/msft_fy2025_golden_eval_table20.json").read_text())
    q10 = select_table_questions(golden50["questions"])
    q20 = select_table_questions(table20["questions"])
    return q10 + q20


def load_llm_scores() -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for path in [
        INFERENCE / "msft_fy2025_golden_eval_table10_results.json",
        INFERENCE / "msft_fy2025_golden_eval_table20_results.json",
    ]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            scores[row["question_id"]] = (row.get("metrics") or {}).get("judge", {}).get("llm_judge_score")
    return scores


def retriever_analysis(
    questions: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    table_chunks: list[dict[str, Any]],
    fw_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        gt = table_ground_truth_ids(q, table_lookup)
        emb = embed_texts_fireworks([q["question"]], model=DEFAULT_EMBED_MODEL, api_key=fw_key)[0]
        scored = []
        for tc in table_chunks:
            tid = normalize_chunk_id(tc.get("metadata", {}).get("table_id") or tc["chunk_id"])
            scored.append((cosine(emb, tc["embedding"]), tid))
        scored.sort(key=lambda x: -x[0])

        best_rank = None
        best_score = None
        best_hit = None
        for rank, (score, tid) in enumerate(scored, 1):
            if gt_in_aliases(gt, tid, table_lookup):
                best_rank, best_score, best_hit = rank, score, tid
                break

        top5 = [(i + 1, tid, round(s, 3)) for i, (s, tid) in enumerate(scored[:5])]
        thr_pass = best_score is not None and best_rank is not None and best_rank <= 5 and best_score >= 0.65

        gt_table = table_lookup.get(q["ground_truth_chunk_id"])
        rows.append(
            {
                "question_id": q["question_id"],
                "question": q["question"][:100],
                "gt_table_id": q["ground_truth_chunk_id"],
                "gt_section": table_header(gt_table),
                "gt_item": item_part(table_header(gt_table)),
                "gt_text_chunk_id": q.get("ground_truth_text_chunk_id"),
                "rank": best_rank,
                "score": round(best_score, 3) if best_score else None,
                "hit_table_id": best_hit,
                "pass_top5_thr065": thr_pass,
                "top5": top5,
            }
        )
    return rows


def pipeline_source_analysis(
    questions: list[dict[str, Any]],
    workspace: Path,
    table_lookup: dict[str, dict[str, Any]],
    llm_scores: dict[str, float | None],
    rerank_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, q in enumerate(questions, 1):
        gt = table_ground_truth_ids(q, table_lookup)
        inf = run_pipeline(
            q["question"],
            db_path=workspace / "index/vectors.db",
            vector_top_k=10,
            bm25_top_k=10,
            rerank_top_n=3,
            max_context_chunks=14,
            chat_model=DEFAULT_CHAT_MODEL,
            embed_model=DEFAULT_EMBED_MODEL,
            assets_path=workspace / "assets.json",
            table_db_path=workspace / "index/table_vectors.db",
            table_similarity_threshold=0.65,
            rerank_model=rerank_model,
        )

        from_tv: set[str] = set()
        from_text: set[str] = set()
        for tc in inf.get("table_contexts") or []:
            tid = tc.get("table_id")
            if not tid or not gt_in_aliases(gt, tid, table_lookup):
                continue
            sk = tc.get("source_kind")
            if sk == "table_vector":
                from_tv.add(tid)
            elif sk == "text_chunk_ref":
                from_text.add(tid)

        if from_tv and from_text:
            gt_source = "overlap"
        elif from_tv:
            gt_source = "table_vector"
        elif from_text:
            gt_source = "text_chunk_ref"
        else:
            gt_source = "neither"

        text_chunks = inf.get("expanded_context") or []
        text_with_gt_ref = 0
        for chunk in text_chunks:
            refs = chunk.get("metadata", {}).get("table_refs") or []
            if any(gt_in_aliases(gt, normalize_chunk_id(r), table_lookup) for r in refs):
                text_with_gt_ref += 1

        tv_rank = None
        for hit in inf.get("table_vector_hits") or []:
            tid = hit.get("table_id")
            if tid and gt_in_aliases(gt, tid, table_lookup):
                tv_rank = hit.get("rank")
                break

        rows.append(
            {
                "question_id": q["question_id"],
                "gt_item": item_part(table_header(table_lookup.get(q["ground_truth_chunk_id"]))),
                "gt_section": table_header(table_lookup.get(q["ground_truth_chunk_id"])),
                "gt_source_in_final": gt_source,
                "gt_table_ids_final": sorted(from_tv | from_text),
                "from_table_vector": sorted(from_tv),
                "from_text_chunk_ref": sorted(from_text),
                "table_vector_rank_in_top5": tv_rank,
                "pass_top5_thr065": bool(
                    any(
                        h.get("passed_threshold")
                        for h in inf.get("table_vector_hits_filtered") or []
                        if gt_in_aliases(gt, h.get("table_id") or "", table_lookup)
                    )
                ),
                "final_text_chunks": len(text_chunks),
                "final_table_contexts": len(inf.get("table_contexts") or []),
                "text_chunks_with_gt_table_ref": text_with_gt_ref,
                "llm_judge": llm_scores.get(q["question_id"]),
            }
        )
        print(f"[{idx}/{len(questions)}] {q['question_id']} {gt_source} llm={llm_scores.get(q['question_id'])}", flush=True)
    return rows


def summarize_retriever(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [r["rank"] for r in rows if r["rank"]]
    scores = [r["score"] for r in rows if r["score"]]
    n = len(rows)
    return {
        "n": n,
        "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "recall_at_1": sum(1 for r in ranks if r <= 1) / n,
        "recall_at_3": sum(1 for r in ranks if r <= 3) / n,
        "recall_at_5": sum(1 for r in ranks if r <= 5) / n,
        "recall_at_10": sum(1 for r in ranks if r <= 10) / n,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "pass_top5_thr065": sum(1 for r in rows if r["pass_top5_thr065"]) / n,
    }


def summarize_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cats: dict[str, list[str]] = {}
    by_item: dict[str, dict[str, int]] = {}
    for r in rows:
        src = r["gt_source_in_final"]
        cats.setdefault(src, []).append(r["question_id"])
        item = r["gt_item"]
        by_item.setdefault(item, {})
        by_item[item][src] = by_item[item].get(src, 0) + 1
    return {"by_source": {k: {"count": len(v), "ids": v} for k, v in cats.items()}, "by_gt_item": by_item}


def main() -> None:
    load_env_file()
    fw_key = os.environ["FIREWORKS_API_KEY"]
    workspace = ROOT / "data/chunk_studio/1779921176-msft-fy2025-10-k-8d505c867d"
    table_lookup = load_table_lookup(workspace / "assets.json")
    table_chunks = load_chunks(workspace / "index/table_vectors.db")
    questions = load_table30_questions()
    llm_scores = load_llm_scores()
    rerank_model = resolve_rerank_model()

    print("=== retriever rank/score (30 Q) ===", flush=True)
    retriever_rows = retriever_analysis(questions, table_lookup, table_chunks, fw_key)
    retriever_summary = summarize_retriever(retriever_rows)

    print("=== final context source (30 Q) ===", flush=True)
    source_rows = pipeline_source_analysis(questions, workspace, table_lookup, llm_scores, rerank_model)
    source_summary = summarize_sources(source_rows)

    merged = []
    src_map = {r["question_id"]: r for r in source_rows}
    for r in retriever_rows:
        s = src_map[r["question_id"]]
        merged.append({**r, **s})

    out = INFERENCE / "msft_fy2025_table30_unified_analysis.json"
    payload = {
        "n": 30,
        "settings": {"table_vector_top_k": 5, "table_threshold": 0.65, "rerank_top_n": 3},
        "retriever_summary": retriever_summary,
        "source_summary": source_summary,
        "questions": merged,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"retriever_summary": retriever_summary, "source_summary": source_summary}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

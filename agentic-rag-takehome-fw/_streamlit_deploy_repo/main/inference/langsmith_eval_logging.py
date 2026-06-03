"""Upload MSFT golden eval results to LangSmith Experiments (Datasets & Testing tab)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

DEFAULT_DATASET_NAME = "msft-fy2025-golden-50"


def _configure_env() -> dict[str, Any]:
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not api_key:
        return {"enabled": False, "reason": "no_api_key"}
    project = (
        os.getenv("LANGSMITH_EVAL_PROJECT")
        or os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "10k-agentic-rag"
    ).strip()
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    return {"enabled": True, "project": project}


def _require_client():
    try:
        from langsmith import Client
    except ImportError as exc:
        raise RuntimeError("langsmith is not installed in the active venv.") from exc
    status = _configure_env()
    if not status.get("enabled"):
        raise RuntimeError(f"LangSmith disabled: {status.get('reason')}")
    return Client(), status


def ensure_golden_dataset(client, dataset_path: Path, *, dataset_name: str) -> str:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        dataset_id = str(dataset.id)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=payload.get("description") or "MSFT FY2025 golden eval dataset",
        )
        dataset_id = str(dataset.id)

    existing = {ex.metadata.get("question_id") for ex in client.list_examples(dataset_id=dataset_id)}
    to_create = []
    for row in payload.get("questions", []):
        qid = row["question_id"]
        if qid in existing:
            continue
        to_create.append(
            {
                "inputs": {
                    "question_id": qid,
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "ground_truth_chunk_id": row["ground_truth_chunk_id"],
                },
                "outputs": {
                    "expected_answer": row.get("expected_answer", ""),
                    "ground_truth_chunk_content": row.get("ground_truth_chunk_content", ""),
                },
                "metadata": {
                    "question_id": qid,
                    "question_type": row.get("question_type"),
                    "paraphrase_of_chunk": row.get("paraphrase_of_chunk"),
                },
            }
        )
    if to_create:
        client.create_examples(
            dataset_id=dataset_id,
            inputs=[row["inputs"] for row in to_create],
            outputs=[row["outputs"] for row in to_create],
            metadata=[row["metadata"] for row in to_create],
        )
    return dataset_name


def _metric_from_run(run: Any, *path: str) -> float | None:
    outputs = run.outputs or {}
    cur: Any = outputs
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, bool):
        return 1.0 if cur else 0.0
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _make_score_evaluator(key: str, *path: str) -> Callable[..., dict[str, Any]]:
    def _evaluator(run: Any, example: Any) -> dict[str, Any]:
        score = _metric_from_run(run, *path)
        if score is None:
            return {"key": key, "comment": "missing metric"}
        return {"key": key, "score": score}

    return _evaluator


def upload_eval_results_to_langsmith(
    *,
    dataset_path: Path,
    results_path: Path,
    experiment_prefix: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a LangSmith Experiment from an on-disk eval results JSON."""
    from langsmith.evaluation import evaluate

    client, status = _require_client()
    dataset_name = ensure_golden_dataset(client, dataset_path, dataset_name=dataset_name)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    cached = {row["question_id"]: row for row in payload.get("results", [])}
    if not cached:
        raise RuntimeError(f"No results found in {results_path}")

    def predict(inputs: dict[str, Any]) -> dict[str, Any]:
        qid = inputs["question_id"]
        row = cached.get(qid)
        if not row:
            return {"error": f"missing cached result for {qid}"}
        metrics = row.get("metrics") or {}
        return {
            "answer": row.get("answer"),
            "metrics": metrics,
            "latency_sec": row.get("latency_sec"),
        }

    evaluators = [
        _make_score_evaluator("vector_recall_at_10", "metrics", "retrieval", "per_path", "vector", "recall_at_k", "10"),
        _make_score_evaluator("bm25_recall_at_10", "metrics", "retrieval", "per_path", "bm25", "recall_at_k", "10"),
        _make_score_evaluator("merged_recall_at_10", "metrics", "retrieval", "per_path", "merged", "recall_at_k", "10"),
        _make_score_evaluator("merged_recall_at_15", "metrics", "retrieval", "per_path", "merged", "recall_at_k", "15"),
        _make_score_evaluator("fusion_gain_pool", "metrics", "retrieval", "fusion_gain_pool"),
        _make_score_evaluator("rerank_kicked_out", "metrics", "rerank", "kicked_out_by_rerank"),
        _make_score_evaluator("faithfulness", "metrics", "judge", "faithfulness"),
        _make_score_evaluator("answer_relevancy", "metrics", "judge", "answer_relevancy"),
        _make_score_evaluator("llm_judge_score", "metrics", "judge", "llm_judge_score"),
    ]

    experiment_results = evaluate(
        predict,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        description=description or f"MSFT golden eval upload from {results_path.name}",
        metadata={
            "dataset_path": str(dataset_path),
            "results_path": str(results_path),
            "n_uploaded": len(cached),
        },
        client=client,
    )

    experiment_name = getattr(experiment_results, "experiment_name", None) or experiment_prefix
    project = status["project"]
    return {
        "enabled": True,
        "project": project,
        "dataset_name": dataset_name,
        "experiment_prefix": experiment_prefix,
        "experiment_name": experiment_name,
        "n_results": len(cached),
        "langsmith_url": f"https://smith.langchain.com/o/default/projects/p/{project}/datasets",
        "hint": "Open LangSmith → Datasets & Testing → select dataset → Experiments tab.",
    }

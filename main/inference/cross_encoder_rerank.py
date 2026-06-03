"""Cross-encoder reranking via Fireworks /v1/rerank (Qwen3 Reranker family)."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

FIREWORKS_RERANK_URL = "https://api.fireworks.ai/inference/v1/rerank"
DEFAULT_CROSS_ENCODER_MODEL = "accounts/fireworks/models/qwen3-reranker-8b"
DEFAULT_RERANK_BACKEND = "cross_encoder"
DEFAULT_RERANK_TASK = (
    "Given a financial 10-K filing question, retrieve the passage that best answers the query."
)


def resolve_rerank_backend() -> str:
    return os.environ.get("RERANK_BACKEND", DEFAULT_RERANK_BACKEND).strip().lower()


def resolve_rerank_model() -> str:
    # FW_RERANK_MODEL is legacy LLM rerank only; cross-encoder uses FW_CROSS_ENCODER_MODEL.
    explicit = os.environ.get("FW_CROSS_ENCODER_MODEL") or os.environ.get("RERANK_MODEL")
    if explicit:
        return explicit
    return DEFAULT_CROSS_ENCODER_MODEL


def format_chunk_for_rerank(candidate: dict[str, Any], *, max_chars: int | None = None) -> str:
    header = " > ".join(candidate.get("header_path") or [])
    body = candidate.get("content") or ""
    if max_chars is not None:
        body = body[:max_chars]
    parts = []
    if header:
        parts.append(f"Section: {header}")
    if body:
        parts.append(body)
    return "\n".join(parts) if parts else candidate.get("candidate_id") or ""


def rerank_documents_fireworks(
    query: str,
    documents: list[str],
    *,
    model: str,
    api_key: str,
    top_n: int | None = None,
    task: str | None = None,
) -> list[dict[str, Any]]:
    if not documents:
        return []

    payload_obj: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False,
    }
    if top_n is not None:
        payload_obj["top_n"] = top_n
    task_text = task or os.environ.get("RERANK_TASK") or DEFAULT_RERANK_TASK
    if task_text:
        payload_obj["task"] = task_text

    payload = json.dumps(payload_obj).encode("utf-8")
    # Match embeddings helper: curl succeeds where urllib gets Cloudflare 403 locally.
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--fail-with-body",
            FIREWORKS_RERANK_URL,
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error_body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
        raise RuntimeError(f"Fireworks rerank failed: {error_body[:500]}")

    data = json.loads(result.stdout.decode("utf-8"))
    rows = []
    for row in data.get("data") or []:
        rows.append(
            {
                "index": int(row["index"]),
                "relevance_score": float(row["relevance_score"]),
            }
        )
    return rows

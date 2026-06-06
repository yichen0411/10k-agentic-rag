"""Shared RAG pipeline defaults for Chunk Studio (online) and golden eval (offline)."""

from __future__ import annotations

from typing import Any

# Prefer Fireworks for chat/answer/judge when both API keys are set (override: LLM_PROVIDER=anthropic).
DEFAULT_LLM_PROVIDER = "fireworks"

# Keep in sync with main/agent/tools.py run_rag_tool defaults.
STUDIO_VECTOR_TOP_K = 10
STUDIO_BM25_TOP_K = 10
STUDIO_RERANK_TOP_N = 3
STUDIO_MAX_CONTEXT_CHUNKS = 10
STUDIO_TABLE_VECTOR_TOP_K = 8
STUDIO_TABLE_SIMILARITY_THRESHOLD = 0.60
STUDIO_MAX_TABLE_CONTEXTS = 3
STUDIO_CONTEXT_EXPANSION_MODE = "sentences"
STUDIO_EXPANSION_ADJACENT_SENTENCES = 2

STUDIO_PIPELINE_DEFAULTS: dict[str, Any] = {
    "vector_top_k": STUDIO_VECTOR_TOP_K,
    "bm25_top_k": STUDIO_BM25_TOP_K,
    "rerank_top_n": STUDIO_RERANK_TOP_N,
    "max_context_chunks": STUDIO_MAX_CONTEXT_CHUNKS,
    "table_vector_top_k": STUDIO_TABLE_VECTOR_TOP_K,
    "table_similarity_threshold": STUDIO_TABLE_SIMILARITY_THRESHOLD,
    "max_table_contexts": STUDIO_MAX_TABLE_CONTEXTS,
}

STUDIO_PIPELINE_PROFILE: dict[str, Any] = {
    "pipeline": "dual_path_hybrid_text_rerank_plus_table_threshold",
    "llm_provider": DEFAULT_LLM_PROVIDER,
    "context_expansion_mode": STUDIO_CONTEXT_EXPANSION_MODE,
    "expansion_adjacent_sentences": STUDIO_EXPANSION_ADJACENT_SENTENCES,
    **STUDIO_PIPELINE_DEFAULTS,
    "profile_summary": (
        "Text path: vector+BM25 top-10 → cross-encoder rerank top-3 → "
        f"sentence expansion (±{STUDIO_EXPANSION_ADJACENT_SENTENCES}). "
        f"Table path: vector top-{STUDIO_TABLE_VECTOR_TOP_K} @ sim≥{STUDIO_TABLE_SIMILARITY_THRESHOLD:.2f} "
        f"→ cap {STUDIO_MAX_TABLE_CONTEXTS} tables to LLM."
    ),
}


def studio_run_pipeline_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword args for run_pipeline matching Chunk Studio rag tool."""
    out = dict(STUDIO_PIPELINE_DEFAULTS)
    out.update(overrides)
    return out

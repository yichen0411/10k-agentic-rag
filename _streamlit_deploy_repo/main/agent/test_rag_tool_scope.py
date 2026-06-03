#!/usr/bin/env python3
"""Unit tests for rag tool scope filter wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
for path in [str(AGENT_DIR), str(MAIN_ROOT / "chunking"), str(MAIN_ROOT / "inference")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from tools import _coerce_scope_filter, run_rag_tool  # noqa: E402


def test_coerce_scope_filter_splits_tickers() -> None:
    assert _coerce_scope_filter("MSFT, GOOGL") == ["MSFT", "GOOGL"]
    assert _coerce_scope_filter("MSFT") == "MSFT"


def test_run_rag_tool_passes_scope_filters() -> None:
    captured: dict = {}

    def fake_pipeline(question: str, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return {
            "answer": "ok",
            "latency": {},
            "reranked_top": [],
            "expanded_context": [],
            "table_contexts": [],
            "settings": {
                "ticker_filter": ["MSFT"],
                "fiscal_year_filter": ["FY2025"],
                "filtered_chunk_count": 10,
            },
        }

    with patch("tools.run_pipeline", side_effect=fake_pipeline):
        result = run_rag_tool(
            "What did management say about cloud growth?",
            ticker="msft",
            fiscal_year="2025",
        )

    assert captured["ticker_filter"] == ["MSFT"]
    assert captured["fiscal_year_filter"] == ["FY2025"]
    assert captured["table_vector_top_k"] == 8
    assert captured["table_similarity_threshold"] == 0.60
    assert captured["max_context_chunks"] == 10
    assert result["input"] == {
        "question": "What did management say about cloud growth?",
        "ticker": ["MSFT"],
        "fiscal_year": ["FY2025"],
    }
    assert result["scope_filters"]["filtered_chunk_count"] == 10


def test_run_rag_tool_rejects_multi_scope() -> None:
    with patch("tools.run_pipeline") as pipeline:
        result = run_rag_tool(
            "Compare AI risk disclosures.",
            ticker="MSFT,GOOGL",
            fiscal_year="2025",
        )

    pipeline.assert_not_called()
    assert result["ok"] is False
    assert result["status"] == "needs_decomposition"
    assert result["input"]["ticker"] == ["MSFT", "GOOGL"]


def test_run_rag_tool_insufficient_fallback_retries_once() -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_pipeline(question: str, **kwargs):
        calls.append((question, kwargs.get("retrieval_query")))
        answer = (
            "The provided context does not support a confident answer."
            if len(calls) == 1
            else "The filing gives concrete evidence about components, growth drivers, and margins."
        )
        return {
            "answer": answer,
            "latency": {},
            "reranked_top": [],
            "expanded_context": [],
            "table_contexts": [],
            "settings": {
                "ticker_filter": ["AAPL"],
                "fiscal_year_filter": ["FY2025"],
            },
        }

    with patch("tools.run_pipeline", side_effect=fake_pipeline):
        result = run_rag_tool(
            "What does Apple's 10-K say about the strategic importance of Services?",
            ticker="AAPL",
            fiscal_year="FY2025",
        )

    assert len(calls) == 2
    assert calls[0] == ("What does Apple's 10-K say about the strategic importance of Services?", None)
    assert calls[1][0] == "What does Apple's 10-K say about the strategic importance of Services?"
    assert calls[1][1] is not None and "preserves the same intent" in calls[1][1]
    assert result["status"] == "fallback_success"
    assert result["fallback_trace"][0]["reason"] == "insufficient_context_retrieval_rewrite"
    assert "components, growth drivers, and margins" in result["answer"]


def test_run_rag_tool_no_fallback_when_answer_supported() -> None:
    calls: list[str] = []

    def fake_pipeline(question: str, **kwargs):
        calls.append(question)
        return {
            "answer": "Apple Services revenue was $109.2 billion.",
            "latency": {},
            "reranked_top": [],
            "expanded_context": [],
            "table_contexts": [],
            "settings": {},
        }

    with patch("tools.run_pipeline", side_effect=fake_pipeline):
        result = run_rag_tool(
            "What was Apple's Services revenue in FY2025?",
            ticker="AAPL",
            fiscal_year="FY2025",
        )

    assert len(calls) == 1
    assert result["status"] == "success"
    assert result["fallback_trace"] == []


if __name__ == "__main__":
    test_coerce_scope_filter_splits_tickers()
    test_run_rag_tool_passes_scope_filters()
    test_run_rag_tool_rejects_multi_scope()
    test_run_rag_tool_insufficient_fallback_retries_once()
    test_run_rag_tool_no_fallback_when_answer_supported()
    print("ok")

#!/usr/bin/env python3
"""Unit tests for cross-encoder rerank helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_encoder_rerank import format_chunk_for_rerank, resolve_rerank_backend, resolve_rerank_model


def test_format_chunk_for_rerank_includes_header_and_body() -> None:
    text = format_chunk_for_rerank(
        {
            "candidate_id": "text_001",
            "header_path": ["Part II", "Item 7", "Revenue"],
            "content": "Net sales increased 5% year over year.",
        }
    )
    assert "Section: Part II > Item 7 > Revenue" in text
    assert "Net sales increased 5%" in text


def test_resolve_defaults() -> None:
    assert resolve_rerank_backend() in {"cross_encoder", "llm"}
    assert resolve_rerank_model()


if __name__ == "__main__":
    test_format_chunk_for_rerank_includes_header_and_body()
    test_resolve_defaults()
    print("ok")

#!/usr/bin/env python3
"""Unit tests for BM25 hybrid retrieval helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bm25_retrieval import bm25_search, merge_vector_and_bm25_hits, tokenize


def _chunk(chunk_id: str, header: list[str], content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": content,
        "section": header[0] if header else None,
        "metadata": {"header_path": header},
    }


def test_tokenize_lowercases_alnum() -> None:
    assert tokenize("Revenue grew 12% in FY2025") == ["revenue", "grew", "12", "in", "fy2025"]


def test_bm25_prefers_lexical_match() -> None:
    chunks = [
        _chunk("a", ["Item 7"], "Cloud revenue increased sharply year over year."),
        _chunk("b", ["Item 1A"], "Risk factors include competition and regulation."),
        _chunk("c", ["Item 7"], "Operating margin expanded due to efficiency gains."),
    ]
    hits = bm25_search("cloud revenue increased", chunks, top_k=2)
    assert hits[0]["candidate_id"] == "a"
    assert hits[0]["score"] > hits[1]["score"]


def test_merge_dedupes_and_preserves_vector_order() -> None:
    vector_hits = [
        {"rank": 1, "score": 0.9, "candidate_id": "a", "chunk_id": "a", "header_path": [], "content": "a", "metadata": {}},
        {"rank": 2, "score": 0.8, "candidate_id": "b", "chunk_id": "b", "header_path": [], "content": "b", "metadata": {}},
    ]
    bm25_hits = [
        {"rank": 1, "score": 4.2, "candidate_id": "b", "chunk_id": "b", "header_path": [], "content": "b", "metadata": {}},
        {"rank": 2, "score": 3.1, "candidate_id": "c", "chunk_id": "c", "header_path": [], "content": "c", "metadata": {}},
    ]
    merged = merge_vector_and_bm25_hits(vector_hits, bm25_hits)
    assert [hit["candidate_id"] for hit in merged] == ["a", "b", "c"]
    by_id = {hit["candidate_id"]: hit for hit in merged}
    assert by_id["b"]["retrieval_sources"] == ["vector", "bm25"]
    assert by_id["c"]["retrieval_sources"] == ["bm25"]


if __name__ == "__main__":
    test_tokenize_lowercases_alnum()
    test_bm25_prefers_lexical_match()
    test_merge_dedupes_and_preserves_vector_order()
    print("ok")

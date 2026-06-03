"""BM25 lexical retrieval over text chunks (hybrid with vector search)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def chunk_search_text(chunk: dict[str, Any]) -> str:
    header = " > ".join(chunk.get("metadata", {}).get("header_path") or chunk.get("header_path") or [])
    body = chunk.get("content") or ""
    if header:
        return f"{header}\n{body}"
    return body


class BM25Index:
    def __init__(self, corpus_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(corpus_tokens) if corpus_tokens else 0.0
        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.n_docs = len(corpus_tokens)
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        if doc_idx < 0 or doc_idx >= self.n_docs:
            return 0.0
        doc = self.corpus_tokens[doc_idx]
        dl = self.doc_len[doc_idx]
        tf = Counter(doc)
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += idf * freq * (self.k1 + 1) / denom
        return total

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        if not self.n_docs or not query_tokens or k <= 0:
            return []
        scored = [(idx, self.score_document(query_tokens, idx)) for idx in range(self.n_docs)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def bm25_search(query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not chunks or top_k <= 0:
        return []

    corpus_tokens = [tokenize(chunk_search_text(chunk)) for chunk in chunks]
    index = BM25Index(corpus_tokens)
    query_tokens = tokenize(query)
    ranked = index.top_k(query_tokens, top_k)

    hits: list[dict[str, Any]] = []
    for rank, (doc_idx, score) in enumerate(ranked, 1):
        chunk = chunks[doc_idx]
        meta = chunk.get("metadata") or {}
        hits.append(
            {
                "rank": rank,
                "score": score,
                "candidate_type": "text",
                "candidate_id": chunk["chunk_id"],
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "section": chunk.get("section"),
                "header_path": meta.get("header_path") or chunk.get("header_path") or [],
                "metadata": meta,
            }
        )
    return hits


def merge_vector_and_bm25_hits(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge vector top-k and BM25 top-k, dedupe by chunk id, vector order first."""
    by_id: dict[str, dict[str, Any]] = {}

    for hit in vector_hits:
        cid = hit["candidate_id"]
        entry = dict(hit)
        entry["vector_score"] = hit["score"]
        entry["vector_rank"] = hit["rank"]
        entry["retrieval_sources"] = ["vector"]
        by_id[cid] = entry

    for hit in bm25_hits:
        cid = hit["candidate_id"]
        if cid in by_id:
            by_id[cid]["bm25_score"] = hit["score"]
            by_id[cid]["bm25_rank"] = hit["rank"]
            if "bm25" not in by_id[cid]["retrieval_sources"]:
                by_id[cid]["retrieval_sources"].append("bm25")
            continue
        entry = dict(hit)
        entry["bm25_score"] = hit["score"]
        entry["bm25_rank"] = hit["rank"]
        entry["score"] = hit["score"]
        entry["retrieval_sources"] = ["bm25"]
        by_id[cid] = entry

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in vector_hits:
        cid = hit["candidate_id"]
        if cid in seen:
            continue
        merged.append(by_id[cid])
        seen.add(cid)
    for hit in bm25_hits:
        cid = hit["candidate_id"]
        if cid in seen:
            continue
        merged.append(by_id[cid])
        seen.add(cid)

    for rank, hit in enumerate(merged, 1):
        hit["rank"] = rank
    return merged

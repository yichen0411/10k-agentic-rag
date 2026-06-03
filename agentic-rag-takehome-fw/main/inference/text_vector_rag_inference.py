#!/usr/bin/env python3
"""Parallel text+table RAG: text rerank path + table threshold path."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path(__file__).resolve().parents[1]
CHUNKING_DIR = MAIN_ROOT / "chunking"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CHUNKING_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNKING_DIR))

from bm25_retrieval import bm25_search, merge_vector_and_bm25_hits
from build_text_vector_db import DEFAULT_EMBED_MODEL, embed_texts_fireworks
from cross_encoder_rerank import (
    format_chunk_for_rerank,
    rerank_documents_fireworks,
    resolve_rerank_backend,
)
from cross_encoder_rerank import resolve_rerank_model as resolve_cross_encoder_rerank_model
from filing_metadata import normalize_filter_values
from vlm_table_parse import compose_table_summary, section_ref_label, table_summary_topic

DEFAULT_DB = ROOT / "data" / "index" / "text_chunks" / "vectors.db"
DEFAULT_TABLE_DB = ROOT / "data" / "index" / "table_summaries" / "vectors.db"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "text_vector_rag_inference_result.json"
DEFAULT_ASSETS = ROOT / "data" / "index" / "merged_assets.json"
DEFAULT_CHAT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_RERANK_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_ANSWER_MODEL = "claude-sonnet-4-20250514"
DEFAULT_FIREWORKS_CHAT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
DEFAULT_FIREWORKS_RERANK_MODEL = "accounts/fireworks/models/qwen3-8b"
DEFAULT_TABLE_SIMILARITY_THRESHOLD = 0.65
DEFAULT_RETRIEVAL_RERANK_THRESHOLD = 0.60
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
TABLE_MARKER_RE = re.compile(r"\[\[TABLE:([^\]]+)\]\]")


def load_env_file(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def vector_from_blob(blob: bytes) -> list[float]:
    return list(array("f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-12
    nb = math.sqrt(sum(y * y for y in b)) + 1e-12
    return dot / (na * nb)


def call_anthropic(prompt: str, api_key: str, model: str, system: str, max_tokens: int = 800) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    result = None
    for attempt in range(6):
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "--fail-with-body",
                ANTHROPIC_MESSAGES_URL,
                "-H",
                f"x-api-key: {api_key}",
                "-H",
                "anthropic-version: 2023-06-01",
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
        if result.returncode == 0:
            break
        error_body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
        if "overloaded_error" not in error_body and "rate_limit_error" not in error_body:
            raise RuntimeError(f"Anthropic call failed: {error_body[:500]}")
        time.sleep(min(20, 3 * (attempt + 1)))
    if result is None or result.returncode != 0:
        error_body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
        raise RuntimeError(f"Anthropic call failed after retries: {error_body[:500]}")
    data = json.loads(result.stdout.decode("utf-8"))
    blocks = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    text = "\n".join(blocks).strip()
    if not text:
        raise RuntimeError(f"Anthropic response missing text: {json.dumps(data)[:500]}")
    return text


def resolve_chat_model() -> tuple[str, str]:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        model = os.environ.get("ANTHROPIC_CHAT_MODEL", os.environ.get("ANTHROPIC_SQL_MODEL", DEFAULT_CHAT_MODEL))
        return "anthropic", model
    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    if fireworks_key:
        model = os.environ.get("FW_CHAT_MODEL", DEFAULT_FIREWORKS_CHAT_MODEL)
        return "fireworks", model
    raise RuntimeError("ANTHROPIC_API_KEY or FIREWORKS_API_KEY is required for chat/rerank.")


def resolve_answer_model() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ.get(
            "ANTHROPIC_ANSWER_MODEL",
            os.environ.get("ANTHROPIC_CHAT_MODEL", DEFAULT_ANTHROPIC_ANSWER_MODEL),
        )
    _, model = resolve_chat_model()
    return model


def resolve_rerank_model() -> str:
    backend = resolve_rerank_backend()
    if backend == "llm":
        if os.environ.get("ANTHROPIC_API_KEY"):
            return os.environ.get("ANTHROPIC_RERANK_MODEL", DEFAULT_ANTHROPIC_RERANK_MODEL)
        explicit = os.environ.get("FW_RERANK_MODEL")
        if explicit:
            return explicit
        _, chat_model = resolve_chat_model()
        if chat_model.startswith("accounts/fireworks/"):
            return os.environ.get("FW_ROUTER_MODEL", DEFAULT_FIREWORKS_RERANK_MODEL)
        return chat_model
    return resolve_cross_encoder_rerank_model()


def call_fireworks_chat(prompt: str, api_key: str, model: str, system: str, max_tokens: int = 800, json_mode: bool = False) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Fireworks chat response missing text.")
    return text


def call_chat(prompt: str, system: str, max_tokens: int = 800, chat_model: str | None = None, json_mode: bool = False) -> str:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        if chat_model and not chat_model.startswith("accounts/fireworks/"):
            model = chat_model
        else:
            model = resolve_answer_model()
        return call_anthropic(prompt, api_key=anthropic_key, model=model, system=system, max_tokens=max_tokens)

    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    if not fireworks_key:
        raise RuntimeError("ANTHROPIC_API_KEY or FIREWORKS_API_KEY is required for chat/rerank.")
    model = chat_model or os.environ.get("FW_CHAT_MODEL", DEFAULT_FIREWORKS_CHAT_MODEL)
    return call_fireworks_chat(
        prompt,
        api_key=fireworks_key,
        model=model,
        system=system,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start : end + 1])


def retrieval_confidence(
    text_anchors: list[dict[str, Any]],
    filtered_table_hits: list[dict[str, Any]],
    *,
    retrieval_rerank_threshold: float,
    min_table_similarity_for_confidence: float,
) -> dict[str, Any]:
    top_text_rerank = text_anchors[0].get("rerank_score") if text_anchors else None
    top_table_score = filtered_table_hits[0].get("score") if filtered_table_hits else None
    weak_text = top_text_rerank is None or float(top_text_rerank) < retrieval_rerank_threshold
    weak_table = top_table_score is None or float(top_table_score) < min_table_similarity_for_confidence
    return {
        "text_top1_rerank_score": top_text_rerank,
        "table_top1_similarity": top_table_score,
        "text_threshold": retrieval_rerank_threshold,
        "table_threshold": min_table_similarity_for_confidence,
        "has_table_context_candidate": bool(filtered_table_hits),
        "weak_text": weak_text,
        "weak_table": weak_table,
        "should_retry": weak_text and weak_table,
    }


def rewrite_retrieval_query(query: str, *, chat_model: str) -> str:
    prompt = f"""Original filing question:
{query}

Rewrite the question as a concise retrieval query for 10-K search.

Rules:
- Preserve the same company, fiscal period, entities, metric/topic, and intent.
- Use concrete 10-K wording, synonyms, and related filing terms likely to appear in the document.
- Do not broaden the scope.
- Do not answer the question.
- Return only the rewritten retrieval query text.
"""
    rewritten = call_chat(
        prompt,
        system="You rewrite financial filing questions into precise retrieval queries.",
        max_tokens=180,
        chat_model=chat_model,
    )
    return re.sub(r"\s+", " ", rewritten).strip().strip('"')


def context_sufficiency_check(
    query: str,
    context_chunks: list[dict[str, Any]],
    table_contexts: list[dict[str, Any]],
    *,
    chat_model: str,
) -> dict[str, Any]:
    if not context_chunks and not table_contexts:
        return {
            "sufficient": False,
            "confidence": "low",
            "reason": "No text or table context was retrieved.",
            "missing": ["relevant filing context"],
            "evidence_type": "none",
        }
    prompt = f"""Question:
{query}

Decide whether the supplied context is sufficient to answer the question using only explicit evidence from text or direct calculation from tables.

Return JSON only:
{{
  "sufficient": true,
  "confidence": "high|medium|low",
  "reason": "short reason",
  "missing": ["optional short missing facts"],
  "evidence_type": "text|table|text_and_table|none"
}}

Rules:
- Do not answer the question.
- Mark sufficient=true only if the context contains direct evidence or enough table values for a direct calculation.
- For table/numeric questions, table context can be sufficient even if narrative text is thin.
- For qualitative intent, strategy, importance, drivers, or risk questions, require explicit support for
  the facts that would be stated in the answer, but do not require the exact abstract wording from the
  question. For example, growth drivers, segment components, margin tables, risk disclosures, and
  business descriptions can be sufficient evidence for a bounded answer about a topic's role or
  importance, as long as the final answer does not invent management intent.

Context:
{format_context(context_chunks, table_contexts)}
"""
    try:
        response = call_chat(
            prompt,
            system="You are a strict evidence sufficiency classifier. Return JSON only.",
            max_tokens=350,
            chat_model=chat_model,
            json_mode=True,
        )
        data = parse_json_response(response)
    except Exception as exc:
        return {
            "sufficient": True,
            "confidence": "low",
            "reason": f"Sufficiency check failed open: {exc}",
            "missing": [],
            "evidence_type": "text_and_table" if table_contexts and context_chunks else ("table" if table_contexts else "text"),
            "error": str(exc),
        }
    return {
        "sufficient": bool(data.get("sufficient")),
        "confidence": data.get("confidence") or "low",
        "reason": data.get("reason") or "",
        "missing": data.get("missing") if isinstance(data.get("missing"), list) else [],
        "evidence_type": data.get("evidence_type") or "none",
    }


def load_chunks(
    db_path: Path,
    *,
    ticker_filter: str | list[str] | None = None,
    fiscal_year_filter: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    tickers = normalize_filter_values(ticker_filter, kind="ticker")
    fiscal_years = normalize_filter_values(fiscal_year_filter, kind="fiscal_year")

    conditions: list[str] = []
    params: list[str] = []
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        conditions.append(f"UPPER(ticker) IN ({placeholders})")
        params.extend(tickers)
    if fiscal_years:
        placeholders = ",".join("?" * len(fiscal_years))
        conditions.append(f"UPPER(fiscal_year) IN ({placeholders})")
        params.extend(fiscal_years)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT id, chunk_id, content, embedding, ticker, fiscal_year, section, source_file, metadata_json
        FROM chunks
        {where_clause}
        ORDER BY id
        """,
        params,
    ).fetchall()
    conn.close()
    return [
        {
            "row_id": row["id"],
            "chunk_id": row["chunk_id"],
            "content": row["content"],
            "embedding": vector_from_blob(row["embedding"]),
            "ticker": row["ticker"],
            "fiscal_year": row["fiscal_year"],
            "section": row["section"],
            "source_file": row["source_file"],
            "metadata": json.loads(row["metadata_json"]),
        }
        for row in rows
    ]


def load_table_lookup(assets_path: Path | list[Path] | None) -> dict[str, dict[str, Any]]:
    if not assets_path:
        return {}
    paths = [assets_path] if isinstance(assets_path, Path) else [path for path in assets_path if path.exists()]
    if not paths:
        return {}

    multi_source = len(paths) > 1 or any(path.name == "merged_assets.json" for path in paths)

    def register(table_id: str, table: dict[str, Any]) -> None:
        if table_id and table_id not in tables:
            tables[table_id] = table

    tables: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_file = payload.get("source_file") or path.name
        if source_file == "all_filings":
            multi_source = True
        for table in payload.get("tables", []):
            table_id = table.get("table_id")
            if not table_id:
                continue
            table_source = table.get("source_file") or source_file
            if multi_source and table_source not in {"all_filings", "source.pdf", ""}:
                register(f"{table_source}::{table_id}", table)
                for source_id in table.get("source_table_ids") or []:
                    register(f"{table_source}::{source_id}", table)
                    register(f"{table_source}::{source_id}_merged", table)
                continue
            register(table_id, table)
            for source_id in table.get("source_table_ids") or []:
                register(source_id, table)
                register(f"{source_id}_merged", table)
    return tables


def resolve_table_lookup_key(table_lookup: dict[str, dict[str, Any]], table_id: str, source_file: str | None = None) -> str | None:
    if not table_id:
        return None
    if source_file and source_file not in {"source.pdf", "all_filings", ""}:
        composite = f"{source_file}::{table_id}"
        if composite in table_lookup:
            return composite
    if table_id in table_lookup:
        return table_id
    return None


def table_markdown(table: dict[str, Any]) -> str:
    vlm = table.get("vlm_parse") or {}
    if vlm.get("status") == "success" and vlm.get("markdown"):
        return vlm["markdown"]
    rows = table.get("raw_rows") or []
    if rows:
        lines = [" | ".join(cell or "" for cell in row) for row in rows]
        return "\n".join(lines)
    return table.get("raw_text") or ""


def table_summary_text(table: dict[str, Any]) -> str:
    vlm = table.get("vlm_parse") or {}
    if vlm.get("summary"):
        return str(vlm["summary"])
    topic = table_summary_topic(vlm)
    if topic:
        return compose_table_summary(table, topic, vlm.get("section_ref"))
    return f"Table on page {table.get('page_start')}: {section_ref_label(table)}"


def table_section_path(table: dict[str, Any]) -> str:
    vlm = table.get("vlm_parse") or {}
    if vlm.get("section_ref"):
        return str(vlm["section_ref"])
    return section_ref_label(table)


def vector_search(query: str, chunks: list[dict[str, Any]], top_k: int, embed_model: str, fireworks_key: str) -> list[dict[str, Any]]:
    query_embedding = embed_texts_fireworks([query], model=embed_model, api_key=fireworks_key)[0]
    return vector_search_with_embedding(query_embedding, chunks, top_k)


def vector_search_with_embedding(query_embedding: list[float], chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = [(cosine(query_embedding, chunk["embedding"]), chunk) for chunk in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "rank": idx,
            "score": score,
            "candidate_type": "text",
            "candidate_id": chunk["chunk_id"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "section": chunk["section"],
            "header_path": chunk["metadata"].get("header_path") or [],
            "metadata": chunk["metadata"],
        }
        for idx, (score, chunk) in enumerate(scored[:top_k], 1)
    ]


def vector_search_table_summaries(
    query: str,
    table_chunks: list[dict[str, Any]],
    top_k: int,
    embed_model: str,
    fireworks_key: str,
) -> list[dict[str, Any]]:
    if not table_chunks:
        return []
    query_embedding = embed_texts_fireworks([query], model=embed_model, api_key=fireworks_key)[0]
    return vector_search_table_summaries_with_embedding(query_embedding, table_chunks, top_k)


def vector_search_table_summaries_with_embedding(
    query_embedding: list[float],
    table_chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not table_chunks:
        return []
    scored = [(cosine(query_embedding, chunk["embedding"]), chunk) for chunk in table_chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = []
    for idx, (score, chunk) in enumerate(scored[:top_k], 1):
        meta = chunk["metadata"]
        hits.append(
            {
                "rank": idx,
                "score": score,
                "candidate_type": "table",
                "candidate_id": meta.get("table_id") or chunk["chunk_id"],
                "table_id": meta.get("table_id"),
                "content": chunk["content"],
                "section": chunk["section"],
                "header_path": meta.get("header_path") or [],
                "metadata": meta,
            }
        )
    return hits


def filter_table_hits_by_threshold(table_hits: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [hit for hit in table_hits if float(hit.get("score") or 0.0) >= threshold]


def merge_retrieval_candidates(text_hits: list[dict[str, Any]], table_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    seen_table: set[str] = set()
    for hit in text_hits:
        cid = hit["candidate_id"]
        if cid in seen_text:
            continue
        seen_text.add(cid)
        merged.append(hit)
    for hit in table_hits:
        tid = hit["candidate_id"]
        if tid in seen_table:
            continue
        seen_table.add(tid)
        merged.append(hit)
    return merged


def _fill_rerank_top_n(reranked: list[dict[str, Any]], text_hits: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    seen = {candidate["candidate_id"] for candidate in reranked}
    for candidate in text_hits:
        if len(reranked) >= top_n:
            break
        if candidate["candidate_id"] not in seen:
            reranked.append(candidate)
            seen.add(candidate["candidate_id"])
    return reranked[:top_n]


def _rerank_text_chunks_llm(query: str, text_hits: list[dict[str, Any]], rerank_model: str, top_n: int) -> list[dict[str, Any]]:
    candidate_text = []
    for candidate in text_hits:
        header = " > ".join(candidate.get("header_path") or [])
        body = candidate.get("content", "")
        vector_score = candidate.get("vector_score", candidate.get("score"))
        bm25_score = candidate.get("bm25_score")
        score_lines = [f"vector_score: {float(vector_score):.4f}"] if vector_score is not None else []
        if bm25_score is not None:
            score_lines.append(f"bm25_score: {float(bm25_score):.4f}")
        candidate_text.append(
            f"""Candidate {candidate['rank']}
candidate_id: {candidate['candidate_id']}
{chr(10).join(score_lines) or "retrieval_score: n/a"}
header: {header}
text:
{body}"""
        )

    joined_candidates = "\n\n---\n\n".join(candidate_text)
    prompt = f"""Query:
{query}

Rerank the candidate text chunks for answering the query.

Return JSON only:
{{
  "selected": [
    {{"candidate_id": "...", "relevance": 0.0, "reason": "..."}}
  ]
}}

Rules:
- Select exactly {top_n} text chunks from the list.
- Prefer direct evidence for numeric questions when the chunk contains the value.
- Prefer narrative chunks for explanation questions.
- Do not invent candidate ids.

Candidates:
{joined_candidates}
"""
    response = call_chat(
        prompt,
        system="You are a strict text-chunk reranker. Return JSON only.",
        max_tokens=700,
        chat_model=rerank_model,
        json_mode=True,
    )
    try:
        data = parse_json_response(response)
    except json.JSONDecodeError:
        return text_hits[:top_n]

    selected_rows = data.get("selected", [])
    by_id = {candidate["candidate_id"]: candidate for candidate in text_hits}

    reranked = []
    seen = set()
    for row in selected_rows:
        candidate_id = row.get("candidate_id")
        if candidate_id in by_id and candidate_id not in seen:
            item = dict(by_id[candidate_id])
            item["rerank_score"] = row.get("relevance")
            item["rerank_reason"] = row.get("reason", "")
            reranked.append(item)
            seen.add(candidate_id)

    return _fill_rerank_top_n(reranked, text_hits, top_n)


def _rerank_text_chunks_cross_encoder(
    query: str,
    text_hits: list[dict[str, Any]],
    rerank_model: str,
    top_n: int,
    *,
    fireworks_key: str,
) -> list[dict[str, Any]]:
    documents = [format_chunk_for_rerank(candidate) for candidate in text_hits]
    try:
        rows = rerank_documents_fireworks(
            query,
            documents,
            model=rerank_model,
            api_key=fireworks_key,
            top_n=top_n,
        )
    except RuntimeError:
        return text_hits[:top_n]

    reranked: list[dict[str, Any]] = []
    for row in rows:
        idx = row["index"]
        if idx < 0 or idx >= len(text_hits):
            continue
        item = dict(text_hits[idx])
        item["rerank_score"] = row["relevance_score"]
        item["rerank_reason"] = f"cross_encoder score={row['relevance_score']:.4f}"
        reranked.append(item)
    return _fill_rerank_top_n(reranked, text_hits, top_n)


def rerank_text_chunks(
    query: str,
    text_hits: list[dict[str, Any]],
    rerank_model: str,
    top_n: int,
    *,
    fireworks_key: str | None = None,
) -> list[dict[str, Any]]:
    if not text_hits:
        return []
    if len(text_hits) <= top_n:
        return list(text_hits[:top_n])

    backend = resolve_rerank_backend()
    if backend == "llm":
        return _rerank_text_chunks_llm(query, text_hits, rerank_model=rerank_model, top_n=top_n)

    api_key = fireworks_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        return text_hits[:top_n]
    return _rerank_text_chunks_cross_encoder(
        query,
        text_hits,
        rerank_model=rerank_model,
        top_n=top_n,
        fireworks_key=api_key,
    )


def rerank_top_chunks(query: str, candidates: list[dict[str, Any]], anthropic_key: str, rerank_model: str, top_n: int) -> list[dict[str, Any]]:
    text_only = [candidate for candidate in candidates if candidate.get("candidate_type", "text") == "text"]
    return rerank_text_chunks(query, text_only, rerank_model=rerank_model, top_n=top_n)


def local_context_around_markers(
    content: str,
    table_ids: set[str] | None = None,
    *,
    before_sentences: int = 2,
    after_sentences: int = 2,
) -> str:
    """Keep narrative immediately before/after inline table markers instead of whole chunk."""
    if not TABLE_MARKER_RE.search(content):
        return content
    wanted = set(table_ids or [])
    parts: list[str] = []
    cursor = 0
    for match in TABLE_MARKER_RE.finditer(content):
        table_id = match.group(1)
        if wanted and table_id not in wanted:
            continue
        prefix = content[cursor : match.start()]
        suffix = content[match.end() :]
        before = [part.strip() for part in re.split(r"(?<=[.!?])\s+", prefix) if part.strip()]
        after = [part.strip() for part in re.split(r"(?<=[.!?])\s+", suffix) if part.strip()]
        snippet = " ".join([*before[-before_sentences:], match.group(0), *after[:after_sentences]]).strip()
        if snippet:
            parts.append(snippet)
        cursor = match.end()
    if parts:
        return " ".join(parts)
    return content


def expansion_refs_for_chunk(chunk: dict[str, Any]) -> list[tuple[str, str]]:
    """Expand only within the same subsection text unit (split siblings), not across subsections."""
    meta = chunk["metadata"]
    anchors = meta.get("table_anchors") or []
    refs: list[tuple[str, str]] = []
    if anchors:
        refs.append((chunk["chunk_id"], "selected"))
        return refs
    if meta.get("same_text_unit_prev_vector_chunk_id"):
        refs.append((meta["same_text_unit_prev_vector_chunk_id"], "adjacent_prev"))
    refs.append((chunk["chunk_id"], "selected"))
    if meta.get("same_text_unit_next_vector_chunk_id"):
        refs.append((meta["same_text_unit_next_vector_chunk_id"], "adjacent_next"))
    return refs


def sentence_trim(text: str, first_n: int = 2, last_n: int = 2) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if len(sentences) <= first_n + last_n:
        return text
    return " ".join([*sentences[:first_n], "...", *sentences[-last_n:]])


def reference_excerpt(text: str, max_chars: int = 420) -> str:
    """Short, user-facing excerpt for citations without dumping whole chunks."""
    cleaned = TABLE_MARKER_RE.sub("[table]", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    excerpt = sentence_trim(cleaned, first_n=1, last_n=1)
    if len(excerpt) <= max_chars:
        return excerpt
    return f"{excerpt[: max_chars - 1].rstrip()}…"


def context_chunk_copy(chunk: dict[str, Any], role: str, table_ids: set[str] | None = None) -> dict[str, Any]:
    item = dict(chunk)
    item["context_role"] = role
    content = chunk["content"]
    if role == "selected" and (chunk["metadata"].get("table_anchors") or []):
        wanted = table_ids or {anchor.get("table_id") for anchor in chunk["metadata"].get("table_anchors") or [] if anchor.get("table_id")}
        item["context_content"] = local_context_around_markers(content, wanted)
    elif role.startswith("adjacent_"):
        item["context_content"] = sentence_trim(content)
    else:
        item["context_content"] = content
    return item


def expand_context(
    selected: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    max_context_chunks: int,
    wanted_table_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ordered: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    seen: set[str] = set()
    for selected_chunk in selected:
        for ref, role in expansion_refs_for_chunk(selected_chunk):
            if ref in seen:
                if role == "selected":
                    ordered[positions[ref]] = context_chunk_copy(
                        by_id[ref],
                        role,
                        wanted_table_ids if role == "selected" else None,
                    )
                continue
            if ref not in by_id:
                continue
            ref_chunk = by_id[ref]
            # Guardrail: expansion refs should already be same-section, but keep
            # this explicit because cross-section expansion is intentionally off.
            if ref != selected_chunk["chunk_id"]:
                same_section = ref_chunk["metadata"].get("section_ref_id") == selected_chunk["metadata"].get("section_ref_id")
                if not same_section or selected_chunk["metadata"].get("cross_section_expansion_allowed"):
                    continue
            ordered.append(
                context_chunk_copy(
                    ref_chunk,
                    role,
                    wanted_table_ids if role == "selected" else None,
                )
            )
            positions[ref] = len(ordered) - 1
            seen.add(ref)
            if len(ordered) >= max_context_chunks:
                return ordered
    return ordered


def lexical_overlap_score(query: str, text: str) -> int:
    query_terms = {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", query.lower()) if len(term) > 2}
    text_lower = text.lower()
    return sum(1 for term in query_terms if term in text_lower)


def collect_table_contexts(
    query: str,
    context_chunks: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    table_anchor_ids: list[str] | None = None,
    table_vector_hits: list[dict[str, Any]] | None = None,
    *,
    preserve_order: bool = False,
) -> list[dict[str, Any]]:
    table_contexts = []
    seen = set()
    order = 0

    def add_table(
        table_id: str,
        source_kind: str,
        source_id: str,
        source_order: int,
        vector_score: float | None = None,
        source_file: str | None = None,
    ) -> None:
        nonlocal order
        lookup_key = resolve_table_lookup_key(table_lookup, table_id, source_file)
        if not lookup_key or lookup_key in seen:
            return
        table = table_lookup[lookup_key]
        section = table.get("section_ref") or {}
        subsection = table.get("subsection_ref") or {}
        header_path = [section.get("section_title"), *(subsection.get("path") or [])]
        section_path = table_section_path(table)
        searchable = section_path + "\n" + table_summary_text(table)
        table_contexts.append(
            {
                "table_id": table_id,
                "source_kind": source_kind,
                "source_id": source_id,
                "source_order": source_order,
                "vector_score": vector_score,
                "query_overlap_score": lexical_overlap_score(query, searchable),
                "header_path": header_path,
                "section_path": section_path,
                "page_start": table.get("page_start"),
                "page_end": table.get("page_end"),
                "summary": table_summary_text(table),
                "markdown": table_markdown(table),
                "raw_text": table.get("raw_text") or "",
                "raw_rows": table.get("raw_rows") or [],
            }
        )
        seen.add(lookup_key)
        order += 1

    if table_vector_hits:
        for hit_idx, hit in enumerate(table_vector_hits):
            table_id = hit.get("table_id")
            if not table_id:
                continue
            source_file = (hit.get("metadata") or {}).get("source_file")
            add_table(
                table_id,
                "table_vector",
                hit.get("candidate_id") or table_id,
                -100 + hit_idx,
                vector_score=hit.get("score"),
                source_file=source_file,
            )
    else:
        for table_id in table_anchor_ids or []:
            add_table(table_id, "table_vector", table_id, -100 + order)

    for chunk_idx, chunk in enumerate(context_chunks):
        chunk_source = chunk.get("source_file") or (chunk.get("metadata") or {}).get("source_file")
        anchor_ids = [anchor.get("table_id") for anchor in chunk["metadata"].get("table_anchors", []) or [] if anchor.get("table_id")]
        for table_id in anchor_ids or chunk["metadata"].get("table_refs", []) or []:
            add_table(table_id, "text_chunk_ref", chunk["chunk_id"], chunk_idx, source_file=chunk_source)

    if not preserve_order:
        table_contexts.sort(key=lambda table: (table["query_overlap_score"], -table["source_order"]), reverse=True)
    return table_contexts


def assemble_dual_path_context(
    text_anchors: list[dict[str, Any]],
    filtered_table_hits: list[dict[str, Any]],
    text_chunks: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    max_context_chunks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table_anchor_ids = [hit["table_id"] for hit in filtered_table_hits if hit.get("table_id")]
    wanted_tables = set(table_anchor_ids)
    expanded: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    by_id = {chunk["chunk_id"]: chunk for chunk in text_chunks}

    for hit in text_anchors:
        selected = by_id.get(hit["chunk_id"])
        if not selected:
            continue
        for ref, role in expansion_refs_for_chunk(selected):
            if ref in seen_text and role != "selected":
                continue
            chunk = by_id.get(ref)
            if not chunk:
                continue
            if ref != selected["chunk_id"]:
                same_section = chunk["metadata"].get("section_ref_id") == selected["metadata"].get("section_ref_id")
                if not same_section or selected["metadata"].get("cross_section_expansion_allowed"):
                    continue
            expanded.append(context_chunk_copy(chunk, role, wanted_tables if role == "selected" else None))
            seen_text.add(ref)
            if len(expanded) >= max_context_chunks:
                break
        if len(expanded) >= max_context_chunks:
            break
    table_contexts = collect_table_contexts(
        "",
        expanded,
        table_lookup,
        table_vector_hits=filtered_table_hits,
        preserve_order=True,
    )
    return expanded[:max_context_chunks], table_contexts


def assemble_context(
    reranked: list[dict[str, Any]],
    text_chunks: list[dict[str, Any]],
    table_lookup: dict[str, dict[str, Any]],
    max_context_chunks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text_anchors = [anchor for anchor in reranked if anchor.get("candidate_type", "text") == "text"]
    table_hits = [anchor for anchor in reranked if anchor.get("candidate_type") == "table"]
    return assemble_dual_path_context(
        text_anchors,
        table_hits,
        text_chunks,
        table_lookup,
        max_context_chunks=max_context_chunks,
    )


def format_indexed_rows(rows: list[list[str | None]], max_rows: int = 30) -> str:
    lines = []
    for row_idx, row in enumerate(rows[:max_rows], 1):
        cells = [f"c{col_idx}={cell}" for col_idx, cell in enumerate(row) if cell not in (None, "")]
        if cells:
            lines.append(f"row {row_idx}: " + " | ".join(cells))
    return "\n".join(lines)


def format_context(chunks: list[dict[str, Any]], table_contexts: list[dict[str, Any]] | None = None) -> str:
    blocks = []
    text_blocks = []
    table_blocks = []

    for chunk in chunks:
        meta = chunk["metadata"]
        header = " > ".join(meta.get("header_path") or [])
        table_refs = meta.get("table_refs") or []
        refs_note = f"\ntable_refs: {table_refs}" if table_refs else ""
        text_blocks.append(
            f"""[Text Context]
chunk_id: {chunk['chunk_id']}
context_role: {chunk.get('context_role', 'selected')}
header: {header}{refs_note}
text:
{chunk.get('context_content', chunk['content'])}"""
        )

    for table in table_contexts or []:
        section_path = table.get("section_path") or " > ".join(part for part in table.get("header_path", []) if part)
        markdown = table.get("markdown") or table.get("raw_text") or ""
        summary = table.get("summary") or ""
        table_blocks.append(
            f"""[Table: {table['table_id']}]
section: {section_path}
pages: {table.get('page_start')}-{table.get('page_end')}
summary: {summary}
markdown:
{markdown[:6000]}
"""
        )

    blocks.extend(text_blocks)
    blocks.extend(table_blocks)
    return "\n\n---\n\n".join(blocks)


def answer_query(query: str, context_chunks: list[dict[str, Any]], table_contexts: list[dict[str, Any]], anthropic_key: str, chat_model: str) -> str:
    table_instructions = ""
    system = "You answer financial filing questions using only the supplied chunks. Do not use outside knowledge."
    if table_contexts:
        system = "You answer financial filing questions using only supplied text and table contexts. Be especially careful reading flattened tables. Do not use outside knowledge."
        table_instructions = """
Table-specific rules:
- Table contexts are the highest-priority evidence for numeric/table-value questions.
- Flattened tables may omit visual column headers; use row labels, nearby labels, row order, and repeated numeric patterns to infer the requested value.
- If the question asks for percentages or ratios and the table provides the numerator and denominator, calculate the percentage or ratio directly and show the formula.
- Before saying context is insufficient, inspect every provided table context and indexed row.
- If a table contains a row label and the requested value/column can be inferred, answer directly.
- For questions asking "according to the table" or asking a total/subtotal, cite the table id.
- If only a narrower table-specific total is available, state that total clearly before noting any broader limitation.
"""
    prompt = f"""Question:
{query}

Use only the context below. If the context is insufficient, say briefly that the provided context does not support a confident answer.
When useful, cite chunk ids in parentheses.
If table context is provided, you may use it for table values and cite table ids.
Do not infer management intent, causal drivers, or qualitative judgment unless the context explicitly states it.
If you compute or compare values from context, label that as a calculation rather than something the filing says.

Answer style:
- Be direct and concise. Answer only what the question asks.
- Do not enumerate unrelated items from neighboring chunks.
- Separate "disclosed by the filing" from "calculated from the provided data" when both appear.
{table_instructions}

Context:
{format_context(context_chunks, table_contexts)}
"""
    return call_chat(
        prompt,
        system=system,
        max_tokens=900,
        chat_model=chat_model,
    )


def run_pipeline(
    query: str,
    db_path: Path,
    vector_top_k: int,
    rerank_top_n: int,
    max_context_chunks: int,
    chat_model: str,
    embed_model: str,
    assets_path: Path | None = DEFAULT_ASSETS,
    table_db_path: Path | None = DEFAULT_TABLE_DB,
    table_vector_top_k: int = 5,
    table_similarity_threshold: float = DEFAULT_TABLE_SIMILARITY_THRESHOLD,
    bm25_top_k: int = 10,
    rerank_model: str | None = None,
    ticker_filter: str | list[str] | None = None,
    fiscal_year_filter: str | list[str] | None = None,
    retrieval_query: str | None = None,
    enable_retrieval_fallback: bool = True,
    retrieval_rerank_threshold: float = DEFAULT_RETRIEVAL_RERANK_THRESHOLD,
    retrieval_fallback_max_attempts: int = 1,
    enable_context_sufficiency_check: bool = False,
    sufficiency_model: str | None = None,
    min_table_similarity_for_confidence: float = 0.60,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    load_env_file()
    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    if not fireworks_key:
        raise RuntimeError("FIREWORKS_API_KEY is required for query embeddings.")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        chat_provider, _ = resolve_chat_model()
    else:
        chat_provider = "anthropic"
    if chat_model in (DEFAULT_CHAT_MODEL, None, ""):
        chat_model = resolve_answer_model()
    if not rerank_model:
        rerank_model = resolve_rerank_model()
    rerank_backend = resolve_rerank_backend()
    search_query = retrieval_query or query
    fallback_trace: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    chunks = load_chunks(db_path, ticker_filter=ticker_filter, fiscal_year_filter=fiscal_year_filter)
    table_chunks = (
        load_chunks(table_db_path, ticker_filter=ticker_filter, fiscal_year_filter=fiscal_year_filter)
        if table_db_path and table_db_path.exists()
        else []
    )
    table_lookup = load_table_lookup(assets_path)
    timings["load_sec"] = time.perf_counter() - t0

    def retrieve_attempt(attempt_query: str) -> dict[str, Any]:
        t_search = time.perf_counter()
        attempt_embedding = embed_texts_fireworks([attempt_query], model=embed_model, api_key=fireworks_key)[0]
        attempt_vector_hits = vector_search_with_embedding(attempt_embedding, chunks, top_k=vector_top_k)
        attempt_bm25_hits = bm25_search(attempt_query, chunks, top_k=bm25_top_k)
        attempt_text_hits = merge_vector_and_bm25_hits(attempt_vector_hits, attempt_bm25_hits)
        attempt_table_hits = vector_search_table_summaries_with_embedding(
            attempt_embedding,
            table_chunks,
            top_k=table_vector_top_k,
        )
        attempt_filtered_table_hits = filter_table_hits_by_threshold(attempt_table_hits, table_similarity_threshold)
        timings["vector_search_sec"] = timings.get("vector_search_sec", 0.0) + (time.perf_counter() - t_search)

        t_rerank = time.perf_counter()
        attempt_text_anchors = rerank_text_chunks(
            attempt_query,
            attempt_text_hits,
            rerank_model=rerank_model,
            top_n=rerank_top_n,
            fireworks_key=fireworks_key,
        )
        timings["rerank_sec"] = timings.get("rerank_sec", 0.0) + (time.perf_counter() - t_rerank)
        attempt_confidence = retrieval_confidence(
            attempt_text_anchors,
            attempt_filtered_table_hits,
            retrieval_rerank_threshold=retrieval_rerank_threshold,
            min_table_similarity_for_confidence=min_table_similarity_for_confidence,
        )
        return {
            "search_query": attempt_query,
            "query_embedding": attempt_embedding,
            "vector_hits": attempt_vector_hits,
            "bm25_hits": attempt_bm25_hits,
            "text_hits": attempt_text_hits,
            "table_hits": attempt_table_hits,
            "filtered_table_hits": attempt_filtered_table_hits,
            "text_anchors": attempt_text_anchors,
            "retrieval_confidence": attempt_confidence,
        }

    attempt = retrieve_attempt(search_query)
    initial_confidence = attempt["retrieval_confidence"]
    if (
        enable_retrieval_fallback
        and retrieval_query is None
        and retrieval_fallback_max_attempts > 0
        and initial_confidence.get("should_retry")
    ):
        t0 = time.perf_counter()
        rewrite_model = (
            os.environ.get("ANTHROPIC_RERANK_MODEL", DEFAULT_ANTHROPIC_RERANK_MODEL)
            if anthropic_key
            else chat_model
        )
        retry_query = rewrite_retrieval_query(query, chat_model=rewrite_model)
        timings["retrieval_rewrite_sec"] = time.perf_counter() - t0
        retry_attempt = retrieve_attempt(retry_query)

        def confidence_value(confidence: dict[str, Any]) -> float:
            return max(
                float(confidence.get("text_top1_rerank_score") or 0.0),
                float(confidence.get("table_top1_similarity") or 0.0),
            )

        selected_retry = confidence_value(retry_attempt["retrieval_confidence"]) >= confidence_value(initial_confidence)
        fallback_trace.append(
            {
                "reason": "low_retrieval_confidence",
                "original_query": query,
                "initial_retrieval_query": search_query,
                "retry_retrieval_query": retry_query,
                "initial_confidence": initial_confidence,
                "retry_confidence": retry_attempt["retrieval_confidence"],
                "selected": "retry" if selected_retry else "initial",
            }
        )
        if selected_retry:
            attempt = retry_attempt

    search_query = attempt["search_query"]
    vector_hits = attempt["vector_hits"]
    bm25_hits = attempt["bm25_hits"]
    text_hits = attempt["text_hits"]
    table_hits = attempt["table_hits"]
    filtered_table_hits = attempt["filtered_table_hits"]
    text_anchors = attempt["text_anchors"]
    retrieval_confidence_info = attempt["retrieval_confidence"]

    t0 = time.perf_counter()
    expanded, table_contexts = assemble_dual_path_context(
        text_anchors,
        filtered_table_hits,
        chunks,
        table_lookup,
        max_context_chunks=max_context_chunks,
    )
    timings["context_expansion_sec"] = time.perf_counter() - t0

    sufficiency_check: dict[str, Any] | None = None
    if enable_context_sufficiency_check:
        if sufficiency_model is None:
            sufficiency_model = (
                os.environ.get("ANTHROPIC_RERANK_MODEL", DEFAULT_ANTHROPIC_RERANK_MODEL)
                if anthropic_key
                else chat_model
            )
        t0 = time.perf_counter()
        sufficiency_check = context_sufficiency_check(
            query,
            expanded,
            table_contexts,
            chat_model=sufficiency_model,
        )
        timings["sufficiency_check_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if sufficiency_check is not None and not sufficiency_check.get("sufficient"):
        answer = "The provided filing context does not contain enough support to answer this question confidently."
    else:
        answer = answer_query(query, expanded, table_contexts, anthropic_key="", chat_model=chat_model)
    timings["answer_sec"] = time.perf_counter() - t0
    timings["total_sec"] = time.perf_counter() - total_start
    return {
        "query": query,
        "retrieval_query": search_query,
        "retrieval_confidence": retrieval_confidence_info,
        "fallback_trace": fallback_trace,
        "sufficiency_check": sufficiency_check,
        "settings": {
            "pipeline": "dual_path_hybrid_text_rerank_plus_table_threshold",
            "rerank_backend": rerank_backend,
            "vector_top_k": vector_top_k,
            "bm25_top_k": bm25_top_k,
            "table_vector_top_k": table_vector_top_k,
            "table_similarity_threshold": table_similarity_threshold,
            "enable_retrieval_fallback": enable_retrieval_fallback,
            "retrieval_rerank_threshold": retrieval_rerank_threshold,
            "retrieval_fallback_max_attempts": retrieval_fallback_max_attempts,
            "enable_context_sufficiency_check": enable_context_sufficiency_check,
            "sufficiency_model": sufficiency_model,
            "min_table_similarity_for_confidence": min_table_similarity_for_confidence,
            "rerank_top_n": rerank_top_n,
            "max_context_chunks": max_context_chunks,
            "ticker_filter": normalize_filter_values(ticker_filter, kind="ticker"),
            "fiscal_year_filter": normalize_filter_values(fiscal_year_filter, kind="fiscal_year"),
            "filtered_chunk_count": len(chunks),
            "filtered_table_chunk_count": len(table_chunks),
            "chat_provider": chat_provider,
            "chat_model": chat_model,
            "answer_model": chat_model,
            "rerank_model": rerank_model,
            "embed_provider": "fireworks",
            "embed_model": embed_model,
            "db_path": str(db_path),
            "assets_path": str(assets_path) if assets_path else None,
            "table_db_path": str(table_db_path) if table_db_path else None,
        },
        "latency": {key: round(value, 3) for key, value in timings.items()},
        "vector_hits": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "candidate_type": hit["candidate_type"],
                "candidate_id": hit["candidate_id"],
                "header_path": hit["header_path"],
            }
            for hit in vector_hits
        ],
        "bm25_hits": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "candidate_type": hit["candidate_type"],
                "candidate_id": hit["candidate_id"],
                "header_path": hit["header_path"],
            }
            for hit in bm25_hits
        ],
        "text_retrieval_hits": [
            {
                "rank": hit["rank"],
                "candidate_id": hit["candidate_id"],
                "vector_score": hit.get("vector_score"),
                "bm25_score": hit.get("bm25_score"),
                "retrieval_sources": hit.get("retrieval_sources") or [],
                "header_path": hit["header_path"],
            }
            for hit in text_hits
        ],
        "table_vector_hits": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "table_id": hit.get("table_id"),
                "header_path": hit["header_path"],
                "section_ref": (hit.get("metadata") or {}).get("section_ref"),
                "summary": (hit.get("metadata") or {}).get("summary"),
                "passed_threshold": float(hit.get("score") or 0.0) >= table_similarity_threshold,
            }
            for hit in table_hits
        ],
        "table_vector_hits_filtered": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "table_id": hit.get("table_id"),
                "header_path": hit["header_path"],
                "section_ref": (hit.get("metadata") or {}).get("section_ref"),
                "summary": (hit.get("metadata") or {}).get("summary"),
            }
            for hit in filtered_table_hits
        ],
        "text_reranked_top": [
            {
                "candidate_type": "text",
                "candidate_id": hit.get("candidate_id"),
                "chunk_id": hit.get("chunk_id"),
                "vector_rank": hit.get("vector_rank") or hit.get("rank"),
                "vector_score": hit.get("vector_score", hit.get("score")),
                "bm25_score": hit.get("bm25_score"),
                "retrieval_sources": hit.get("retrieval_sources") or [],
                "rerank_score": hit.get("rerank_score"),
                "header_path": hit["header_path"],
                "rerank_reason": hit.get("rerank_reason", ""),
                "source_file": (hit.get("metadata") or {}).get("source_file"),
                "excerpt": reference_excerpt(hit.get("content") or ""),
            }
            for hit in text_anchors
        ],
        "reranked_top": [
            {
                "candidate_type": "text",
                "candidate_id": hit.get("candidate_id"),
                "chunk_id": hit.get("chunk_id"),
                "vector_rank": hit.get("vector_rank") or hit.get("rank"),
                "vector_score": hit.get("vector_score", hit.get("score")),
                "bm25_score": hit.get("bm25_score"),
                "retrieval_sources": hit.get("retrieval_sources") or [],
                "rerank_score": hit.get("rerank_score"),
                "header_path": hit["header_path"],
                "rerank_reason": hit.get("rerank_reason", ""),
                "source_file": (hit.get("metadata") or {}).get("source_file"),
                "excerpt": reference_excerpt(hit.get("content") or ""),
            }
            for hit in text_anchors
        ],
        "expanded_context": [
            {
                "chunk_id": chunk["chunk_id"],
                "context_role": chunk.get("context_role"),
                "header_path": chunk["metadata"].get("header_path") or [],
                "section_ref_id": chunk["metadata"].get("section_ref_id"),
                "text_unit_kind": chunk["metadata"].get("text_unit_kind"),
                "token_count": chunk["metadata"].get("token_count"),
                "table_refs": chunk["metadata"].get("table_refs") or [],
                "source_file": chunk["metadata"].get("source_file"),
                "excerpt": reference_excerpt(chunk.get("context_content") or chunk.get("content") or ""),
            }
            for chunk in expanded
        ],
        "table_contexts": [
            {
                "table_id": table["table_id"],
                "source_kind": table.get("source_kind"),
                "source_id": table.get("source_id"),
                "header_path": [part for part in table.get("header_path", []) if part],
                "section_path": table.get("section_path"),
                "summary": table.get("summary"),
                "page_start": table.get("page_start"),
                "page_end": table.get("page_end"),
                "has_markdown": bool(table.get("markdown")),
            }
            for table in table_contexts
        ],
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text vector RAG inference with top10 retrieval, rerank top3, and context expansion.")
    parser.add_argument("query")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--table-db", type=Path, default=DEFAULT_TABLE_DB)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vector-top-k", type=int, default=10)
    parser.add_argument("--bm25-top-k", type=int, default=10)
    parser.add_argument("--table-vector-top-k", type=int, default=5)
    parser.add_argument("--table-similarity-threshold", type=float, default=DEFAULT_TABLE_SIMILARITY_THRESHOLD)
    parser.add_argument("--rerank-top-n", type=int, default=3)
    parser.add_argument("--max-context-chunks", type=int, default=14)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--ticker", action="append", default=[], help="Restrict retrieval to ticker(s), e.g. MSFT.")
    parser.add_argument("--fiscal-year", action="append", default=[], help="Restrict retrieval to fiscal year(s), e.g. FY2025.")
    args = parser.parse_args()

    result = run_pipeline(
        args.query,
        db_path=args.db,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_n=args.rerank_top_n,
        max_context_chunks=args.max_context_chunks,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        assets_path=args.assets,
        table_db_path=args.table_db,
        table_vector_top_k=args.table_vector_top_k,
        table_similarity_threshold=args.table_similarity_threshold,
        ticker_filter=args.ticker or None,
        fiscal_year_filter=args.fiscal_year or None,
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"answer": result["answer"], "reranked_top": result["reranked_top"], "expanded_context": result["expanded_context"]}, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

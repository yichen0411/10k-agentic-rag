#!/usr/bin/env python3
"""Generate multi-chunk questions and measure whether retrieval finds the source chunk set."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_text_vector_db import DEFAULT_EMBED_MODEL, embed_texts_fireworks

DEFAULT_DB = ROOT / "data" / "index" / "text_chunks" / "vectors.db"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "AAPL_FY2025_cross_chunk_hit_eval.json"
DEFAULT_CHAT_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def load_env_file(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def vector_from_blob(blob: bytes) -> list[float]:
    return list(array("f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-12
    nb = math.sqrt(sum(y * y for y in b)) + 1e-12
    return dot / (na * nb)


def call_anthropic(prompt: str, api_key: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": (
                "You generate retrieval evaluation questions. "
                "Return JSON only, with keys question and rationale."
            ),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 420,
        }
    ).encode("utf-8")
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
    if result.returncode != 0:
        error_body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
        raise RuntimeError(f"Anthropic call failed: {error_body[:500]}")
    data = json.loads(result.stdout.decode("utf-8"))
    blocks = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    text = "\n".join(blocks).strip()
    if not text:
        raise RuntimeError(f"Anthropic response missing text: {json.dumps(data)[:500]}")
    return text


def parse_json_response(text: str) -> dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            question_match = re.search(r'"question"\s*:\s*"([^"]+)', text, flags=re.S)
            if question_match:
                return {"question": question_match.group(1).strip(), "rationale": "JSON response was truncated."}
            return {"question": text.splitlines()[0].strip(), "rationale": "Non-JSON response."}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            question_match = re.search(r'"question"\s*:\s*"([^"]+)', text[start : end + 1], flags=re.S)
            if question_match:
                return {"question": question_match.group(1).strip(), "rationale": "JSON response was truncated."}
            return {"question": text.splitlines()[0].strip(), "rationale": "Malformed JSON response."}
    return {"question": str(data.get("question", "")).strip(), "rationale": str(data.get("rationale", "")).strip()}


def load_chunks(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, chunk_id, content, embedding, ticker, fiscal_year, section, source_file, metadata_json
        FROM chunks
        ORDER BY id
        """
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


def make_groups(chunks: list[dict[str, Any]], n: int, max_group_size: int, seed: int) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: list[list[dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(groups) < n and attempts < n * 200:
        attempts += 1
        size = rng.randint(2, max_group_size)
        start = rng.randint(0, len(chunks) - size)
        group = chunks[start : start + size]
        sections = {chunk["section"] for chunk in group}
        # Keep groups local enough that a real cross-chunk question is coherent.
        if len(sections) > 2:
            continue
        key = tuple(chunk["chunk_id"] for chunk in group)
        if key in seen:
            continue
        seen.add(key)
        groups.append(group)
    if len(groups) < n:
        raise RuntimeError(f"Could only build {len(groups)} groups")
    return groups


def make_question(group: list[dict[str, Any]], api_key: str, model: str) -> dict[str, str]:
    chunks_text = []
    for idx, chunk in enumerate(group, 1):
        header = " > ".join(chunk["metadata"].get("header_path") or [])
        chunks_text.append(f"Chunk {idx} header: {header}\nChunk {idx} text:\n{chunk['content'][:1800]}")
    joined_chunks = "\n\n---\n\n".join(chunks_text)
    prompt = f"""Write one natural-language question that requires combining information from at least 2 of the chunks below.

Rules:
- The question must be answerable using only these chunks.
- Do not ask for exact table values unless they appear in the text.
- Make the question specific enough that retrieval should need this local chunk group.
- Return JSON only: {{"question": "...", "rationale": "which chunks are needed and why"}}

{joined_chunks}
"""
    return parse_json_response(call_anthropic(prompt, api_key=api_key, model=model))


def search(query_embedding: list[float], chunks: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    scored = [(cosine(query_embedding, chunk["embedding"]), chunk) for chunk in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "rank": idx,
            "score": score,
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "header_path": chunk["metadata"].get("header_path") or [],
        }
        for idx, (score, chunk) in enumerate(scored[:top_k], 1)
    ]


def target_metrics(target_ids: set[str], retrieved: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top_ids = [row["chunk_id"] for row in retrieved[:k]]
    found = target_ids.intersection(top_ids)
    return {
        f"target_recall_at_{k}": len(found) / len(target_ids),
        f"all_targets_hit_at_{k}": len(found) == len(target_ids),
        f"any_target_hit_at_{k}": bool(found),
        f"found_targets_at_{k}": sorted(found),
    }


def summarize(results: list[dict[str, Any]], db_path: Path, seed: int, max_group_size: int, chat_model: str, embed_model: str) -> dict[str, Any]:
    summary = {
        "n": len(results),
        "seed": seed,
        "max_group_size": max_group_size,
        "db_path": str(db_path),
        "chat_model": chat_model,
        "embed_model": embed_model,
    }
    if not results:
        return summary
    for k in [3, 5, 10, 20]:
        summary[f"mean_target_recall_at_{k}"] = sum(r[f"target_recall_at_{k}"] for r in results) / len(results)
        summary[f"all_targets_hit_at_{k}"] = sum(r[f"all_targets_hit_at_{k}"] for r in results) / len(results)
        summary[f"any_target_hit_at_{k}"] = sum(r[f"any_target_hit_at_{k}"] for r in results) / len(results)
    return summary


def evaluate(db_path: Path, output_path: Path, n: int, seed: int, max_group_size: int, chat_model: str, embed_model: str) -> dict[str, Any]:
    load_env_file()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required.")
    if not fireworks_key:
        raise RuntimeError("FIREWORKS_API_KEY is required.")

    chunks = load_chunks(db_path)
    groups = make_groups(chunks, n=n, max_group_size=max_group_size, seed=seed)
    results = []
    for idx, group in enumerate(groups, 1):
        target_ids = {chunk["chunk_id"] for chunk in group}
        print(f"[{idx}/{len(groups)}] group_size={len(group)} targets={list(target_ids)[:2]}...", flush=True)
        question_data = make_question(group, api_key=anthropic_key, model=chat_model)
        question = question_data["question"]
        query_embedding = embed_texts_fireworks([question], model=embed_model, api_key=fireworks_key)[0]
        retrieved = search(query_embedding, chunks, top_k=20)
        metrics = {}
        for k in [3, 5, 10, 20]:
            metrics.update(target_metrics(target_ids, retrieved, k))
        results.append(
            {
                "target_chunk_ids": [chunk["chunk_id"] for chunk in group],
                "group_size": len(group),
                "sections": [chunk["section"] for chunk in group],
                "header_paths": [chunk["metadata"].get("header_path") or [] for chunk in group],
                "question": question,
                "rationale": question_data["rationale"],
                **metrics,
                "top_results": retrieved,
            }
        )
        partial_payload = {
            "summary": summarize(results, db_path, seed, max_group_size, chat_model, embed_model),
            "results": results,
            "partial": len(results) < len(groups),
        }
        output_path.write_text(json.dumps(partial_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize(results, db_path, seed, max_group_size, chat_model, embed_model)
    payload = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector retrieval for generated cross-chunk questions.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-group-size", type=int, default=5)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    payload = evaluate(args.db, args.output, args.n, args.seed, args.max_group_size, args.chat_model, args.embed_model)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

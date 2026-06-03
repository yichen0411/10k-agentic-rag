#!/usr/bin/env python3
"""Generate chunk-grounded questions and measure vector retrieval hit rate."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import random
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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "AAPL_FY2025_text_vector_hit_eval.json"
DEFAULT_CHAT_MODEL = "claude-haiku-4-5-20251001"
FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
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


def call_fireworks_chat(prompt: str, api_key: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate retrieval evaluation questions. "
                        "Return only one concise question, no preamble."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 160,
        }
    ).encode("utf-8")
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--fail-with-body",
            FIREWORKS_CHAT_URL,
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
        raise RuntimeError(f"Fireworks chat failed: {error_body[:500]}")
    data = json.loads(result.stdout.decode("utf-8"))
    message = data["choices"][0].get("message") or {}
    content = message.get("content") or data["choices"][0].get("text")
    if not content:
        raise RuntimeError(f"Fireworks chat response did not include content: {json.dumps(data)[:500]}")
    return content.strip().strip('"')


def call_anthropic_chat(prompt: str, api_key: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": (
                "You generate retrieval evaluation questions. "
                "Return only one concise question, no preamble."
            ),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 120,
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
        raise RuntimeError(f"Anthropic chat failed: {error_body[:500]}")
    data = json.loads(result.stdout.decode("utf-8"))
    text_blocks = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    content = "\n".join(text_blocks).strip()
    if not content:
        raise RuntimeError(f"Anthropic response did not include text content: {json.dumps(data)[:500]}")
    return content.strip().strip('"')


def load_chunks(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT chunk_id, content, embedding, ticker, fiscal_year, section, source_file, metadata_json
        FROM chunks
        ORDER BY id
        """
    ).fetchall()
    conn.close()
    return [
        {
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


def make_question(chunk: dict[str, Any], api_key: str, model: str, provider: str) -> str:
    metadata = chunk["metadata"]
    header = " > ".join(metadata.get("header_path") or [])
    prompt = f"""Write one natural-language question whose answer is contained in the chunk below.

Rules:
- The question must be answerable using only this chunk.
- Do not ask for exact table values unless they appear in the text.
- Prefer specific wording from the chunk so retrieval can be evaluated.
- Return only the question.

Header: {header}

Chunk:
{chunk["content"][:3000]}
"""
    if provider == "anthropic":
        return call_anthropic_chat(prompt, api_key=api_key, model=model)
    if provider == "fireworks":
        return call_fireworks_chat(prompt, api_key=api_key, model=model)
    raise ValueError(f"Unsupported question provider: {provider}")


def search(query_embedding: list[float], chunks: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    scored = []
    for chunk in chunks:
        scored.append((cosine(query_embedding, chunk["embedding"]), chunk))
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


def load_excluded_chunk_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            chunk_id = result.get("target_chunk_id")
            if chunk_id:
                excluded.add(chunk_id)
    return excluded


def evaluate(
    db_path: Path,
    output_path: Path,
    n: int,
    seed: int,
    chat_model: str,
    embed_model: str,
    question_provider: str,
    exclude_results: list[Path] | None = None,
) -> dict[str, Any]:
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
    if not fireworks_api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required.")
    question_api_key = os.environ.get("ANTHROPIC_API_KEY") if question_provider == "anthropic" else fireworks_api_key
    if not question_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when --question-provider=anthropic.")

    chunks = load_chunks(db_path)
    excluded_ids = load_excluded_chunk_ids(exclude_results or [])
    candidate_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in excluded_ids]
    rng = random.Random(seed)
    sample = rng.sample(candidate_chunks, min(n, len(candidate_chunks)))

    results = []
    for idx, chunk in enumerate(sample, 1):
        print(f"[{idx}/{len(sample)}] generating question for {chunk['chunk_id']}", flush=True)
        question = make_question(chunk, api_key=question_api_key, model=chat_model, provider=question_provider)
        query_embedding = embed_texts_fireworks([question], model=embed_model, api_key=fireworks_api_key)[0]
        retrieved = search(query_embedding, chunks, top_k=10)
        ranks = [hit["rank"] for hit in retrieved if hit["chunk_id"] == chunk["chunk_id"]]
        rank = ranks[0] if ranks else None
        results.append(
            {
                "target_chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "header_path": chunk["metadata"].get("header_path") or [],
                "question": question,
                "target_rank": rank,
                "hit_at_1": rank is not None and rank <= 1,
                "hit_at_3": rank is not None and rank <= 3,
                "hit_at_5": rank is not None and rank <= 5,
                "hit_at_10": rank is not None and rank <= 10,
                "top_results": retrieved,
            }
        )

    summary = {
        "n": len(results),
        "seed": seed,
        "db_path": str(db_path),
        "excluded_chunks": len(excluded_ids),
        "candidate_chunks": len(candidate_chunks),
        "question_provider": question_provider,
        "chat_model": chat_model,
        "embed_model": embed_model,
        "hit_at_1": sum(r["hit_at_1"] for r in results) / len(results),
        "hit_at_3": sum(r["hit_at_3"] for r in results) / len(results),
        "hit_at_5": sum(r["hit_at_5"] for r in results) / len(results),
        "hit_at_10": sum(r["hit_at_10"] for r in results) / len(results),
        "misses": sum(1 for r in results if not r["hit_at_10"]),
    }
    payload = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text vector hit rate with generated chunk-grounded questions.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--question-provider", choices=["anthropic", "fireworks"], default="anthropic")
    parser.add_argument("--exclude-results", type=Path, action="append", default=[], help="Existing eval JSON whose target chunks should be excluded.")
    args = parser.parse_args()

    load_env_file()
    payload = evaluate(
        args.db,
        args.output,
        args.n,
        args.seed,
        args.chat_model,
        args.embed_model,
        args.question_provider,
        args.exclude_results,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

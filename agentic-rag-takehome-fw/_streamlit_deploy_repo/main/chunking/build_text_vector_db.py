#!/usr/bin/env python3
"""Embed text-only RAG chunks and store them in the local vector DB."""

from __future__ import annotations

import argparse
from array import array
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNKING_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "index" / "text_chunks"
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FIREWORKS_EMBEDDINGS_URL = "https://api.fireworks.ai/inference/v1/embeddings"
TABLE_MARKER_RE = re.compile(r"\[\[TABLE:[^\]]+\]\]")

from filing_metadata import filing_identity, parse_source_id  # noqa: E402


def text_for_embedding(text: str) -> str:
    return re.sub(r"\s+", " ", TABLE_MARKER_RE.sub(" ", text or "")).strip()


def load_text_chunks(paths: list[Path]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    chunk_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_file, ticker, fiscal_year = filing_identity(
            payload.get("source_file") or path.name,
            original_filename=payload.get("original_filename"),
            json_path=path,
        )

        for chunk in payload.get("chunks", []):
            if chunk.get("chunk_type") != "text":
                continue
            text = chunk.get("text") or ""
            if not text.strip():
                continue

            unique_id = f"{source_file}::{chunk['chunk_id']}"
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "vector_chunk_id": unique_id,
                "source_json": str(path),
                "source_file": source_file,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "section": chunk.get("section_title"),
                "section_ref_id": chunk.get("section_ref_id"),
                "subsection_ref_id": chunk.get("subsection_ref_id"),
                "text_unit_id": chunk.get("text_unit_id"),
                "text_unit_kind": chunk.get("text_unit_kind"),
                "header_path": chunk.get("header_path") or [],
                "table_refs": chunk.get("table_refs") or [],
                "table_anchors": chunk.get("table_anchors") or [],
                "image_refs": chunk.get("image_refs") or [],
                "split_index": chunk.get("split_index"),
                "split_count": chunk.get("split_count"),
                "is_split_continuation": chunk.get("is_split_continuation"),
                "same_text_unit_prev_chunk_id": chunk.get("same_text_unit_prev_chunk_id"),
                "same_text_unit_next_chunk_id": chunk.get("same_text_unit_next_chunk_id"),
                "same_section_prev_chunk_id": chunk.get("same_section_prev_chunk_id"),
                "same_section_next_chunk_id": chunk.get("same_section_next_chunk_id"),
                "same_text_unit_prev_vector_chunk_id": qualify_chunk_ref(source_file, chunk.get("same_text_unit_prev_chunk_id")),
                "same_text_unit_next_vector_chunk_id": qualify_chunk_ref(source_file, chunk.get("same_text_unit_next_chunk_id")),
                "same_section_prev_vector_chunk_id": qualify_chunk_ref(source_file, chunk.get("same_section_prev_chunk_id")),
                "same_section_next_vector_chunk_id": qualify_chunk_ref(source_file, chunk.get("same_section_next_chunk_id")),
                "neighbor_expansion_scope": chunk.get("neighbor_expansion_scope"),
                "cross_section_expansion_allowed": chunk.get("cross_section_expansion_allowed"),
                "token_count": chunk.get("token_count"),
                "chunk_type": "text",
            }
            chunk_ids.append(unique_id)
            contents.append(text_for_embedding(text))
            metadatas.append(metadata)

    return chunk_ids, contents, metadatas


def qualify_chunk_ref(source_file: str, chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    if "::text_" in chunk_id:
        return chunk_id
    return f"{source_file}::{chunk_id}"


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    if inputs:
        return sorted(path for path in inputs if path.exists())
    return sorted(CHUNKING_DIR.glob("*_rag_chunks.json"))


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            ticker TEXT,
            fiscal_year TEXT,
            item TEXT,
            section TEXT,
            source_file TEXT,
            page_start INTEGER,
            page_end INTEGER,
            parent_chunk_id TEXT,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_year ON chunks(ticker, fiscal_year)")
    conn.commit()


def embedding_to_blob(embedding: list[float]) -> bytes:
    return array("f", embedding).tobytes()


def embed_texts_fireworks(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    # Fireworks currently returns Cloudflare 403s for urllib from this local
    # environment, while curl succeeds with the same key and payload.
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--fail-with-body",
            FIREWORKS_EMBEDDINGS_URL,
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
        raise RuntimeError(f"Fireworks embeddings failed: {error_body[:500]}")
    data = json.loads(result.stdout.decode("utf-8"))
    return [row["embedding"] for row in data["data"]]


def upsert_batch(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
    contents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    rows = []
    for chunk_id, content, embedding, metadata in zip(chunk_ids, contents, embeddings, metadatas):
        rows.append(
            (
                chunk_id,
                content,
                embedding_to_blob(embedding),
                metadata.get("ticker"),
                metadata.get("fiscal_year"),
                metadata.get("item"),
                metadata.get("section"),
                metadata.get("source_file"),
                metadata.get("page_start"),
                metadata.get("page_end"),
                metadata.get("parent_chunk_id"),
                json.dumps(metadata, ensure_ascii=False),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks (
            chunk_id, content, embedding, ticker, fiscal_year, item, section,
            source_file, page_start, page_end, parent_chunk_id, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def build(
    inputs: list[Path],
    out_dir: Path,
    rebuild: bool = True,
    batch_size: int = 64,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> None:
    paths = resolve_inputs(inputs)
    if not paths:
        raise FileNotFoundError("No *_rag_chunks.json files found.")

    print("Loading text chunks from:")
    for path in paths:
        print(f"  - {path}")
    chunk_ids, contents, metadatas = load_text_chunks(paths)
    print(f"Loaded {len(chunk_ids)} text chunks")

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "vectors.db"
    conn = sqlite3.connect(db_path)
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.commit()
    create_schema(conn)

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks embeddings.")

    print(f"Embedding via Fireworks model: {embed_model}")
    embeddings: list[list[float]] = []
    for start in range(0, len(contents), batch_size):
        end = min(start + batch_size, len(contents))
        print(f"  embedding {start + 1}-{end} / {len(contents)}")
        embeddings.extend(embed_texts_fireworks(contents[start:end], model=embed_model, api_key=api_key))

    if len(embeddings) != len(chunk_ids):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(chunk_ids)}")

    print(f"Writing vectors to {db_path}...")
    for start in range(0, len(chunk_ids), 200):
        end = min(start + 200, len(chunk_ids))
        upsert_batch(
            conn,
            chunk_ids[start:end],
            contents[start:end],
            embeddings[start:end],
            metadatas[start:end],
        )

    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    print(f"Done. {count} vectors in {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector DB from text-only RAG chunks.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="RAG chunk JSON file. Repeat for multiple files.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output index directory containing vectors.db.")
    parser.add_argument("--no-rebuild", action="store_true", help="Upsert without clearing existing vectors.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Fireworks embedding model name.")
    args = parser.parse_args()

    build(args.input, args.out, rebuild=not args.no_rebuild, batch_size=args.batch_size, embed_model=args.model)


if __name__ == "__main__":
    main()

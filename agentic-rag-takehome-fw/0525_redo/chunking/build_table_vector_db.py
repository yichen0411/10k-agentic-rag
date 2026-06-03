#!/usr/bin/env python3
"""Stage 2: embed VLM table summaries into a separate vector DB."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNKING_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "index" / "table_summaries"
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

from build_text_vector_db import (  # noqa: E402
    create_schema,
    embed_texts_fireworks,
    embedding_to_blob,
    upsert_batch,
)
from filing_metadata import filing_identity  # noqa: E402
from vlm_table_parse import compose_table_summary, denoise_context_sentence, section_ref_label, table_summary_topic  # noqa: E402


def table_label_terms_from_markdown(markdown: str, *, max_labels: int = 10) -> list[str]:
    labels: list[str] = []
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|") if cell.strip()]
        if not cells:
            continue
        label = denoise_context_sentence(cells[0])
        if not label or label.startswith("("):
            continue
        if label not in labels:
            labels.append(label)
        if len(labels) >= max_labels:
            break
    return labels


def embed_safe_text(text: str) -> str:
    text = re.sub(r"\[(?:num|amount|year)\]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def table_summary_embed_text(table: dict[str, Any], vlm: dict[str, Any]) -> str:
    summary = embed_safe_text(
        vlm.get("summary") or compose_table_summary(table, table_summary_topic(vlm), vlm.get("section_ref"))
    )
    row_labels = [embed_safe_text(label) for label in table_label_terms_from_markdown(vlm.get("markdown") or "")]
    parts = [summary, *row_labels]
    return " ".join(part for part in parts if part).strip()


def load_table_summaries(paths: list[Path]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
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
        for table in payload.get("tables", []):
            vlm = table.get("vlm_parse") or {}
            summary = (vlm.get("summary") or "").strip()
            if vlm.get("status") != "success" or not summary:
                continue
            table_id = table["table_id"]
            unique_id = f"{source_file}::table::{table_id}"
            section = table.get("section_ref") or {}
            subsection = table.get("subsection_ref") or {}
            header_path = [section.get("section_title"), *(subsection.get("path") or [])]
            metadata = {
                "chunk_type": "table_summary",
                "table_id": table_id,
                "vector_chunk_id": unique_id,
                "source_json": str(path),
                "source_file": source_file,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "section": section.get("section_title"),
                "section_ref_id": section.get("section_ref_id"),
                "subsection_ref_id": subsection.get("subsection_ref_id"),
                "header_path": [part for part in header_path if part],
                "section_ref": vlm.get("section_ref") or " > ".join(part for part in header_path if part),
                "page_start": table.get("page_start"),
                "page_end": table.get("page_end"),
                "summary": summary,
                "has_markdown": bool(vlm.get("markdown")),
            }
            chunk_ids.append(unique_id)
            contents.append(table_summary_embed_text(table, vlm))
            metadatas.append(metadata)
    return chunk_ids, contents, metadatas


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    if inputs:
        return sorted(path for path in inputs if path.exists())
    return sorted(CHUNKING_DIR.glob("*_section_assets.json"))


def build(
    inputs: list[Path],
    out_dir: Path,
    rebuild: bool = True,
    batch_size: int = 64,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> None:
    paths = resolve_inputs(inputs)
    if not paths:
        raise FileNotFoundError("No assets JSON files found.")

    chunk_ids, contents, metadatas = load_table_summaries(paths)
    if not chunk_ids:
        raise RuntimeError("No successful vlm_parse summaries found. Run vlm_table_parse.py first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "vectors.db"
    import sqlite3

    conn = sqlite3.connect(db_path)
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.commit()
    create_schema(conn)

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks embeddings.")

    embeddings: list[list[float]] = []
    for start in range(0, len(contents), batch_size):
        end = min(start + batch_size, len(contents))
        embeddings.extend(embed_texts_fireworks(contents[start:end], model=embed_model, api_key=api_key))

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
    print(f"Done. {count} table-summary vectors in {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector DB from VLM table summaries.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="assets.json path(s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()
    build(args.input, args.out, rebuild=not args.no_rebuild, batch_size=args.batch_size, embed_model=args.model)


if __name__ == "__main__":
    main()

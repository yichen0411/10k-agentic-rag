"""PyMuPDF4LLM with Layout (GNN) — pip install pymupdf4llm."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run(pdf_path: Path, pages: set[int]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        import pymupdf4llm
    except ImportError as exc:
        return {
            "adapter": "pymupdf4llm_layout",
            "status": "skipped",
            "error": str(exc),
            "tables": [],
            "latency_sec": round(time.perf_counter() - t0, 3),
            "install_note": "pip install pymupdf4llm  # pulls pymupdf-layout automatically",
            "applicability": "Good markdown/JSON for RAG; layout closed-source; cross-page table linking not guaranteed.",
        }

    tables: list[dict[str, Any]] = []
    page_list = sorted(pages)
    full_text_parts: list[str] = []
    try:
        if hasattr(pymupdf4llm, "to_markdown"):
            md = pymupdf4llm.to_markdown(str(pdf_path), pages=page_list)
            if isinstance(md, list):
                for chunk in md:
                    if isinstance(chunk, dict):
                        full_text_parts.append(chunk.get("text") or "")
                    else:
                        full_text_parts.append(str(chunk))
            else:
                full_text_parts.append(str(md))
        if hasattr(pymupdf4llm, "to_json"):
            payload = pymupdf4llm.to_json(str(pdf_path), pages=page_list)
            if isinstance(payload, str):
                payload = json.loads(payload)
            tables.extend(_tables_from_json(payload, page_list))
        for idx, pipe in enumerate(_extract_pipe_tables("\n".join(full_text_parts))):
            tables.append(
                {
                    "table_id": f"p4llm_pipe_{idx:02d}",
                    "pages": page_list,
                    "cross_page": len(page_list) > 1,
                    "markdown": pipe,
                    "raw_text": pipe,
                }
            )
    except Exception as exc:
        return {
            "adapter": "pymupdf4llm_layout",
            "status": "error",
            "error": str(exc),
            "tables": tables,
            "full_text": "\n".join(full_text_parts),
            "latency_sec": round(time.perf_counter() - t0, 3),
            "install_note": "pip install pymupdf4llm",
            "applicability": "Layout weights not inspectable; evaluate markdown fidelity only.",
        }

    return {
        "adapter": "pymupdf4llm_layout",
        "status": "ok",
        "tables": tables,
        "full_text": "\n".join(full_text_parts),
        "latency_sec": round(time.perf_counter() - t0, 3),
        "install_note": "pip install pymupdf4llm (layout auto-activates on import)",
        "applicability": "Strong layout+markdown; no native Item 7 subsection linking or cross-page group IDs.",
    }


def _tables_from_json(payload: dict[str, Any], pages: list[int]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    page_items = payload.get("pages") or payload.get("page_chunks") or []
    if isinstance(page_items, dict):
        page_items = [{"page": int(k), **v} for k, v in page_items.items()]

    for item in page_items:
        page_no = int(item.get("page") or item.get("page_number") or item.get("metadata", {}).get("page_number") or 0)
        if page_no not in pages:
            continue
        text = item.get("text") or item.get("markdown") or ""
        for idx, block in enumerate(item.get("tables") or []):
            md = block if isinstance(block, str) else block.get("markdown") or block.get("text") or ""
            tables.append(
                {
                    "table_id": f"p{page_no}_p4llm_{idx:02d}",
                    "pages": [page_no],
                    "cross_page": False,
                    "markdown": md,
                    "raw_text": md,
                }
            )
        for idx, pipe in enumerate(_extract_pipe_tables(text)):
            tables.append(
                {
                    "table_id": f"p{page_no}_p4llm_md_{idx:02d}",
                    "pages": [page_no],
                    "cross_page": False,
                    "markdown": pipe,
                    "raw_text": pipe,
                }
            )
    return tables


def _tables_from_markdown(md: str, pages: list[int]) -> list[dict[str, Any]]:
    if isinstance(md, list):
        tables = []
        for chunk in md:
            if isinstance(chunk, dict):
                page_no = int(chunk.get("metadata", {}).get("page_number") or chunk.get("page") or pages[0])
                text = chunk.get("text") or ""
            else:
                page_no, text = pages[0], str(chunk)
            for idx, pipe in enumerate(_extract_pipe_tables(text)):
                tables.append(
                    {
                        "table_id": f"p{page_no}_p4llm_md_{idx:02d}",
                        "pages": [page_no],
                        "cross_page": False,
                        "markdown": pipe,
                        "raw_text": pipe,
                    }
                )
        return tables
    tables = []
    for idx, pipe in enumerate(_extract_pipe_tables(str(md))):
        tables.append(
            {
                "table_id": f"p4llm_md_{idx:02d}",
                "pages": pages,
                "cross_page": len(pages) > 1,
                "markdown": pipe,
                "raw_text": pipe,
            }
        )
    return tables


def _extract_pipe_tables(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in (text or "").splitlines():
        if "|" in line and line.count("|") >= 2:
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks

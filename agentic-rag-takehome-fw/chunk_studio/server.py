"""Small web product for exploring generated 10-K chunks, tables, figures, and QA."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
from chunk_studio.rag_index_utils import vector_db_health, workspace_table_db_path
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import fitz
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
AGENT_DIR = ROOT / "main" / "agent"
CHUNKING_DIR = ROOT / "main" / "chunking"


def _load_agent_memory_module() -> Any:
    """Import main/agent/agent_memory.py."""
    agent_path = str(AGENT_DIR)
    if agent_path not in sys.path:
        sys.path.insert(0, agent_path)
    import agent_memory  # noqa: E402

    return agent_memory


def _load_langsmith_tracing_module() -> Any:
    agent_path = str(AGENT_DIR)
    if agent_path not in sys.path:
        sys.path.insert(0, agent_path)
    import langsmith_tracing  # noqa: E402

    return langsmith_tracing
WORKSPACE_DIR = ROOT / "data" / "chunk_studio"
STATIC_DIR = APP_DIR / "static"
PROCESSING: set[str] = set()
VLM_PARSE_RUNNING: set[str] = set()
METADATA_LOCKS: dict[str, threading.Lock] = {}
METADATA_LOCKS_GUARD = threading.Lock()
PROCESS_STEPS = [
    {
        "key": "sectioning",
        "label": "Reading structure",
        "description": "Parse TOC, locate Item sections, and detect subsections.",
    },
    {
        "key": "extracting_assets",
        "label": "Finding assets",
        "description": "Extract tables and figures, then attach them to nearby sections.",
    },
    {
        "key": "building_chunks",
        "label": "Writing chunks",
        "description": "Create text chunks with header paths, table refs, and image refs.",
    },
]
PROCESS_STEP_INDEX = {step["key"]: idx for idx, step in enumerate(PROCESS_STEPS)}

sys.path[:0] = [str(CHUNKING_DIR), str(ROOT)]

from build_table_vector_db import build as build_table_vector_index  # noqa: E402
from build_text_vector_db import DEFAULT_EMBED_MODEL, build as build_vector_index  # noqa: E402
from rag_chunk_builder import build_rag_payload, build_text_chunks, drop_empty_headers  # noqa: E402
from section_asset_extractor import build_asset_payload  # noqa: E402
from vlm_table_parse import DEFAULT_VLM_DPI, DEFAULT_VLM_MULTI_PAGE_DPI, enrich_assets_with_vlm, load_env_file  # noqa: E402
from toc_guided_section_probe import (  # noqa: E402
    attach_subsections,
    build_line_records,
    build_part_hierarchy,
    build_sections,
    collect_pages,
    detect_toc_pages,
    find_global_headings,
    menu_guided_headings,
    parse_visible_toc,
    summarize_subsection_counts,
)

load_env_file(ROOT / ".env")


def _port() -> int:
    try:
        return int(os.environ.get("PORT", "8010"))
    except ValueError:
        return 8010


def _host() -> str:
    return os.environ.get("HOST", "0.0.0.0")


app = FastAPI(
    title="Chunk Studio",
    description="Upload a 10-K PDF, generate chunks/assets, and visualize the chunk result.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentTraceBody(BaseModel):
    question: str = Field(..., min_length=1)
    max_steps: int = Field(6, ge=1, le=12)
    session_id: Optional[str] = Field(
        None,
        description="Optional session id for multi-turn follow-ups (e.g. shorter).",
    )


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem[:60] or "document"


def _json_path(file_id: str, name: str) -> Path:
    return WORKSPACE_DIR / file_id / name


def _workspace(file_id: str) -> Path:
    path = WORKSPACE_DIR / file_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown file_id: {file_id}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_lock(file_id: str) -> threading.Lock:
    with METADATA_LOCKS_GUARD:
        lock = METADATA_LOCKS.get(file_id)
        if lock is None:
            lock = threading.Lock()
            METADATA_LOCKS[file_id] = lock
        return lock


def _read_metadata_file(path: Path, file_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"file_id": file_id, "status": "unknown"}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {"file_id": file_id, "status": "unknown"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"file_id": file_id, "status": "unknown"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _metadata(file_id: str) -> dict[str, Any]:
    path = _json_path(file_id, "metadata.json")
    with _metadata_lock(file_id):
        meta = _read_metadata_file(path, file_id)
    return _enrich_file_meta(file_id, meta)


def _workspace_table_db_path(workspace: Path) -> Path | None:
    return workspace_table_db_path(workspace)


def _normalize_file_counts(workspace: Path, counts: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure UI-facing count keys (chunks/tables/images) are always populated."""
    c = dict(counts or {})
    if c.get("tables") is None and c.get("tables_after_pipeline") is not None:
        c["tables"] = c["tables_after_pipeline"]
    if c.get("images") is None and c.get("images_detected") is not None:
        c["images"] = c["images_detected"]

    assets_path = workspace / "assets.json"
    assets = _read_json(assets_path) if assets_path.is_file() else None
    if assets:
        if c.get("tables") is None:
            c["tables"] = len(assets.get("tables", []))
        if c.get("images") is None:
            c["images"] = len(assets.get("images", []))
        if c.get("sections") is None:
            c["sections"] = len(assets.get("sections", []))

    chunks_path = workspace / "chunks.json"
    if c.get("chunks") is None and chunks_path.is_file():
        chunks = _read_json(chunks_path)
        c["chunks"] = chunks.get("counts", {}).get("chunks") or len(chunks.get("chunks", []))

    return c


def _vector_db_health(db_path: Path | None) -> dict[str, Any]:
    return vector_db_health(db_path)


def _enrich_file_meta(file_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    workspace = WORKSPACE_DIR / file_id
    text_db = workspace / "index" / "vectors.db"
    table_db = _workspace_table_db_path(workspace)
    text_health = _vector_db_health(text_db)
    table_health = _vector_db_health(table_db)
    has_assets = (workspace / "assets.json").is_file()
    out = dict(meta)
    out["has_vector_index"] = text_health["valid"]
    out["has_table_vector_index"] = table_health["valid"]
    out["vector_index_rows"] = text_health["row_count"]
    out["table_vector_index_rows"] = table_health["row_count"]
    out["vector_index_issue"] = None if text_health["valid"] else text_health.get("reason")
    out["table_vector_index_issue"] = None if table_health["valid"] else table_health.get("reason")
    out["has_assets"] = has_assets
    out["vlm_parse_running"] = file_id in VLM_PARSE_RUNNING
    out["agent_ready"] = out.get("status") == "ready" and text_health["valid"]
    out["counts"] = _normalize_file_counts(workspace, out.get("counts"))
    return out


def _save_metadata(file_id: str, **updates: Any) -> dict[str, Any]:
    path = _json_path(file_id, "metadata.json")
    with _metadata_lock(file_id):
        current = _read_metadata_file(path, file_id)
        current.update(updates)
        current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(path, current)
        return current


def _progress_for(step: str, state: str) -> int:
    if step == "ready":
        return 100
    idx = PROCESS_STEP_INDEX.get(step)
    if idx is None:
        return 0
    base = int((idx / len(PROCESS_STEPS)) * 100)
    if state in {"done", "skipped"}:
        return int(((idx + 1) / len(PROCESS_STEPS)) * 100)
    return min(95, base + 8)


def _record_event(file_id: str, step: str, state: str, message: str, **extra: Any) -> dict[str, Any]:
    current = _metadata(file_id)
    events = current.get("process_events") or []
    label = next((row["label"] for row in PROCESS_STEPS if row["key"] == step), step.replace("_", " ").title())
    events.append(
        {
            "step": step,
            "label": label,
            "state": state,
            "message": message,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        }
    )
    updates = {
        "process_events": events[-80:],
        "current_step": step,
        "current_step_label": label,
        "current_message": message,
        "progress_pct": _progress_for(step, state),
    }
    if state == "error":
        updates["status"] = "error"
        updates["last_error"] = message
    return _save_metadata(file_id, **updates)


def _build_sections_payload(pdf_path: Path, *, source_file: str | None = None) -> dict[str, Any]:
    canonical_source = source_file or pdf_path.name
    doc = fitz.open(pdf_path)
    try:
        pages = collect_pages(doc)
        toc_pages = detect_toc_pages(pages)
        toc_entries = parse_visible_toc(pages, toc_pages)
        full_text, records = build_line_records(pages)
        global_headings = find_global_headings(records)
        menu_guided = menu_guided_headings(toc_entries, records)
        sections = build_sections(full_text, menu_guided)
        attach_subsections(sections, records, full_text)
        return {
            "source_file": canonical_source,
            "toc_pages_detected": toc_pages,
            "method": "visible_toc_menu_then_body_regex_offsets",
            "toc_entry_count": len(toc_entries),
            "global_heading_count": len(global_headings),
            "matched_heading_count": sum(1 for heading in menu_guided if heading["chosen_offset"] is not None),
            "subsection_counts": summarize_subsection_counts(sections),
            "toc_entries": toc_entries,
            "global_headings": global_headings,
            "menu_guided_headings": menu_guided,
            "parts": build_part_hierarchy(sections),
        }
    finally:
        doc.close()


def _empty_assets(source_file: str) -> dict[str, Any]:
    return {"source_file": source_file, "sections": [], "tables": [], "images": []}


def _build_preview_chunks(sections: dict[str, Any]) -> dict[str, Any]:
    assets = _empty_assets(sections["source_file"])
    text_chunks = build_text_chunks(sections, assets)
    chunks = drop_empty_headers(text_chunks)
    return {
        "source_file": sections["source_file"],
        "method": "text_only_rag_chunks_with_header_paths_table_refs",
        "preview": True,
        "counts": {
            "chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "table_chunks": 0,
            "image_chunks": 0,
            "chunks_over_500_tokens": sum(1 for chunk in chunks if chunk["token_count"] > 500),
            "chunks_under_100_tokens": sum(1 for chunk in chunks if chunk["token_count"] < 100),
            "chunks_with_table_refs": 0,
            "chunks_with_image_refs": 0,
        },
        "chunks": chunks,
    }


def _process_file(file_id: str, build_vectors: bool = True) -> dict[str, Any]:
    workspace = _workspace(file_id)
    pdf_path = workspace / "source.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Missing uploaded PDF.")

    source_file = _metadata(file_id).get("original_filename") or pdf_path.name

    _save_metadata(
        file_id,
        status="processing",
        current_step="queued",
        current_step_label="Queued",
        current_message="Waiting for the processing worker to start.",
        progress_pct=0,
        process_plan=PROCESS_STEPS,
        last_error=None,
        process_events=[],
    )
    _record_event(file_id, "queued", "done", "Processing job accepted.")

    started = time.perf_counter()
    _save_metadata(file_id, status="sectioning")
    step_started = time.perf_counter()
    _record_event(file_id, "sectioning", "running", "Parsing TOC and detecting filing sections.")
    sections = _build_sections_payload(pdf_path, source_file=source_file)
    sections_path = workspace / "sections.json"
    _write_json(sections_path, sections)
    _record_event(
        file_id,
        "sectioning",
        "done",
        f"Found {sections.get('matched_heading_count', 0)} matched sections and {sum(sections.get('subsection_counts', {}).values())} subsections.",
        duration_sec=round(time.perf_counter() - step_started, 3),
    )
    preview_chunks = _build_preview_chunks(sections)
    chunks_path = workspace / "chunks.json"
    _write_json(chunks_path, preview_chunks)
    subsection_count = sum(sections.get("subsection_counts", {}).values())
    _save_metadata(
        file_id,
        counts={
            "sections": sections.get("matched_heading_count", 0),
            "subsections": subsection_count,
            "chunks": preview_chunks["counts"]["chunks"],
            "tables": 0,
            "images": 0,
        },
    )

    _save_metadata(file_id, status="extracting_assets")
    step_started = time.perf_counter()
    _record_event(file_id, "extracting_assets", "running", "Extracting and linking tables and figures.")
    assets = build_asset_payload(pdf_path, sections_path)
    assets_path = workspace / "assets.json"
    _write_json(assets_path, assets)
    _record_event(
        file_id,
        "extracting_assets",
        "done",
        f"Detected {len(assets.get('tables', []))} tables and {len(assets.get('images', []))} figures.",
        duration_sec=round(time.perf_counter() - step_started, 3),
    )

    _save_metadata(file_id, status="building_chunks")
    step_started = time.perf_counter()
    _record_event(file_id, "building_chunks", "running", "Building text chunks with table and figure references.")
    chunks = build_rag_payload(sections_path, assets_path)
    chunks["preview"] = False
    _write_json(chunks_path, chunks)
    _record_event(
        file_id,
        "building_chunks",
        "done",
        f"Built {chunks.get('counts', {}).get('chunks', 0)} text chunks.",
        duration_sec=round(time.perf_counter() - step_started, 3),
    )

    vector_status = "skipped"
    vector_error = None
    load_env_file(ROOT / ".env")
    if build_vectors:
        if os.environ.get("FIREWORKS_API_KEY"):
            try:
                _save_metadata(file_id, status="building_vectors")
                step_started = time.perf_counter()
                _record_event(file_id, "building_vectors", "running", "Embedding chunks and writing the vector index.")
                build_vector_index([chunks_path], workspace / "index", rebuild=True, batch_size=64, embed_model=DEFAULT_EMBED_MODEL)
                vector_status = "ready"
                _record_event(
                    file_id,
                    "building_vectors",
                    "done",
                    "Vector index is ready for Q&A.",
                    duration_sec=round(time.perf_counter() - step_started, 3),
                )
            except Exception as exc:  # Keep the visualizer usable even if embeddings fail.
                vector_status = "failed"
                vector_error = str(exc)
                _record_event(file_id, "building_vectors", "error", vector_error)
        else:
            vector_status = "missing_fireworks_api_key"
            _record_event(file_id, "building_vectors", "skipped", "FIREWORKS_API_KEY was not found; Q&A vector index was skipped.")

    counts = {
        "sections": len(assets.get("sections", [])),
        "subsections": sum(len(section.get("subsections", [])) for section in assets.get("sections", [])),
        "chunks": chunks.get("counts", {}).get("chunks", 0),
        "tables": len(assets.get("tables", [])),
        "images": len(assets.get("images", [])),
    }
    return _save_metadata(
        file_id,
        status="ready",
        current_step="ready",
        current_step_label="Ready",
        current_message="All processing steps are complete.",
        progress_pct=100,
        total_duration_sec=round(time.perf_counter() - started, 3),
        counts=counts,
        vector_status=vector_status,
        vector_error=vector_error,
        artifacts={
            "sections": str(sections_path.relative_to(ROOT)),
            "assets": str(assets_path.relative_to(ROOT)),
            "chunks": str(chunks_path.relative_to(ROOT)),
            "vectors": str((workspace / "index" / "vectors.db").relative_to(ROOT)),
        },
    )


def _run_vlm_parse_job(file_id: str, *, force: bool = False, build_table_vectors: bool = True) -> None:
    workspace = _workspace(file_id)
    pdf_path = workspace / "source.pdf"
    assets_path = workspace / "assets.json"
    if not pdf_path.exists() or not assets_path.exists():
        raise FileNotFoundError("PDF or assets.json is missing. Process the file first.")

    load_env_file(ROOT / ".env")
    started = time.perf_counter()
    _save_metadata(
        file_id,
        vlm_parse_progress={"index": 0, "total": 0, "table_id": None, "parsed_count": 0, "skipped_count": 0},
        current_step="vlm_parse",
        current_step_label="VLM table parse",
        current_message="Starting VLM table parse...",
        progress_pct=0,
    )
    _record_event(file_id, "vlm_parse", "running", "Running VLM table parse on all tables...")

    def _on_progress(row: dict[str, Any]) -> None:
        total = max(int(row.get("total") or 0), 1)
        index = int(row.get("index") or 0)
        pct = int((index / total) * 100)
        table_id = row.get("table_id") or ""
        _save_metadata(
            file_id,
            vlm_parse_progress=row,
            current_step="vlm_parse",
            current_step_label="VLM table parse",
            current_message=f"Parsing table {index}/{total}: {table_id}",
            progress_pct=min(99, pct),
        )

    try:
        result = enrich_assets_with_vlm(
            pdf_path,
            assets_path,
            skip_existing=not force,
            min_page=3,
            dpi=DEFAULT_VLM_DPI,
            multi_page_dpi=DEFAULT_VLM_MULTI_PAGE_DPI,
            on_progress=_on_progress,
        )
        table_vector_status = "skipped"
        table_vector_error = None
        if build_table_vectors:
            if os.environ.get("FIREWORKS_API_KEY"):
                try:
                    index_dir = workspace / "index" / "table_vectors"
                    index_dir.mkdir(parents=True, exist_ok=True)
                    build_table_vector_index([assets_path], index_dir, rebuild=True, batch_size=64, embed_model=DEFAULT_EMBED_MODEL)
                    built_db = index_dir / "vectors.db"
                    canonical_db = workspace / "index" / "table_vectors.db"
                    canonical_db.parent.mkdir(parents=True, exist_ok=True)
                    if built_db.exists():
                        canonical_db.write_bytes(built_db.read_bytes())
                    table_vector_status = "ready"
                    _record_event(
                        file_id,
                        "table_vectors",
                        "done",
                        f"Built table summary index ({built_db.stat().st_size // 1024} KB).",
                    )
                except Exception as exc:
                    table_vector_status = "failed"
                    table_vector_error = str(exc)
                    _record_event(file_id, "table_vectors", "error", table_vector_error)
            else:
                table_vector_status = "missing_fireworks_api_key"
                _record_event(
                    file_id,
                    "table_vectors",
                    "skipped",
                    "FIREWORKS_API_KEY was not found; table summary embeddings were skipped.",
                )

        assets_payload = _read_json(assets_path)
        assets_counts = dict(assets_payload.get("counts") or {})
        current_counts = dict(_read_metadata_file(workspace / "metadata.json", file_id).get("counts") or {})
        counts = {**current_counts, **assets_counts}
        counts["tables"] = len(assets_payload.get("tables", [])) or assets_counts.get(
            "tables_after_pipeline", current_counts.get("tables", 0)
        )
        counts["images"] = len(assets_payload.get("images", [])) or assets_counts.get(
            "images_detected", current_counts.get("images", 0)
        )
        counts["vlm_tables_parsed"] = result.get("vlm_tables_parsed", counts.get("vlm_tables_parsed", 0))
        if counts.get("chunks") is None and (workspace / "chunks.json").is_file():
            chunks_payload = _read_json(workspace / "chunks.json")
            counts["chunks"] = chunks_payload.get("counts", {}).get("chunks") or len(
                chunks_payload.get("chunks", [])
            )
        message = (
            f"VLM parsed {result.get('vlm_tables_parsed', 0)} tables "
            f"({result.get('parsed_count', 0)} new, {result.get('skipped_count', 0)} skipped)."
        )
        _record_event(
            file_id,
            "vlm_parse",
            "done",
            message,
            duration_sec=round(time.perf_counter() - started, 3),
        )
        _save_metadata(
            file_id,
            current_message=message,
            current_step="ready",
            current_step_label="Ready",
            progress_pct=100,
            vlm_parse_progress=None,
            counts=counts,
            table_vector_status=table_vector_status,
            table_vector_error=table_vector_error,
        )
    except Exception as exc:
        _record_event(file_id, "vlm_parse", "error", str(exc))
        _save_metadata(file_id, last_error=str(exc), vlm_parse_progress=None, current_step="ready", progress_pct=0)
        raise
    finally:
        VLM_PARSE_RUNNING.discard(file_id)


def _run_process_job(file_id: str, build_vectors: bool) -> None:
    try:
        _process_file(file_id, build_vectors=build_vectors)
    except Exception as exc:
        _record_event(file_id, "process", "error", str(exc))
    finally:
        PROCESSING.discard(file_id)


def _chunk_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    text = chunk.get("text") or ""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "section_ref_id": chunk.get("section_ref_id"),
        "subsection_ref_id": chunk.get("subsection_ref_id"),
        "text_unit_kind": chunk.get("text_unit_kind"),
        "header_path": chunk.get("header_path") or [],
        "token_count": chunk.get("token_count"),
        "table_refs": chunk.get("table_refs") or [],
        "image_refs": chunk.get("image_refs") or [],
        "split_index": chunk.get("split_index"),
        "split_count": chunk.get("split_count"),
        "text": text,
        "preview": text[:260],
    }


def _table_summary(table: dict[str, Any]) -> dict[str, Any]:
    section = table.get("section_ref") or {}
    subsection = table.get("subsection_ref") or {}
    if table.get("bbox_by_page"):
        crops = [
            {"crop_idx": idx, "page": row.get("page"), "bbox": row.get("bbox")}
            for idx, row in enumerate(table.get("bbox_by_page") or [])
        ]
    else:
        crops = [{"crop_idx": 0, "page": table.get("page_start"), "bbox": table.get("bbox")}]
    vlm = table.get("vlm_parse") or {}
    return {
        "table_id": table.get("table_id"),
        "asset_type": table.get("asset_type"),
        "page_start": table.get("page_start"),
        "page_end": table.get("page_end"),
        "row_count": table.get("row_count"),
        "col_count": table.get("col_count"),
        "complexity_score": table.get("complexity_score"),
        "section_title": section.get("section_title"),
        "header_path": [section.get("section_title"), *(subsection.get("path") or [])],
        "source_table_ids": table.get("source_table_ids") or [],
        "crops": crops,
        "vlm_status": vlm.get("status"),
        "vlm_summary": vlm.get("summary"),
        "vlm_parse": {
            "status": vlm.get("status"),
            "summary": vlm.get("summary"),
            "markdown": vlm.get("markdown"),
            "image_path": vlm.get("image_path"),
            "parsed_at": vlm.get("parsed_at"),
            "model": vlm.get("model"),
            "latency_ms": vlm.get("latency_ms"),
            "error": vlm.get("error") or vlm.get("raw_response"),
        } if vlm else None,
    }


def _table_parse_record(table: dict[str, Any]) -> dict[str, Any]:
    section = table.get("section_ref") or {}
    subsection = table.get("subsection_ref") or {}
    vlm = table.get("vlm_parse") or {}
    header_path = [section.get("section_title"), *(subsection.get("path") or [])]
    return {
        "table_id": table.get("table_id"),
        "page_start": table.get("page_start"),
        "page_end": table.get("page_end"),
        "section_ref": vlm.get("section_ref") or " > ".join(part for part in header_path if part),
        "header_path": [part for part in header_path if part],
        "status": vlm.get("status") or "missing",
        "summary": vlm.get("summary"),
        "markdown": vlm.get("markdown"),
        "context_before": vlm.get("context_before") or [],
        "context_after": vlm.get("context_after") or [],
        "image_path": vlm.get("image_path"),
        "model": vlm.get("model"),
        "parsed_at": vlm.get("parsed_at"),
        "latency_ms": vlm.get("latency_ms"),
        "error": vlm.get("error") or vlm.get("raw_response"),
    }


def _bbox_height(bbox: list[float]) -> float:
    return float(bbox[3] - bbox[1])


def _is_tight_table_fragment(bbox: list[float]) -> bool:
    return _bbox_height(bbox) < 40


def _page_bottom_top_pad(page: fitz.Page, bbox: list[float]) -> float:
    if bbox[1] > page.rect.height * 0.82:
        return 36
    return 36


def _preview_clip_for_crop(page: fitz.Page, table: dict[str, Any], crop_idx: int, bbox: list[float]) -> fitz.Rect:
    from section_asset_extractor import is_header_only_table_crop, table_crop_clip, table_crop_padding

    if is_header_only_table_crop(table, crop_idx, bbox, page.rect.height):
        return fitz.Rect(
            max(page.rect.x0, bbox[0] - 4),
            max(page.rect.y0, bbox[1] - 4),
            min(page.rect.x1, bbox[2] + 4),
            min(page.rect.y1, bbox[3] + 8),
        )
    top_pad, bottom_pad = table_crop_padding(table, crop_idx, bbox, page.rect.height)
    return table_crop_clip(page, bbox, top_pad, bottom_pad)


def _cross_page_header_clip(doc: fitz.Document, multi_crops: list[dict[str, Any]]) -> tuple[fitz.Page, fitz.Rect] | None:
    if not multi_crops:
        return None
    first = multi_crops[0]
    page_no = first.get("page")
    bbox = first.get("bbox")
    if not page_no or not bbox:
        return None
    page = doc[int(page_no) - 1]
    top_pad = _page_bottom_top_pad(page, bbox)
    clip = fitz.Rect(0, max(page.rect.y0, bbox[1] - top_pad), page.rect.x1, min(page.rect.y1, bbox[3] + 12))
    return page, clip & page.rect


def _combine_pixmaps_vertical(top: fitz.Pixmap, bottom: fitz.Pixmap) -> fitz.Pixmap:
    from PIL import Image

    top_img = Image.open(io.BytesIO(top.tobytes("png")))
    bottom_img = Image.open(io.BytesIO(bottom.tobytes("png")))
    width = max(top_img.width, bottom_img.width)
    height = top_img.height + bottom_img.height
    combined_img = Image.new("RGB", (width, height), "white")
    combined_img.paste(top_img, (0, 0))
    combined_img.paste(bottom_img, (0, top_img.height))
    buf = io.BytesIO()
    combined_img.save(buf, format="PNG")
    return fitz.Pixmap(buf.getvalue())


def _asset_tables(workspace: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return extractor tables as-is; grouping is done offline in table_pipeline."""
    return list(payload.get("tables", []))


def _find_table_by_id(workspace: Path, assets: dict[str, Any], table_id: str) -> dict[str, Any] | None:
    matches = [table for table in assets.get("tables", []) if table.get("table_id") == table_id]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        with_crops = [table for table in matches if table.get("bbox_by_page")]
        if with_crops:
            return with_crops[0]
        return matches[0]
    for table in assets.get("tables", []):
        if table_id in (table.get("source_table_ids") or []):
            return table
    return None


def _image_summary(image: dict[str, Any]) -> dict[str, Any]:
    section = image.get("section_ref") or {}
    subsection = image.get("subsection_ref") or {}
    return {
        "image_id": image.get("image_id"),
        "page": image.get("page"),
        "bbox": image.get("bbox"),
        "width": image.get("width"),
        "height": image.get("height"),
        "section_title": section.get("section_title"),
        "header_path": [section.get("section_title"), *(subsection.get("path") or [])],
    }


def _html_response(name: str) -> HTMLResponse:
    path = STATIC_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=500, detail=f"Missing static UI: {name}")
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/{asset_path:path}", include_in_schema=False)
def static_asset(asset_path: str) -> FileResponse:
    path = (STATIC_DIR / asset_path).resolve()
    if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="Static asset not found.")
    return FileResponse(path)


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    path = STATIC_DIR / "favicon.svg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Missing favicon.")
    return FileResponse(path, media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    return favicon_svg()


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return _html_response("index.html")


@app.get("/agent", response_class=HTMLResponse)
def agent_ui() -> HTMLResponse:
    """Standalone agent Q&A + tool trace UI."""
    return _html_response("agent.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/files")
def list_files() -> dict[str, Any]:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(WORKSPACE_DIR.iterdir(), reverse=True):
        if path.is_dir() and (path / "metadata.json").exists():
            files.append(_metadata(path.name))
    return {"files": files}


@app.post("/api/files")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    digest = hashlib.sha1(content).hexdigest()[:10]
    file_id = f"{int(time.time())}-{_safe_stem(file.filename)}-{digest}"
    workspace = WORKSPACE_DIR / file_id
    workspace.mkdir(parents=True, exist_ok=False)
    pdf_path = workspace / "source.pdf"
    pdf_path.write_bytes(content)

    meta = _save_metadata(
        file_id,
        original_filename=file.filename,
        status="uploaded",
        size_bytes=len(content),
        vector_status="not_built",
        process_events=[],
        process_plan=PROCESS_STEPS,
        current_step=None,
        current_step_label=None,
        current_message=None,
        progress_pct=0,
        last_error=None,
    )
    return {"file": meta}


@app.post("/api/files/{file_id}/process")
def process_file(file_id: str, build_vectors: bool = False) -> dict[str, Any]:
    _workspace(file_id)
    if file_id in PROCESSING:
        return {"file": _metadata(file_id), "started": False}
    PROCESSING.add(file_id)
    meta = _save_metadata(
        file_id,
        status="processing",
        current_step="queued",
        current_step_label="Queued",
        current_message="Processing job accepted.",
        progress_pct=0,
        last_error=None,
    )
    thread = threading.Thread(target=_run_process_job, args=(file_id, build_vectors), daemon=True)
    thread.start()
    return {"file": meta, "started": True}


@app.get("/api/files/{file_id}")
def get_file(file_id: str) -> dict[str, Any]:
    _workspace(file_id)
    return {"file": _metadata(file_id)}


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str) -> dict[str, str]:
    workspace = _workspace(file_id)
    shutil.rmtree(workspace)
    return {"status": "deleted", "file_id": file_id}


def _load_chunks_payload(file_id: str) -> dict[str, Any]:
    workspace = _workspace(file_id)
    chunks_path = workspace / "chunks.json"
    if chunks_path.exists():
        return json.loads(chunks_path.read_text(encoding="utf-8"))
    sections_path = workspace / "sections.json"
    if not sections_path.exists():
        raise HTTPException(status_code=404, detail="Missing artifact: chunks.json")
    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    return _build_preview_chunks(sections)


@app.get("/api/files/{file_id}/chunks")
def get_chunks(file_id: str) -> dict[str, Any]:
    payload = _load_chunks_payload(file_id)
    chunks = [_chunk_summary(chunk) for chunk in payload.get("chunks", [])]
    return {
        "counts": payload.get("counts", {}),
        "preview": bool(payload.get("preview")),
        "chunks": chunks,
    }


@app.get("/api/files/{file_id}/table-parses")
def get_table_parses(file_id: str) -> dict[str, Any]:
    workspace = _workspace(file_id)
    payload = _read_json(workspace / "assets.json")
    records = [_table_parse_record(table) for table in payload.get("tables", [])]
    success = sum(1 for row in records if row.get("status") == "success")
    table_db = _workspace_table_db_path(workspace)
    meta = _read_metadata_file(workspace / "metadata.json", file_id)
    running = file_id in VLM_PARSE_RUNNING
    return {
        "running": running,
        "has_table_vector_index": table_db is not None,
        "table_vector_path": str(table_db.relative_to(ROOT)) if table_db else None,
        "progress": meta.get("vlm_parse_progress"),
        "current_message": meta.get("current_message") if running else None,
        "progress_pct": meta.get("progress_pct") if running else None,
        "counts": {
            "tables": len(records),
            "parsed": success,
            "missing": sum(1 for row in records if row.get("status") == "missing"),
            "failed": len(records) - success - sum(1 for row in records if row.get("status") == "missing"),
        },
        "parses": records,
    }


@app.post("/api/files/{file_id}/table-parses/run")
async def run_table_parses(file_id: str, force: bool = False, build_table_vectors: bool = True) -> dict[str, Any]:
    workspace = _workspace(file_id)
    if file_id in VLM_PARSE_RUNNING:
        return {"started": False, "running": True, "message": "VLM table parse is already running."}
    if not (workspace / "assets.json").exists():
        raise HTTPException(status_code=409, detail="assets.json is missing. Process the file first.")
    if not (workspace / "source.pdf").exists():
        raise HTTPException(status_code=404, detail="Missing uploaded PDF.")

    load_env_file(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("FIREWORKS_API_KEY"):
        raise HTTPException(
            status_code=409,
            detail="ANTHROPIC_API_KEY or FIREWORKS_API_KEY is required for VLM table parse.",
        )

    VLM_PARSE_RUNNING.add(file_id)

    def _job() -> None:
        try:
            _run_vlm_parse_job(file_id, force=force, build_table_vectors=build_table_vectors)
        except Exception:
            pass
        finally:
            VLM_PARSE_RUNNING.discard(file_id)

    threading.Thread(target=_job, daemon=True).start()
    return {
        "started": True,
        "running": True,
        "message": "VLM table parse started. Poll /table-parses for progress.",
    }


@app.get("/api/files/{file_id}/tables/{table_id}/parse-image.png")
def get_table_parse_image(file_id: str, table_id: str) -> Response:
    workspace = _workspace(file_id)
    payload = _read_json(workspace / "assets.json")
    table = next((row for row in payload.get("tables", []) if row.get("table_id") == table_id), None)
    if not table:
        raise HTTPException(status_code=404, detail=f"Unknown table_id: {table_id}")
    image_rel = (table.get("vlm_parse") or {}).get("image_path")
    if not image_rel:
        raise HTTPException(status_code=404, detail="No parse image saved for this table.")
    image_path = workspace / image_rel
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Parse image file is missing on disk.")
    return Response(
        content=image_path.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/files/{file_id}/assets")
def get_assets(file_id: str) -> dict[str, Any]:
    workspace = _workspace(file_id)
    payload = _read_json(workspace / "assets.json")
    api_tables = _asset_tables(workspace, payload)
    counts = dict(payload.get("counts", {}))
    counts["api_tables"] = len(api_tables)
    return {
        "counts": counts,
        "sections": payload.get("sections", []),
        "tables": [_table_summary(table) for table in api_tables],
        "images": [_image_summary(image) for image in payload.get("images", [])],
    }


@app.get("/api/files/{file_id}/tables/{table_id}")
def get_table(file_id: str, table_id: str) -> dict[str, Any]:
    workspace = _workspace(file_id)
    payload = _read_json(workspace / "assets.json")
    table = _find_table_by_id(workspace, payload, table_id)
    if not table:
        raise HTTPException(status_code=404, detail=f"Unknown table_id: {table_id}")
    return {"table": _table_summary(table)}


@app.get("/api/files/{file_id}/tables/{table_id}/crops/{crop_idx}.png")
def get_table_crop(file_id: str, table_id: str, crop_idx: int) -> Response:
    workspace = _workspace(file_id)
    assets = _read_json(workspace / "assets.json")
    table = _find_table_by_id(workspace, assets, table_id)
    if not table:
        raise HTTPException(status_code=404, detail=f"Unknown table_id: {table_id}")

    if table.get("bbox_by_page"):
        crops = table.get("bbox_by_page") or []
        if crop_idx < 0 or crop_idx >= len(crops):
            raise HTTPException(status_code=404, detail=f"Unknown crop index: {crop_idx}")
        page_no = crops[crop_idx].get("page")
        bbox = crops[crop_idx].get("bbox")
    else:
        if crop_idx != 0:
            raise HTTPException(status_code=404, detail=f"Unknown crop index: {crop_idx}")
        page_no = table.get("page_start")
        bbox = table.get("bbox")

    if not bbox or not page_no:
        raise HTTPException(status_code=422, detail="Table asset has no crop coordinates.")

    doc = fitz.open(workspace / "source.pdf")
    try:
        page = doc[int(page_no) - 1]
        multi_page_crops = table.get("bbox_by_page") or []
        clip = _preview_clip_for_crop(page, table, crop_idx, bbox) & page.rect
        matrix = fitz.Matrix(2.5, 2.5)
        body_pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)

        # When bbox_by_page has multiple crops, the UI shows each part separately
        # (header crop + data crop). Do not re-stitch the header onto later crops.
        show_parts_separately = len(multi_page_crops) > 1
        stitch_header = (
            not show_parts_separately
            and crop_idx > 0
            and len(multi_page_crops) > 1
            and multi_page_crops[0].get("page") != page_no
            and (table.get("asset_type") == "table_group" or _is_tight_table_fragment(bbox))
        )
        if stitch_header:
            header_info = _cross_page_header_clip(doc, multi_page_crops)
            if header_info:
                header_page, header_clip = header_info
                header_pix = header_page.get_pixmap(matrix=matrix, clip=header_clip, alpha=False)
                pix = _combine_pixmaps_vertical(header_pix, body_pix)
            else:
                pix = body_pix
        else:
            header_crop = table.get("header_crop")
            if crop_idx > 0 and header_crop and header_crop.get("bbox"):
                header_page = doc[int(header_crop["page"]) - 1]
                header_clip = fitz.Rect(header_crop["bbox"]) & header_page.rect
                header_pix = header_page.get_pixmap(matrix=matrix, clip=header_clip, alpha=False)
                pix = _combine_pixmaps_vertical(header_pix, body_pix)
            else:
                pix = body_pix
        return Response(
            content=pix.tobytes("png"),
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    finally:
        doc.close()


@app.get("/api/files/{file_id}/images/{image_id}.png")
def get_image(file_id: str, image_id: str) -> Response:
    workspace = _workspace(file_id)
    assets = _read_json(workspace / "assets.json")
    image = next((image for image in assets.get("images", []) if image.get("image_id") == image_id), None)
    if not image:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    bbox = image.get("bbox")
    page_no = image.get("page")
    if not bbox or not page_no:
        raise HTTPException(status_code=422, detail="Image asset has no crop coordinates.")

    doc = fitz.open(workspace / "source.pdf")
    try:
        page = doc[int(page_no) - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(bbox), alpha=False)
        return Response(content=pix.tobytes("png"), media_type="image/png")
    finally:
        doc.close()


@app.get("/api/agent/langsmith/status")
async def agent_langsmith_status() -> dict[str, Any]:
    """Whether LangSmith tracing is configured (for the agent UI)."""
    load_env_file(ROOT / ".env")
    return _load_langsmith_tracing_module().langsmith_status()


@app.get("/api/agent/memory")
async def agent_memory_get(
    session_id: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Snapshot of short-term / episodic / semantic memory for the UI."""
    return _load_agent_memory_module().memory_snapshot(session_id, query=query or "")


@app.delete("/api/agent/memory")
async def agent_memory_delete(session_id: str = Query(..., min_length=1)) -> dict[str, Any]:
    """Clear server-side session (New chat)."""
    _load_agent_memory_module().clear_session(session_id)
    return {"ok": True, "session_id": session_id}


@app.get("/api/agent/index-status")
async def agent_index_status() -> dict[str, Any]:
    from chunk_studio.agent_bridge import global_index_status  # noqa: E402

    return global_index_status()


@app.get("/api/agent/rag-pipeline-profile")
async def agent_rag_pipeline_profile() -> dict[str, Any]:
    """Studio RAG inference defaults (shared with offline golden eval)."""
    inference_dir = str(ROOT / "main" / "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    from rag_pipeline_config import STUDIO_PIPELINE_PROFILE  # noqa: E402

    return STUDIO_PIPELINE_PROFILE


@app.post("/api/agent/trace/stream")
def agent_trace_stream_global(body: AgentTraceBody) -> StreamingResponse:
    """Stream agent steps against the merged all-filings RAG index."""
    try:
        from chunk_studio.agent_bridge import iter_trace_events_global  # noqa: E402

        return StreamingResponse(
            iter_trace_events_global(
                body.question.strip(),
                body.max_steps,
                session_id=body.session_id,
            ),
            media_type="application/x-ndjson",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/files/{file_id}/agent/trace/stream")
def agent_trace_stream(file_id: str, body: AgentTraceBody) -> StreamingResponse:
    """Stream agent steps as NDJSON (step_start → step → done)."""
    workspace = _workspace(file_id)
    meta = _metadata(file_id)
    try:
        from chunk_studio.agent_bridge import iter_trace_events_for_workspace  # noqa: E402

        label = meta.get("original_filename") or file_id

        return StreamingResponse(
            iter_trace_events_for_workspace(
                workspace,
                body.question.strip(),
                body.max_steps,
                file_label=label,
                session_id=body.session_id,
            ),
            media_type="application/x-ndjson",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    p = _port()
    print(
        f"\n  Chunk Studio:  http://127.0.0.1:{p}/\n"
        f"  Agent Q&A:     http://127.0.0.1:{p}/agent\n",
        flush=True,
    )
    uvicorn.run(app, host=_host(), port=_port(), log_level="info")


if __name__ == "__main__":
    main()

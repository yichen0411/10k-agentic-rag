"""Run the LangChain agent against global or per-workspace RAG indexes."""

from __future__ import annotations

import importlib.util
import json
import queue
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "main" / "agent"

MAIN_SQL = ROOT / "main" / "sql"
MAIN_INFERENCE = ROOT / "main" / "inference"
_MAIN_PREFIX = "chunk_studio_agent_"

GLOBAL_TEXT_DB = ROOT / "data" / "index" / "text_chunks" / "vectors.db"
GLOBAL_TABLE_DB = ROOT / "data" / "index" / "table_summaries" / "vectors.db"
GLOBAL_ASSETS = ROOT / "data" / "index" / "merged_assets.json"
GLOBAL_SCOPE_ID = "all_filings"


def _ensure_paths() -> None:
    for path in [str(MAIN_INFERENCE), str(MAIN_SQL), str(AGENT_DIR), str(ROOT)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_module(name: str, path: Path) -> Any:
    full_name = _MAIN_PREFIX + name
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _load_memory() -> Any:
    _ensure_paths()
    return _load_module("agent_memory", AGENT_DIR / "agent_memory.py")


def _preload_numpy_without_macos_polyfit_check() -> None:
    """Avoid a NumPy/Accelerate FPE during transformers import on this macOS env."""
    if "numpy" in sys.modules:
        return
    real_platform = sys.platform
    try:
        # NumPy's darwin-only import sanity check calls polyfit/linalg and can
        # raise SIGFPE with this local Accelerate build before Python can catch it.
        sys.platform = "linux"
        __import__("numpy")
    finally:
        sys.platform = real_platform


def _load_agent_stack() -> tuple[Any, Any]:
    _ensure_paths()
    _preload_numpy_without_macos_polyfit_check()
    tools_mod = _load_module("tools", AGENT_DIR / "tools.py")
    sys.modules["tools"] = tools_mod
    sys.modules["system_prompt"] = _load_module("system_prompt", AGENT_DIR / "system_prompt.py")
    trace_mod = _load_module("trace_format", AGENT_DIR / "trace_format.py")
    lc_mod = _load_module("langchain_agent", AGENT_DIR / "langchain_agent.py")
    return lc_mod, trace_mod


def _workspace_table_db(workspace: Path) -> Path | None:
    for rel in ("index/table_vectors.db", "index/table_vectors/vectors.db"):
        path = workspace / rel
        if path.is_file():
            return path
    return None


def _vector_db_health(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return {"exists": False, "row_count": 0, "valid": False, "reason": "missing"}
    conn = sqlite3.connect(db_path)
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if row_count == 0:
            return {"exists": True, "row_count": 0, "valid": False, "reason": "empty"}
        null_ticker = conn.execute("SELECT COUNT(*) FROM chunks WHERE ticker IS NULL OR ticker = ''").fetchone()[0]
        bad_source = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_file = 'source.pdf'").fetchone()[0]
        groups = conn.execute(
            "SELECT ticker, fiscal_year, COUNT(*) FROM chunks GROUP BY ticker, fiscal_year ORDER BY 3 DESC"
        ).fetchall()
        valid = null_ticker == 0 and bad_source == 0
        reason = None
        if not valid:
            reason = "stale_source_file" if bad_source else "missing_ticker_metadata"
        return {
            "exists": True,
            "row_count": row_count,
            "valid": valid,
            "null_ticker": null_ticker,
            "bad_source": bad_source,
            "groups": groups,
            "reason": reason,
        }
    finally:
        conn.close()


def global_rag_paths() -> tuple[Path, Path | None, Path | None]:
    return GLOBAL_TEXT_DB, GLOBAL_TABLE_DB if GLOBAL_TABLE_DB.exists() else None, GLOBAL_ASSETS if GLOBAL_ASSETS.exists() else None


def global_index_status() -> dict[str, Any]:
    text = _vector_db_health(GLOBAL_TEXT_DB)
    table = _vector_db_health(GLOBAL_TABLE_DB)
    filings = sorted({f"{row[0]} {row[1]}" for row in text.get("groups") or [] if row[0] and row[1]})
    return {
        "scope": GLOBAL_SCOPE_ID,
        "ready": text["valid"] and table["valid"] and GLOBAL_ASSETS.exists(),
        "text_db": str(GLOBAL_TEXT_DB),
        "table_db": str(GLOBAL_TABLE_DB),
        "assets_path": str(GLOBAL_ASSETS),
        "text": text,
        "table": table,
        "assets_exists": GLOBAL_ASSETS.exists(),
        "filings": filings,
        "filing_count": len(filings),
    }


def _require_rag_ready(db_path: Path, *, scope_label: str) -> None:
    health = _vector_db_health(db_path)
    if health["valid"]:
        return
    raise RuntimeError(
        f"{scope_label} vector index is missing, empty, or stale. "
        "Run main/chunking/rebuild_workspace_indexes.py and "
        "main/chunking/merge_filing_assets.py."
    )


def _run_agent(
    *,
    query: str,
    max_steps: int,
    db_path: Path,
    table_db_path: Path | None,
    assets_path: Path | None,
    scope_id: str,
    scope_label: str,
    session_id: Optional[str],
    on_event: Optional[Any] = None,
    mode_label: str,
) -> dict[str, Any]:
    _require_rag_ready(db_path, scope_label=scope_label)
    mem_mod = _load_memory()
    sid = mem_mod.get_or_create(session_id)
    mem = mem_mod.build_agent_memory(sid, query=query)
    lc_mod, trace_mod = _load_agent_stack()
    payload = lc_mod.run_langchain_agent(
        query,
        max_steps=max_steps,
        rag_db_path=db_path,
        rag_assets_path=assets_path,
        rag_table_db_path=table_db_path,
        mode_label=mode_label,
        on_event=on_event,
        chat_history=mem["chat_history"],
        memory_context=mem.get("memory_context") or "",
        session_id=sid,
        file_id=scope_id,
    )
    formatted = trace_mod.format_trace_payload(payload)
    prompt_injection = trace_mod.build_prompt_injection_record(
        query=query,
        memory_context=mem.get("memory_context") or "",
        chat_history=mem.get("chat_history") or [],
        trace=payload.get("trace") or [],
    )
    mem_mod.save_last_prompt_snapshot(sid, prompt_injection)
    mem_mod.append_turn(
        sid,
        query,
        payload.get("answer") or "",
        file_id=scope_id,
        tool_steps=formatted.get("steps") or [],
        prompt_injection=prompt_injection,
    )
    formatted["prompt_injection"] = prompt_injection
    formatted["langsmith"] = payload.get("langsmith") or {}
    formatted["file_id"] = scope_id
    formatted["file_label"] = scope_label
    formatted["rag_index"] = str(db_path)
    formatted["rag_scope"] = scope_label
    formatted["session_id"] = sid
    return formatted


def run_trace_global(
    query: str,
    max_steps: int = 6,
    *,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    db_path, table_db_path, assets_path = global_rag_paths()
    return _run_agent(
        query=query,
        max_steps=max_steps,
        db_path=db_path,
        table_db_path=table_db_path,
        assets_path=assets_path,
        scope_id=GLOBAL_SCOPE_ID,
        scope_label="All 6 filings (AAPL/MSFT/GOOGL FY2024–FY2025)",
        session_id=session_id,
        mode_label="langchain_global",
    )


def iter_trace_events_global(
    query: str,
    max_steps: int = 6,
    *,
    session_id: Optional[str] = None,
) -> Iterator[bytes]:
    db_path, table_db_path, assets_path = global_rag_paths()
    _require_rag_ready(db_path, scope_label="Global")

    event_queue: queue.Queue[Any] = queue.Queue()

    def _emit(event: dict[str, Any]) -> None:
        event_queue.put(event)

    mem_mod = _load_memory()
    sid = mem_mod.get_or_create(session_id)

    def _worker() -> None:
        try:
            formatted = _run_agent(
                query=query,
                max_steps=max_steps,
                db_path=db_path,
                table_db_path=table_db_path,
                assets_path=assets_path,
                scope_id=GLOBAL_SCOPE_ID,
                scope_label="All 6 filings (AAPL/MSFT/GOOGL FY2024–FY2025)",
                session_id=sid,
                on_event=_emit,
                mode_label="langchain_global",
            )
            event_queue.put({"type": "done", **formatted})
        except Exception as exc:
            event_queue.put({"type": "error", "detail": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    event_queue.put({"type": "started", "query": query, "session_id": sid, "scope": GLOBAL_SCOPE_ID})
    while True:
        event = event_queue.get()
        if event is None:
            break
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def run_trace_for_workspace(
    workspace: Path,
    query: str,
    max_steps: int = 6,
    *,
    require_vectors: bool = True,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    db_path = workspace / "index" / "vectors.db"
    table_db_path = _workspace_table_db(workspace)
    assets_path = workspace / "assets.json"
    if require_vectors:
        _require_rag_ready(db_path, scope_label=workspace.name)
    return _run_agent(
        query=query,
        max_steps=max_steps,
        db_path=db_path,
        table_db_path=table_db_path,
        assets_path=assets_path if assets_path.exists() else None,
        scope_id=workspace.name,
        scope_label=workspace.name,
        session_id=session_id,
        mode_label="langchain_chunk_studio",
    )


def iter_trace_events_for_workspace(
    workspace: Path,
    query: str,
    max_steps: int = 6,
    *,
    require_vectors: bool = True,
    file_label: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Iterator[bytes]:
    db_path = workspace / "index" / "vectors.db"
    table_db_path = _workspace_table_db(workspace)
    assets_path = workspace / "assets.json"
    if require_vectors:
        _require_rag_ready(db_path, scope_label=file_label or workspace.name)

    event_queue: queue.Queue[Any] = queue.Queue()

    def _emit(event: dict[str, Any]) -> None:
        event_queue.put(event)

    mem_mod = _load_memory()
    sid = mem_mod.get_or_create(session_id)

    def _worker() -> None:
        try:
            formatted = _run_agent(
                query=query,
                max_steps=max_steps,
                db_path=db_path,
                table_db_path=table_db_path,
                assets_path=assets_path if assets_path.exists() else None,
                scope_id=workspace.name,
                scope_label=file_label or workspace.name,
                session_id=sid,
                on_event=_emit,
                mode_label="langchain_chunk_studio",
            )
            event_queue.put({"type": "done", **formatted})
        except Exception as exc:
            event_queue.put({"type": "error", "detail": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    event_queue.put({"type": "started", "query": query, "session_id": sid})
    while True:
        event = event_queue.get()
        if event is None:
            break
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

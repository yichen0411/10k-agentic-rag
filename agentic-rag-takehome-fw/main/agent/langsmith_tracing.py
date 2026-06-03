"""Optional LangSmith tracing for the LangChain agent (env-gated)."""

from __future__ import annotations

import os
from typing import Any, Optional


def _env_disabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"false", "0", "no", "off"}


def _env_enabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"true", "1", "yes", "on"}


def configure_langsmith_tracing() -> dict[str, Any]:
    """
    Turn on LangSmith auto-tracing for LangChain only when explicitly enabled.

    Uses LANGSMITH_* with LANGCHAIN_* fallbacks. Set LANGSMITH_TRACING=true
    or LANGCHAIN_TRACING_V2=true to enable.
    """
    if _env_disabled("LANGSMITH_TRACING") or _env_disabled("LANGCHAIN_TRACING_V2"):
        return {"enabled": False, "reason": "tracing_disabled"}

    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not api_key:
        return {"enabled": False, "reason": "no_api_key"}
    if not (_env_enabled("LANGSMITH_TRACING") or _env_enabled("LANGCHAIN_TRACING_V2")):
        return {"enabled": False, "reason": "tracing_not_enabled"}

    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "fireworks-agentic-rag"
    ).strip()

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "").strip()
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    return {"enabled": True, "project": project, "endpoint": endpoint or None}


def build_langsmith_run_config(
    *,
    query: str,
    mode_label: str = "langchain",
    session_id: Optional[str] = None,
    file_id: Optional[str] = None,
    callbacks: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """RunnableConfig for AgentExecutor.invoke (tags/metadata + UI callbacks)."""
    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = list(callbacks)

    q = (query or "").strip()
    config["run_name"] = f"agent:{q[:120]}" if q else "agent"
    tags = ["10k-agentic-rag", mode_label]
    if file_id:
        tags.append(f"file:{file_id[:48]}")
    if session_id:
        tags.append(f"session:{session_id[:12]}")
    config["tags"] = tags

    metadata: dict[str, Any] = {"mode": mode_label}
    if q:
        metadata["query"] = q
    if session_id:
        metadata["session_id"] = session_id
    if file_id:
        metadata["file_id"] = file_id
    config["metadata"] = metadata
    return config


def langsmith_status() -> dict[str, Any]:
    """UI/API: whether tracing would activate (does not call LangSmith)."""
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "fireworks-agentic-rag"
    ).strip()
    if _env_disabled("LANGSMITH_TRACING") or _env_disabled("LANGCHAIN_TRACING_V2"):
        return {
            "configured": bool(api_key),
            "enabled": False,
            "reason": "tracing_disabled",
            "project": project,
        }
    enabled = bool(api_key) and (_env_enabled("LANGSMITH_TRACING") or _env_enabled("LANGCHAIN_TRACING_V2"))
    return {
        "configured": bool(api_key),
        "enabled": enabled,
        "reason": "ok" if enabled else ("tracing_not_enabled" if api_key else "no_api_key"),
        "project": project,
        "docs_url": "https://docs.smith.langchain.com/",
        "app_url": "https://smith.langchain.com/",
    }


def _run_url_from_id(run_id: str) -> str:
    rid = (run_id or "").strip()
    if not rid:
        return ""
    try:
        from langsmith import Client

        return str(Client().get_run_url(rid) or "")
    except Exception:
        return f"https://smith.langchain.com/public/{rid}/r"


def capture_langsmith_run_link(tracer: Any = None) -> dict[str, str]:
    """Best-effort root run id/url after invoke (works in worker threads)."""
    out: dict[str, str] = {}

    try:
        from langsmith.run_helpers import get_current_run_tree

        tree = get_current_run_tree()
        if tree is not None:
            run_id = str(getattr(tree, "id", "") or "")
            url = str(getattr(tree, "url", "") or "")
            if run_id:
                out["run_id"] = run_id
            if url:
                out["url"] = url
    except Exception:
        pass

    if not out.get("run_id") and tracer is not None:
        run_map = getattr(tracer, "run_map", None) or {}
        root = None
        for run in run_map.values():
            parent = getattr(run, "parent_run_id", None)
            if parent is None:
                root = run
                break
        if root is None and run_map:
            root = next(iter(run_map.values()))
        if root is not None:
            run_id = str(getattr(root, "id", "") or "")
            if run_id:
                out["run_id"] = run_id
            url = str(getattr(root, "url", "") or "")
            if url:
                out["url"] = url

    if out.get("run_id") and not out.get("url"):
        out["url"] = _run_url_from_id(out["run_id"])

    return out


def make_langchain_tracer(project: str) -> Any:
    """LangChainTracer callback so we can read run id after AgentExecutor.invoke."""
    from langchain_core.tracers.langchain import LangChainTracer

    return LangChainTracer(project_name=project)

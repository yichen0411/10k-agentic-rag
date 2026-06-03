from __future__ import annotations

import json
import os
import sys
import time
import traceback
import importlib.util
import html
from pathlib import Path
from typing import Any

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets as environment variables for existing code."""
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    for key, value in secrets.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value
    # Hosted demo should prioritize reliability over tracing callbacks.
    if os.environ.get("ENABLE_LANGSMITH_IN_STREAMLIT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


_load_streamlit_secrets()

from chunk_studio.agent_bridge import global_index_status, run_trace_global  # noqa: E402


AGENT_DIR = ROOT / "main" / "agent"
DATA_INDEX = ROOT / "data" / "index"


EXAMPLE_QUESTIONS = [
    "Which segment grew closest to MSFT revenue growth?",
    "Compare Apple and Microsoft FY2025 revenue growth and cite the filings.",
    "What drove Alphabet's revenue growth in FY2025?",
    "Summarize Salesforce FY2025 risk factors related to AI and competition.",
]


def _load_direct_tools_module() -> Any:
    """Load the existing tool implementations without relying on LangChain."""
    for path in [ROOT / "main" / "sql", ROOT / "main" / "inference", ROOT / "main" / "chunking", AGENT_DIR, ROOT]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    module_name = "streamlit_direct_agent_tools"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, AGENT_DIR / "tools.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load agent tools.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _question_scope(question: str) -> tuple[str | None, str | None]:
    upper = question.upper()
    ticker = None
    for candidate in ("AAPL", "MSFT", "GOOGL", "SALESFORCE"):
        if candidate in upper:
            ticker = candidate
            break
    aliases = {
        "APPLE": "AAPL",
        "MICROSOFT": "MSFT",
        "ALPHABET": "GOOGL",
        "GOOGLE": "GOOGL",
        "SALESFORCE": "SALESFORCE",
    }
    if ticker is None:
        for alias, value in aliases.items():
            if alias in upper:
                ticker = value
                break

    year = None
    for candidate in ("FY2025", "FY2024", "2025", "2024"):
        if candidate in upper:
            year = candidate if candidate.startswith("FY") else f"FY{candidate}"
            break
    return ticker, year


def _wants_sql(question: str) -> bool:
    q = question.lower()
    keywords = (
        "revenue",
        "growth",
        "segment",
        "income",
        "margin",
        "assets",
        "liabilities",
        "cash",
        "debt",
        "eps",
        "compare",
        "closest",
        "highest",
        "lowest",
    )
    return any(keyword in q for keyword in keywords)


def _wants_rag(question: str) -> bool:
    q = question.lower()
    keywords = (
        "why",
        "what drove",
        "driver",
        "summarize",
        "risk",
        "explain",
        "disclose",
        "filing",
        "10-k",
        "cite",
        "management",
        "competition",
        "ai",
    )
    return any(keyword in q for keyword in keywords)


def _format_sql_answer(rows: Any, sql: str | None) -> str:
    if isinstance(rows, list) and rows:
        preview = rows[:8]
        lines = ["Structured result:"]
        for idx, row in enumerate(preview, start=1):
            if isinstance(row, dict):
                values = ", ".join(f"{key}: {value}" for key, value in row.items())
            else:
                values = str(row)
            lines.append(f"{idx}. {values}")
        if len(rows) > len(preview):
            lines.append(f"... plus {len(rows) - len(preview)} more rows.")
        return "\n".join(lines)
    if sql:
        return "I ran the structured financial database query, but it did not return rows for this question."
    return ""


def _compact_for_synthesis(value: Any, max_chars: int = 9000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _synthesize_with_llm(question: str, steps: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Generate the final answer from tool observations in the direct fallback path."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "", "ANTHROPIC_API_KEY is not configured."

    model = (
        os.environ.get("ANTHROPIC_AGENT_MODEL")
        or os.environ.get("ANTHROPIC_CHAT_MODEL")
        or os.environ.get("ANTHROPIC_ANSWER_MODEL")
        or "claude-haiku-4-5-20251001"
    )
    observations = []
    for step in steps:
        output = step.get("output") or {}
        observations.append(
            {
                "step": step.get("step"),
                "tool": step.get("tool"),
                "input": step.get("input"),
                "observation": output,
            }
        )

    prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Tool observations:\n"
        f"{_compact_for_synthesis(observations)}\n\n"
        "Write a concise financial research answer. Use the SQL rows for exact numbers, "
        "use filing/RAG observations for narrative explanation, and cite source files or "
        "sections when available. If evidence is incomplete, say so directly."
    )
    payload = {
        "model": model,
        "max_tokens": 900,
        "temperature": 0.1,
        "system": "You are a careful financial research agent answering from tool observations only.",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(part.strip() for part in parts if part.strip()), None
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _references_from_tool_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in steps:
        output = step.get("output") or {}
        for source in output.get("sources") or output.get("reranked_top") or []:
            key = str(source.get("chunk_id") or source.get("source_file") or source.get("header_path") or "")
            if not key or key in seen:
                continue
            header = source.get("header_path")
            if isinstance(header, list):
                header = " > ".join(str(part) for part in header)
            refs.append(
                {
                    "source_file": source.get("source_file") or "10-K filing",
                    "header_path": header or "",
                    "excerpt": source.get("excerpt") or source.get("content") or "",
                    "score": source.get("score") or source.get("rerank_score"),
                }
            )
            seen.add(key)
            if len(refs) >= 4:
                return refs
    return refs


def _direct_tool_agent(question: str, *, original_error: Exception | None = None) -> dict[str, Any]:
    tools = _load_direct_tools_module()
    started = time.perf_counter()
    steps: list[dict[str, Any]] = []

    if _wants_sql(question) or not _wants_rag(question):
        sql_result = tools.run_sql_tool(question)
        steps.append(
            {
                "step": len(steps) + 1,
                "reasoning": "Use the structured financial database for numeric facts and comparisons.",
                "tool": "sql",
                "input": question,
                "output": {
                    "tool": "sql",
                    "ok": sql_result.get("ok"),
                    "status": sql_result.get("status"),
                    "sql": sql_result.get("sql"),
                    "row_count": sql_result.get("row_count"),
                    "rows": sql_result.get("result") or sql_result.get("rows"),
                    "error_message": sql_result.get("error_message"),
                    "latency_sec": sql_result.get("latency_sec"),
                },
            }
        )

    should_call_rag = _wants_rag(question) or not steps or any(
        (step.get("output") or {}).get("error_message") for step in steps
    )
    if should_call_rag:
        ticker, fiscal_year = _question_scope(question)
        rag_result = tools.run_rag_tool(
            question,
            ticker=ticker,
            fiscal_year=fiscal_year,
            db_path=DATA_INDEX / "text_chunks" / "vectors.db",
            assets_path=DATA_INDEX / "merged_assets.json",
            table_db_path=DATA_INDEX / "table_summaries" / "vectors.db",
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "reasoning": "Search filing text and table context for narrative evidence.",
                "tool": "rag",
                "input": question,
                "output": rag_result,
            }
        )

    answer_parts: list[str] = []
    rag_answers = [
        (step.get("output") or {}).get("answer")
        for step in steps
        if (step.get("output") or {}).get("tool") == "rag" and (step.get("output") or {}).get("answer")
    ]
    if rag_answers:
        answer_parts.extend(str(answer).strip() for answer in rag_answers if str(answer).strip())
    for step in steps:
        output = step.get("output") or {}
        if output.get("tool") == "sql":
            sql_answer = _format_sql_answer(output.get("rows"), output.get("sql"))
            if sql_answer:
                answer_parts.insert(0, sql_answer)

    synthesized_answer, synthesis_error = _synthesize_with_llm(question, steps)
    if synthesized_answer:
        answer_parts = [synthesized_answer]

    if not answer_parts:
        errors = [str((step.get("output") or {}).get("error_message")) for step in steps if (step.get("output") or {}).get("error_message")]
        answer_parts.append("I could not complete the tool run." + (f" Error: {errors[0]}" if errors else ""))

    payload = {
        "query": question,
        "answer": "\n\n".join(answer_parts).strip(),
        "mode": "streamlit_direct_tool_fallback",
        "max_steps": len(steps),
        "latency": round(time.perf_counter() - started, 3),
        "steps": steps,
        "references": _references_from_tool_steps(steps),
        "session_id": st.session_state.get("session_id"),
    }
    if original_error is not None:
        payload["fallback_error"] = f"{type(original_error).__name__}: {original_error}"
    if synthesis_error:
        payload["synthesis_error"] = synthesis_error
    return payload


def _run_agent(question: str, *, max_steps: int, session_id: str | None) -> dict[str, Any]:
    try:
        return run_trace_global(question, max_steps=max_steps, session_id=session_id)
    except Exception as exc:
        return _direct_tool_agent(question, original_error=exc)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f7f9;
            --panel: #ffffff;
            --panel-soft: #fbfbfc;
            --line: #e5e7eb;
            --line-strong: #d4d7dd;
            --text: #111827;
            --muted: #6b7280;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --sql: #0d9488;
            --rag: #7c3aed;
            --good: #15803d;
            --bad: #b91c1c;
            --radius: 16px;
            --shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
            --thought: #fffbeb;
            --thought-border: #fde68a;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28rem),
                radial-gradient(circle at top right, rgba(124, 58, 237, 0.05), transparent 22rem),
                var(--bg);
            color: var(--text);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
            display: none;
        }
        .block-container {
            padding-top: 1.1rem;
            max-width: 1680px;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 1rem;
        }
        .agent-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 14px;
        }
        .agent-header h1 {
            margin: 0;
            font-size: clamp(1.5rem, 3vw, 2.1rem);
            letter-spacing: -0.04em;
        }
        .agent-subtitle {
            margin: 0.35rem 0 0;
            color: var(--muted);
            font-size: 0.9rem;
            max-width: 44rem;
        }
        .version-badge {
            display: inline-flex;
            margin-top: 0.45rem;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            background: var(--accent-soft);
            color: #1d4ed8;
            padding: 0.15rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 750;
        }
        .file-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 14px;
            padding: 12px 14px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            color: var(--good);
            font-size: 0.78rem;
            font-weight: 650;
        }
        .panel-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(229, 231, 235, 0.9);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            margin-bottom: 12px;
        }
        .panel-head {
            padding: 14px 16px 12px;
            border-bottom: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.7);
        }
        .panel-title {
            margin: 0;
            font-size: 0.92rem;
            font-weight: 750;
            letter-spacing: -0.01em;
        }
        .panel-note {
            margin: 0.2rem 0 0;
            color: var(--muted);
            font-size: 0.78rem;
        }
        .chat-body {
            padding: 14px 16px;
            min-height: 20rem;
            max-height: 58vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .msg {
            display: flex;
            flex-direction: column;
            max-width: 88%;
            animation: fadeIn 0.18s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: none; }
        }
        .msg.user { align-self: flex-end; align-items: flex-end; }
        .msg.assistant { align-self: flex-start; }
        .msg.system { align-self: center; max-width: 100%; }
        .bubble {
            padding: 0.7rem 0.95rem;
            border-radius: 16px;
            font-size: 0.9rem;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.55;
        }
        .msg.user .bubble {
            background: var(--accent);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .msg.assistant .bubble {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-bottom-left-radius: 5px;
        }
        .msg.system .bubble {
            background: var(--accent-soft);
            border: 1px solid #bfdbfe;
            color: #1e40af;
            font-size: 0.82rem;
            text-align: center;
            border-radius: 12px;
        }
        .msg-meta {
            font-size: 0.68rem;
            color: var(--muted);
            margin-top: 0.2rem;
        }
        .composer-wrap {
            padding: 12px 14px 14px;
            border-top: 1px solid var(--line);
            background: rgba(255,255,255,0.85);
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.78rem;
        }
        div[data-testid="stTextArea"] textarea {
            border: 1px solid var(--line-strong);
            border-radius: 14px;
            background: var(--panel-soft);
            min-height: 5.2rem !important;
            font-size: 0.9rem;
        }
        div[data-testid="stButton"] button {
            border-radius: 999px;
            background: var(--accent);
            color: white;
            border: 0;
            font-weight: 700;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextArea"] label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .tool-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.12rem 0.5rem;
            margin-right: 0.35rem;
            font-size: 0.62rem;
            font-weight: 800;
            text-transform: uppercase;
            color: white;
        }
        .tool-pill.sql { background: var(--sql); }
        .tool-pill.rag { background: var(--rag); }
        .tool-pill.send_email { background: #ea580c; }
        .tool-pill.tool { background: var(--accent); }
        .trace-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            overflow: hidden;
            background: linear-gradient(180deg, #ffffff, #fbfbfc);
            margin-bottom: 10px;
        }
        .trace-head {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
            padding: 8px 10px;
            background: #f3f4f6;
            border-bottom: 1px solid var(--line);
            font-size: 0.76rem;
            font-weight: 750;
        }
        .trace-body-inner {
            padding: 10px;
            font-size: 0.8rem;
        }
        .flow-strip {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: linear-gradient(180deg, #f8fafc, #ffffff);
            padding: 10px 12px;
            margin-bottom: 12px;
        }
        .flow-chip {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.68rem;
            font-weight: 700;
            background: white;
            color: #334155;
        }
        .flow-chip.user { border-color: #bfdbfe; background: var(--accent-soft); color: #1d4ed8; }
        .flow-chip.sql { border-color: #99f6e4; background: #f0fdfa; color: #0f766e; }
        .flow-chip.rag { border-color: #ddd6fe; background: #f5f3ff; color: #6d28d9; }
        .flow-chip.answer { border-color: #bbf7d0; background: #f0fdf4; color: #15803d; }
        .flow-arrow {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
        }
        .source-card {
            border: 1px solid var(--line);
            border-left: 3px solid var(--rag);
            border-radius: 10px;
            padding: 8px 10px;
            background: var(--panel-soft);
            margin-bottom: 8px;
            font-size: 0.78rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: transparent;
            border-bottom: 0;
        }
        .stTabs [data-baseweb="tab"] {
            border: 1px solid var(--line-strong);
            border-radius: 10px;
            background: var(--panel-soft);
            color: var(--muted);
            padding: 0.35rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-soft) !important;
            border-color: #bfdbfe !important;
            color: #1d4ed8 !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff, #fbfbfc);
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_status(status: dict[str, Any]) -> None:
    ready = bool(status.get("ready"))
    st.metric("Knowledge base", "Ready" if ready else "Not ready")
    st.caption(", ".join(status.get("filings") or []))

    table = status.get("table") or {}
    c1, c2 = st.columns(2)
    c1.metric("Filings", status.get("filing_count", 0))
    c2.metric("Financial tables", table.get("row_count", 0))

    if not ready:
        st.error("The deployed app can open, but the knowledge base is not ready.")


def _short_text(value: Any, max_chars: int = 700) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _tool_label(tool: str | None) -> str:
    names = {
        "sql": "SQL",
        "rag": "Filing RAG",
        "send_email": "Email",
    }
    return names.get(tool or "", tool or "Tool")


def _render_tool_flow(steps: list[dict[str, Any]]) -> None:
    if not steps:
        st.info("No tool calls yet.")
        return

    parts = ['<div class="flow-strip"><span class="flow-chip user">Question</span>']
    for step in steps:
        tool = str(step.get("tool") or "tool")
        parts.append('<span class="flow-arrow">→</span>')
        parts.append(
            f'<span class="flow-chip {html.escape(tool)}">'
            f'{html.escape(str(step.get("step", "?")))}. {html.escape(_tool_label(tool))}'
            "</span>"
        )
    parts.append('<span class="flow-arrow">→</span><span class="flow-chip answer">Answer</span></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_references(references: list[dict[str, Any]]) -> None:
    if not references:
        st.info("No source snippets were returned yet.")
        return
    for ref in references:
        title = ref.get("source_file") or "10-K filing"
        header = ref.get("header_path")
        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        if header:
            st.caption(header)
        if ref.get("excerpt"):
            st.write(_short_text(ref["excerpt"], 900))
        if ref.get("score") is not None:
            st.caption(f"score: {ref['score']}")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_step_detail(step: dict[str, Any]) -> None:
    tool = step.get("tool")
    output = step.get("output") or {}

    st.markdown(
        f'<span class="tool-pill {html.escape(str(tool or "tool"))}">{_tool_label(tool)}</span> '
        f'<span class="small-muted">step {step.get("step", "?")}</span>',
        unsafe_allow_html=True,
    )
    if step.get("reasoning"):
        st.markdown("**Reasoning**")
        st.write(step["reasoning"])
    if step.get("input"):
        st.markdown("**Input**")
        st.code(str(step["input"]), language="text")

    if output.get("tool") == "sql" and output.get("sql"):
        st.markdown("**Generated SQL**")
        st.code(output["sql"], language="sql")

    rows = output.get("rows")
    if rows:
        st.markdown("**Result Rows**")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    sources = output.get("sources") or output.get("reranked_top") or []
    if sources:
        st.markdown("**Retrieved Filing Evidence**")
        for source in sources[:5]:
            title = source.get("source_file") or source.get("chunk_id") or "10-K filing"
            st.markdown(f"**{title}**")
            header = source.get("header_path")
            if isinstance(header, list):
                header = " > ".join(str(part) for part in header)
            if header:
                st.caption(header)
            excerpt = source.get("excerpt") or source.get("content")
            if excerpt:
                st.write(_short_text(excerpt, 600))
            if source.get("score") is not None or source.get("rerank_score") is not None:
                st.caption(f"score: {source.get('score') or source.get('rerank_score')}")

    table_contexts = output.get("table_contexts") or []
    if table_contexts:
        st.markdown("**Table Contexts**")
        for table in table_contexts[:5]:
            with st.expander(table.get("table_id") or "table context"):
                if table.get("section_path"):
                    st.caption(table["section_path"])
                if table.get("summary"):
                    st.write(table["summary"])
                st.json(table, expanded=False)

    if output.get("answer"):
        st.markdown("**Tool Answer**")
        st.write(output["answer"])
    if output.get("error_message"):
        st.error(output["error_message"])


def _render_steps(steps: list[dict[str, Any]]) -> None:
    if not steps:
        st.info("No tool calls yet.")
        return
    for step in steps:
        title = f"Step {step.get('step', '?')}: {_tool_label(step.get('tool'))}"
        with st.expander(title):
            _render_step_detail(step)


def main() -> None:
    st.set_page_config(page_title="10-K Financial Research Agent", layout="wide")
    _inject_css()

    status = None
    with st.sidebar:
        st.header("Agent")
        try:
            status = global_index_status()
            _render_status(status)
        except Exception as exc:
            st.error(f"Could not load index status: {exc}")

        st.header("Settings")
        max_steps = st.slider("Max agent steps", min_value=1, max_value=12, value=6)
        show_raw = st.toggle("Show raw JSON", value=False)
        show_debug = st.toggle("Show debug errors", value=False)

    if "question" not in st.session_state:
        st.session_state.question = EXAMPLE_QUESTIONS[0]
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    st.markdown(
        """
        <div class="agent-header">
          <div>
            <h1>Agent Q&amp;A</h1>
            <p class="agent-subtitle">Multi-turn financial research across 10-K filings. Steps and evidence show on the right.</p>
            <span class="version-badge">Markdown answers · SQL · RAG</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if status:
        ready = "Ready" if status.get("ready") else "Not ready"
        filings = ", ".join(status.get("filings") or [])
        st.markdown(
            f'<div class="file-row">Knowledge base: {html.escape(ready)} · {html.escape(filings)}</div>',
            unsafe_allow_html=True,
        )

    result = st.session_state.get("last_result")
    if show_debug and result and result.get("fallback_error"):
        st.warning(f"Direct tool fallback was used: {result['fallback_error']}")
    if show_debug and result and result.get("synthesis_error"):
        st.warning(f"Fallback final LLM synthesis failed: {result['synthesis_error']}")

    answer_col, viz_col = st.columns([1.02, 1], gap="large")

    with answer_col:
        st.markdown(
            """
            <div class="panel-card">
              <div class="panel-head">
                <h2 class="panel-title">Conversation</h2>
                <p class="panel-note">Enter a filing question below. Use ticker/fiscal year for specific filings.</p>
              </div>
              <div class="chat-body">
                <div class="msg system">
                  <div class="bubble">Ask across Apple, Microsoft, Alphabet, and Salesforce filings. The agent can combine structured SQL facts with filing evidence.</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if result:
            user_q = html.escape(result.get("query") or st.session_state.question)
            answer = html.escape(result.get("answer") or "No answer returned.")
            answer = answer.replace("\n", "<br>")
            latency = f"{result['latency']:.2f}s" if result.get("latency") else ""
            st.markdown(
                f"""
                <div class="msg user">
                  <div class="bubble">{user_q}</div>
                </div>
                <div class="msg assistant">
                  <div class="bubble">{answer}</div>
                  <div class="msg-meta">{html.escape(latency)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
              </div>
              <div class="composer-wrap">
            """,
            unsafe_allow_html=True,
        )

        example = st.selectbox("Examples", EXAMPLE_QUESTIONS)
        col_use, col_new = st.columns([1, 1])
        with col_use:
            if st.button("Use example"):
                st.session_state.question = example
                st.rerun()
        with col_new:
            if st.button("New chat"):
                st.session_state.last_result = None
                st.session_state.session_id = None
                st.rerun()

        question = st.text_area("Question", key="question", height=110)
        run = st.button("Run agent", type="primary", disabled=not question.strip())
        st.markdown('<div class="small-muted">Same thread remembers prior answers · Shift+Enter for newline</div>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        if result:
            st.markdown('<div class="panel-card"><div class="panel-head"><h2 class="panel-title">Sources</h2><p class="panel-note">Retrieved filing snippets used by the answer</p></div><div style="padding:12px 14px 16px;">', unsafe_allow_html=True)
            _render_references(result.get("references") or [])
            st.markdown("</div></div>", unsafe_allow_html=True)

        if run:
            with st.spinner("Running agent..."):
                try:
                    result = _run_agent(
                        question.strip(),
                        max_steps=max_steps,
                        session_id=st.session_state.session_id,
                    )
                    st.session_state.session_id = result.get("session_id")
                    st.session_state.last_result = result
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
                    if show_debug:
                        st.exception(exc)
                        st.code(traceback.format_exc(), language="text")
                    st.stop()

    with viz_col:
        st.markdown(
            '<div class="panel-card"><div class="panel-head"><h2 class="panel-title">Tool steps</h2><p class="panel-note">Reasoning → tool params → observation → next step</p></div><div style="padding:12px 14px 16px;">',
            unsafe_allow_html=True,
        )
        tab_flow, tab_trace, tab_data = st.tabs(["Execution path", "Tool Trace", "Data"])
        steps = result.get("steps") if result else []
        with tab_flow:
            _render_tool_flow(steps or [])
        with tab_trace:
            _render_steps(steps or [])
        with tab_data:
            rows_rendered = False
            for step in steps or []:
                output = step.get("output") or {}
                rows = output.get("rows")
                if rows:
                    rows_rendered = True
                    st.markdown(f"**Step {step.get('step', '?')} SQL rows**")
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                table_contexts = output.get("table_contexts") or []
                if table_contexts:
                    rows_rendered = True
                    st.markdown(f"**Step {step.get('step', '?')} table contexts**")
                    st.json(table_contexts, expanded=False)
            if not rows_rendered:
                st.info("No structured rows or table contexts returned yet.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    if show_raw:
        st.subheader("Raw result")
        st.code(json.dumps(result, indent=2, ensure_ascii=False, default=str), language="json")


if __name__ == "__main__":
    main()

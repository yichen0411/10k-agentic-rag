"""LangChain tool-calling agent over SQL and filing RAG tools."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langsmith_tracing import (
    build_langsmith_run_config,
    capture_langsmith_run_link,
    configure_langsmith_tracing,
    make_langchain_tracer,
)
from system_prompt import build_langchain_system_prompt
from tools import TOOL_SCHEMA, run_rag_tool, run_send_email_tool, run_sql_tool
from trace_format import format_trace_item


AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
PROJECT_ROOT = MAIN_ROOT.parent
INFERENCE_DIR = MAIN_ROOT / "inference"
LOG_PATH = AGENT_DIR / "agent_log.jsonl"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from text_vector_rag_inference import (  # noqa: E402
    DEFAULT_FIREWORKS_CHAT_MODEL,
    FIREWORKS_BASE_URL,
    call_chat,
    load_env_file,
    resolve_llm_provider,
)

_MAX_ITERATION_MARKERS = (
    "agent stopped due to max iterations",
    "stopped due to max iterations",
)
_RAG_BOILERPLATE_MARKERS = (
    "the provided context does not",
    "does not support a confident answer",
    "insufficient context",
    "none of the supplied chunks",
    "does not describe the strategic",
)
_USER_META_LINE_PATTERNS = (
    re.compile(r"^.*\b(sql|rag|sqlite|segment_revenue|income_statements)\b.*$", re.I | re.M),
    re.compile(r"^.*\b(insufficient_context|fallback_success|retrieved chunks?|tool observation)\b.*$", re.I | re.M),
    re.compile(r"^.*\b(sql query|query results?|see sql|from the sql)\b.*$", re.I | re.M),
    re.compile(r"^.*\b(growth percentages were calculated|data come from the)\b.*$", re.I | re.M),
    re.compile(r"^.*\b(the rag search returned|after a retry)\b.*$", re.I | re.M),
    re.compile(r"^.*\bcalculated from sql\b.*$", re.I | re.M),
)


def _resolve_agent_model() -> tuple[str, str]:
    provider = resolve_llm_provider()
    if provider == "fireworks":
        model = (
            os.environ.get("FW_AGENT_MODEL")
            or os.environ.get("FW_CHAT_MODEL")
            or DEFAULT_FIREWORKS_CHAT_MODEL
        )
        return provider, model
    model = os.environ.get(
        "ANTHROPIC_AGENT_MODEL",
        os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5-20251001"),
    )
    return provider, model


def _build_llm():
    provider, model = _resolve_agent_model()
    if provider == "fireworks":
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required when LLM_PROVIDER=fireworks.")
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=1200,
            api_key=api_key,
            base_url=FIREWORKS_BASE_URL,
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
    return ChatAnthropic(
        model=model,
        temperature=0.1,
        max_tokens=1200,
        api_key=api_key,
    )


class SqlToolInput(BaseModel):
    question: str = Field(
        ...,
        description=next(t["parameters"]["question"]["description"] for t in TOOL_SCHEMA if t["name"] == "sql"),
    )


class RagToolInput(BaseModel):
    question: str = Field(
        ...,
        description=next(t["parameters"]["question"]["description"] for t in TOOL_SCHEMA if t["name"] == "rag"),
    )
    ticker: Optional[str] = Field(
        default=None,
        description=next(t["parameters"]["ticker"]["description"] for t in TOOL_SCHEMA if t["name"] == "rag"),
    )
    fiscal_year: Optional[str] = Field(
        default=None,
        description=next(t["parameters"]["fiscal_year"]["description"] for t in TOOL_SCHEMA if t["name"] == "rag"),
    )


class SendEmailToolInput(BaseModel):
    to_email: str = Field(
        ...,
        description=next(t["parameters"]["to_email"]["description"] for t in TOOL_SCHEMA if t["name"] == "send_email"),
    )
    subject: str = Field(
        ...,
        description=next(t["parameters"]["subject"]["description"] for t in TOOL_SCHEMA if t["name"] == "send_email"),
    )
    body: str = Field(
        ...,
        description=next(t["parameters"]["body"]["description"] for t in TOOL_SCHEMA if t["name"] == "send_email"),
    )


def _compact_sql_payload(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("result")
    if isinstance(rows, list) and len(rows) > 50:
        rows = rows[:50]
    return {
        "tool": "sql",
        "ok": result.get("ok"),
        "question": result.get("input"),
        "row_count": result.get("row_count"),
        "rows": rows,
        "error_message": result.get("error_message"),
        "latency_sec": result.get("latency_sec"),
    }


def _compact_email_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": "send_email",
        "ok": result.get("ok"),
        "status": result.get("status"),
        "to_email": result.get("to_email"),
        "subject": result.get("subject"),
        "body_chars": result.get("body_chars"),
        "error_message": result.get("error_message"),
        "latency_sec": result.get("latency_sec"),
    }


def _is_rag_boilerplate(answer: str | None) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in _RAG_BOILERPLATE_MARKERS)


def _filing_excerpts_from_rag(result: dict[str, Any], *, limit: int = 3) -> list[str]:
    excerpts: list[str] = []
    for chunk in (result.get("expanded_context") or [])[:limit]:
        excerpt = (chunk.get("excerpt") or chunk.get("content") or "").strip()
        header = " › ".join(chunk.get("header_path") or [])
        if excerpt:
            excerpts.append(f"{header}: {excerpt[:500]}" if header else excerpt[:500])
    if excerpts:
        return excerpts
    for chunk in (result.get("reranked_top") or [])[:limit]:
        excerpt = (chunk.get("excerpt") or "").strip()
        if excerpt:
            excerpts.append(excerpt[:500])
    return excerpts


def _user_facing_rag_narrative(result: dict[str, Any]) -> str:
    answer = (result.get("answer") or "").strip()
    excerpts = _filing_excerpts_from_rag(result)
    if _is_rag_boilerplate(answer):
        if excerpts:
            return "\n\n".join(excerpts)
        return "The filing excerpts here do not directly address this wording."
    return answer


def _compact_rag_payload(result: dict[str, Any]) -> dict[str, Any]:
    tool_input = result.get("input")
    if isinstance(tool_input, str):
        tool_input = {"question": tool_input}
    return {
        "tool": "rag",
        "ok": result.get("ok"),
        "question": tool_input,
        "filing_narrative": _user_facing_rag_narrative(result),
        "filing_excerpts": _filing_excerpts_from_rag(result),
        "expanded_context": (result.get("expanded_context") or [])[:8],
        "reranked_top": (result.get("reranked_top") or [])[:3],
        "table_contexts": (result.get("table_contexts") or [])[:5],
        "scope_filters": {
            "ticker_filter": (result.get("scope_filters") or {}).get("ticker_filter"),
            "fiscal_year_filter": (result.get("scope_filters") or {}).get("fiscal_year_filter"),
            "filtered_chunk_count": (result.get("scope_filters") or {}).get("filtered_chunk_count"),
            "filtered_table_chunk_count": (result.get("scope_filters") or {}).get("filtered_table_chunk_count"),
        },
        "pipeline_latency": result.get("pipeline_latency") or {},
        "error_message": result.get("error_message"),
        "latency_sec": result.get("latency_sec"),
    }


def build_langchain_tools(
    *,
    rag_db_path: Path | None = None,
    rag_assets_path: Path | None = None,
    rag_table_db_path: Path | None = None,
) -> list[StructuredTool]:
    def _sql_tool_fn(question: str) -> str:
        _load_env()
        return json.dumps(_compact_sql_payload(run_sql_tool(question)), ensure_ascii=False)

    def _rag_tool_fn(question: str, ticker: str | None = None, fiscal_year: str | None = None) -> str:
        _load_env()
        return json.dumps(
            _compact_rag_payload(
                run_rag_tool(
                    question,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    db_path=rag_db_path,
                    assets_path=rag_assets_path,
                    table_db_path=rag_table_db_path,
                )
            ),
            ensure_ascii=False,
        )

    def _email_tool_fn(to_email: str, subject: str, body: str) -> str:
        _load_env()
        return json.dumps(
            _compact_email_payload(run_send_email_tool(to_email, subject, body)),
            ensure_ascii=False,
        )

    def _tool_meta(name: str) -> dict[str, Any]:
        return next(t for t in TOOL_SCHEMA if t["name"] == name)

    return [
        StructuredTool.from_function(
            func=_sql_tool_fn,
            name="sql",
            description=_tool_meta("sql")["description"],
            args_schema=SqlToolInput,
        ),
        StructuredTool.from_function(
            func=_rag_tool_fn,
            name="rag",
            description=_tool_meta("rag")["description"],
            args_schema=RagToolInput,
        ),
        StructuredTool.from_function(
            func=_email_tool_fn,
            name="send_email",
            description=_tool_meta("send_email")["description"],
            args_schema=SendEmailToolInput,
        ),
    ]


def _load_env() -> None:
    load_env_file(PROJECT_ROOT / ".env")


def _coerce_chat_history(raw: list[dict[str, str]] | None) -> list[Any]:
    if not raw:
        return []
    messages: list[Any] = []
    for item in raw:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _format_action_input(tool_name: str, tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        if tool_name == "send_email":
            return (
                f"To: {tool_input.get('to_email', '')}\n"
                f"Subject: {tool_input.get('subject', '')}"
            )
        return tool_input.get("question") or json.dumps(tool_input, ensure_ascii=False)
    return str(tool_input).strip()


def _sanitize_user_answer(answer: str | None) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    for pattern in _USER_META_LINE_PATTERNS:
        text = "\n".join(line for line in text.splitlines() if not pattern.match(line.strip()))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(
        r"The (?:rag )?search returned[^.]*\.",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"after a retry[^.]*\.",
        "",
        text,
        flags=re.I,
    ).strip()
    return text


def _hit_max_iterations(answer: str | None) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in _MAX_ITERATION_MARKERS)


def _format_tool_evidence_for_synthesis(tool_outputs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, output in enumerate(tool_outputs, 1):
        tool = output.get("tool")
        if tool == "sql" and output.get("ok"):
            rows = output.get("rows") or []
            preview = json.dumps(rows[:20], ensure_ascii=False, indent=2)
            blocks.append(
                f"### SQL step {index}\n"
                f"Question: {output.get('input')}\n"
                f"SQL: {output.get('sql')}\n"
                f"Rows ({output.get('row_count', len(rows))}):\n{preview}"
            )
        elif tool == "rag":
            inp = output.get("question") or output.get("input") or {}
            question = inp.get("question") if isinstance(inp, dict) else inp
            ticker = inp.get("ticker") if isinstance(inp, dict) else ""
            fiscal_year = inp.get("fiscal_year") if isinstance(inp, dict) else ""
            narrative = output.get("filing_narrative") or output.get("answer") or ""
            blocks.append(
                f"### Filing step {index}\n"
                f"Topic: {question}\n"
                f"Company/year: {ticker} / {fiscal_year}\n"
                f"Filing narrative:\n{narrative[:3500]}"
            )
    return "\n\n".join(blocks)


def _synthesize_final_answer_from_tools(
    query: str,
    tool_outputs: list[dict[str, Any]],
    *,
    reason: str,
) -> str:
    evidence = _format_tool_evidence_for_synthesis(tool_outputs)
    if not evidence.strip():
        return ""
    system = (
        "You are a financial research assistant. Write a concise, executive-ready final answer "
        "using ONLY the evidence below. Present numbers with fiscal years and segment names. "
        "Present filing content as natural prose ('In its FY2025 10-K, Apple...'). "
        "Never mention sql, rag, tools, chunks, retries, insufficient_context, table names, "
        "or how data was gathered. If qualitative wording is missing, say so in one plain sentence."
    )
    prompt = (
        f"User question:\n{query.strip()}\n\n"
        f"Collected tool evidence ({reason}):\n{evidence}\n\n"
        "Write the best grounded final answer now."
    )
    _, model = _resolve_agent_model()
    return call_chat(prompt, system=system, max_tokens=1400, chat_model=model).strip()


def _normalize_answer(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        parts: list[str] = []
        for block in output:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(output).strip()


def _build_executor(
    max_steps: int,
    *,
    rag_db_path: Path | None = None,
    rag_assets_path: Path | None = None,
    rag_table_db_path: Path | None = None,
    memory_context: str = "",
) -> AgentExecutor:
    llm = _build_llm()
    tools = build_langchain_tools(
        rag_db_path=rag_db_path,
        rag_assets_path=rag_assets_path,
        rag_table_db_path=rag_table_db_path,
    )
    system = build_langchain_system_prompt()
    if memory_context.strip():
        system = (
            system
            + "\n\n## Session memory (longer horizon)\n"
            + memory_context.strip()
            + "\n\nUse session memory for follow-ups and user-specific facts (e.g. email). "
            "Episodic facts are authoritative; ignore semantic lines that do not match the current question. "
            "Do not invent facts not listed here or in chat_history."
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_steps,
        early_stopping_method="force",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def _trace_from_intermediate_steps(
    intermediate_steps: list[tuple[Any, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    tool_outputs: list[dict[str, Any]] = []

    for index, (action, observation) in enumerate(intermediate_steps, 1):
        tool_name = getattr(action, "tool", None) or "unknown"
        raw_input = getattr(action, "tool_input", "")

        try:
            parsed_obs = json.loads(observation)
        except json.JSONDecodeError:
            parsed_obs = {"raw": observation}

        parsed_obs["step_index"] = index
        tool_outputs.append(parsed_obs)
        trace.append(
            {
                "step": index,
                "thought": getattr(action, "log", "") or "",
                "action": tool_name,
                "action_input": raw_input if isinstance(raw_input, dict) else _format_action_input(tool_name, raw_input),
                "observation": parsed_obs,
            }
        )
    return trace, tool_outputs


def _messages_to_conversation(query: str, intermediate_steps: list[tuple[Any, str]], answer: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "user", "content": query}]
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "tool")
        tool_input = getattr(action, "tool_input", "")
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": tool_name,
                        "action_input": tool_input,
                        "log": getattr(action, "log", ""),
                    },
                    ensure_ascii=False,
                ),
            }
        )
        messages.append({"role": "user", "content": f"Tool observation ({tool_name}):\n{observation}"})
    messages.append({"role": "assistant", "content": answer})
    return {"query": query, "messages": messages}


def _log_run(payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": payload.get("query"),
        "mode": payload.get("mode"),
        "trace": payload.get("trace"),
        "conversation": payload.get("conversation"),
        "tool_count": len(payload.get("tool_outputs", [])),
        "latency": payload.get("latency"),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _StreamingTraceHandler(BaseCallbackHandler):
    """Emit UI events as each tool finishes (and when a tool is chosen)."""

    def __init__(self, on_event: Callable[[dict[str, Any]], None]) -> None:
        self._on_event = on_event
        self._step_index = 0
        self._pending_action: Any = None

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        self._pending_action = action
        self._step_index += 1
        tool_name = getattr(action, "tool", None) or "unknown"
        raw_input = getattr(action, "tool_input", "")
        partial = format_trace_item(
            {
                "step": self._step_index,
                "thought": getattr(action, "log", "") or "",
                "action": tool_name,
                "action_input": raw_input,
                "observation": None,
                "pending": True,
            }
        )
        if partial:
            partial["pending"] = True
            self._on_event({"type": "step_start", "step": partial})

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        action = self._pending_action
        tool_name = getattr(action, "tool", None) or "unknown" if action else "unknown"
        raw_input = getattr(action, "tool_input", "") if action else ""
        try:
            parsed_obs = json.loads(output)
        except json.JSONDecodeError:
            parsed_obs = {"raw": output}
        trace_row = {
            "step": self._step_index,
            "thought": getattr(action, "log", "") or "" if action else "",
            "action": tool_name,
            "action_input": raw_input,
            "observation": parsed_obs,
        }
        formatted = format_trace_item(trace_row)
        if formatted:
            formatted["pending"] = False
            self._on_event({"type": "step", "step": formatted})
        self._pending_action = None


def run_langchain_agent(
    query: str,
    max_steps: int = 5,
    *,
    rag_db_path: Path | None = None,
    rag_assets_path: Path | None = None,
    rag_table_db_path: Path | None = None,
    mode_label: str = "langchain",
    on_event: Callable[[dict[str, Any]], None] | None = None,
    chat_history: list[dict[str, str]] | None = None,
    memory_context: str = "",
    session_id: str | None = None,
    file_id: str | None = None,
) -> dict[str, Any]:
    _load_env()
    langsmith_status = configure_langsmith_tracing()
    start = time.perf_counter()
    executor = _build_executor(
        max_steps=max_steps,
        rag_db_path=rag_db_path,
        rag_assets_path=rag_assets_path,
        rag_table_db_path=rag_table_db_path,
        memory_context=memory_context,
    )
    callbacks: list[Any] = []
    langsmith_tracer = None
    if langsmith_status.get("enabled"):
        langsmith_tracer = make_langchain_tracer(langsmith_status.get("project") or "fireworks-agentic-rag")
        callbacks.append(langsmith_tracer)
    if on_event:
        callbacks.append(_StreamingTraceHandler(on_event))
    config = build_langsmith_run_config(
        query=query,
        mode_label=mode_label,
        session_id=session_id,
        file_id=file_id,
        callbacks=callbacks or None,
    )
    result = executor.invoke(
        {
            "input": query,
            "chat_history": _coerce_chat_history(chat_history),
        },
        config=config,
    )
    intermediate_steps = result.get("intermediate_steps") or []
    trace, tool_outputs = _trace_from_intermediate_steps(intermediate_steps)
    answer = _sanitize_user_answer(_normalize_answer(result.get("output")))
    stopped_reason: str | None = None
    if _hit_max_iterations(answer) and tool_outputs:
        stopped_reason = "max_iterations"
        synthesized = _synthesize_final_answer_from_tools(
            query,
            tool_outputs,
            reason="partial run",
        )
        if synthesized:
            answer = _sanitize_user_answer(synthesized)
    elif not answer.strip() and tool_outputs:
        stopped_reason = "empty_output"
        synthesized = _synthesize_final_answer_from_tools(
            query,
            tool_outputs,
            reason="tool evidence only",
        )
        if synthesized:
            answer = _sanitize_user_answer(synthesized)

    payload = {
        "query": query,
        "mode": mode_label,
        "max_steps": max_steps,
        "stopped_reason": stopped_reason,
        "trace": trace,
        "conversation": _messages_to_conversation(query, intermediate_steps, answer),
        "tool_outputs": tool_outputs,
        "answer": answer,
        "latency": {
            "total_sec": round(time.perf_counter() - start, 3),
            "tool_sec": round(sum(float(output.get("latency_sec", 0)) for output in tool_outputs), 3),
        },
    }
    if langsmith_status.get("enabled"):
        payload["langsmith"] = {
            "project": langsmith_status.get("project"),
            **capture_langsmith_run_link(langsmith_tracer),
        }
    _log_run(payload)
    return payload

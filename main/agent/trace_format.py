"""Format LangChain agent traces for the Q&A UI."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def _short_text(value: Any, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


def _unescape_text(raw: str) -> str:
    return (
        raw.replace("\\n", "\n")
        .replace("\\'", "'")
        .replace('\\"', '"')
        .replace("\\t", "\t")
    )


def parse_agent_log(thought: str) -> dict[str, Any]:
    """Extract reasoning text and tool call metadata from LangChain action logs."""
    thought = (thought or "").strip()
    out: dict[str, Any] = {"reasoning": "", "invoke_tool": None, "invoke_question": None}
    if not thought:
        return out

    invoke = re.search(
        r"Invoking:\s*[`']?(\w+)[`']?\s+with\s+[`'](\{.*?\})[`']",
        thought,
        re.DOTALL,
    )
    if invoke:
        out["invoke_tool"] = invoke.group(1)
        blob = invoke.group(2).replace("`", "")
        try:
            payload = ast.literal_eval(blob)
            if isinstance(payload, dict):
                q = payload.get("question")
                if q:
                    out["invoke_question"] = str(q).strip()
        except (SyntaxError, ValueError):
            qm = re.search(r"['\"]question['\"]\s*:\s*['\"](.+?)['\"]", blob)
            if qm:
                out["invoke_question"] = _unescape_text(qm.group(1))

    resp = re.search(r"responded:\s*(\[.*\])\s*$", thought, re.DOTALL)
    if resp:
        try:
            blocks = ast.literal_eval(resp.group(1))
            texts: list[str] = []
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text"):
                        texts.append(str(block["text"]).strip())
            out["reasoning"] = "\n\n".join(texts).strip()
        except (SyntaxError, ValueError):
            pass

    if not out["reasoning"]:
        # Legacy single-quoted text blobs
        texts = re.findall(r"'text': '((?:[^'\\]|\\.)*)'", thought)
        if texts:
            out["reasoning"] = "\n\n".join(_unescape_text(t) for t in texts).strip()

    if not out["reasoning"]:
        cleaned = re.sub(r"Invoking:.*", "", thought, flags=re.DOTALL)
        cleaned = re.sub(r"responded:.*", "", cleaned, flags=re.DOTALL).strip()
        if cleaned and len(cleaned) < 400 and "partial_json" not in cleaned:
            out["reasoning"] = cleaned

    return out


def extract_responded_text(thought: str) -> str:
    """Backward-compatible alias: reasoning only, no raw JSON."""
    return parse_agent_log(thought)["reasoning"]


def _format_money(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) >= 1e9:
        return f"${num / 1e9:.2f}B"
    if abs(num) >= 1e6:
        return f"${num / 1e6:.1f}M"
    return f"${num:,.0f}"


def compact_tool_output(observation: dict[str, Any] | None) -> dict[str, Any]:
    if not observation:
        return {}
    tool = observation.get("tool")
    if tool == "sql":
        rows = observation.get("result") or observation.get("rows")
        if isinstance(rows, list) and len(rows) > 30:
            rows = rows[:30]
        display_rows = rows
        if isinstance(rows, list):
            display_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    display_rows.append(row)
                    continue
                pretty = dict(row)
                for key, val in pretty.items():
                    if key in ("revenue", "net_income", "operating_income", "total_assets") and isinstance(
                        val, (int, float)
                    ):
                        pretty[key] = _format_money(val)
                display_rows.append(pretty)
        return {
            "tool": "sql",
            "ok": observation.get("ok"),
            "status": observation.get("status"),
            "sql": observation.get("sql"),
            "row_count": observation.get("row_count"),
            "rows": display_rows,
            "error_message": observation.get("error_message"),
            "latency_sec": observation.get("latency_sec"),
        }
    if tool == "send_email":
        return {
            "tool": "send_email",
            "ok": observation.get("ok"),
            "status": observation.get("status"),
            "to_email": observation.get("to_email"),
            "subject": observation.get("subject"),
            "body_chars": observation.get("body_chars"),
            "error_message": observation.get("error_message"),
            "latency_sec": observation.get("latency_sec"),
        }
    if tool == "rag":
        answer = observation.get("answer") or ""
        if isinstance(answer, str) and len(answer) > 2000:
            answer = answer[:2000] + "…"
        context_by_id = {
            c.get("chunk_id"): c
            for c in observation.get("expanded_context") or []
            if c.get("chunk_id")
        }
        sources = []
        for c in (observation.get("reranked_top") or [])[:5]:
            chunk_id = c.get("chunk_id") or c.get("candidate_id")
            ctx = context_by_id.get(chunk_id) or {}
            sources.append(
                {
                    "chunk_id": chunk_id,
                    "header_path": " › ".join(c.get("header_path") or ctx.get("header_path") or []),
                    "score": c.get("score") or c.get("rerank_score"),
                    "source_file": c.get("source_file") or ctx.get("source_file"),
                    "excerpt": _short_text(c.get("excerpt") or ctx.get("excerpt")),
                }
            )
        seen_source_ids = {source.get("chunk_id") for source in sources if source.get("chunk_id")}
        for c in (observation.get("expanded_context") or [])[:8]:
            chunk_id = c.get("chunk_id")
            excerpt = _short_text(c.get("excerpt"))
            if not chunk_id or chunk_id in seen_source_ids or not excerpt:
                continue
            sources.append(
                {
                    "chunk_id": chunk_id,
                    "header_path": " › ".join(c.get("header_path") or []),
                    "score": None,
                    "source_file": c.get("source_file"),
                    "excerpt": excerpt,
                }
            )
            seen_source_ids.add(chunk_id)
            if len(sources) >= 5:
                break
        tool_input = observation.get("input")
        if isinstance(tool_input, str):
            tool_input = {"question": tool_input}
        scope_filters = observation.get("scope_filters") or {}
        pipeline_latency = observation.get("pipeline_latency") or {}
        table_contexts = []
        for table in (observation.get("table_contexts") or [])[:5]:
            table_contexts.append(
                {
                    "table_id": table.get("table_id"),
                    "section_path": table.get("section_path"),
                    "summary": (table.get("summary") or "")[:240],
                    "has_markdown": table.get("has_markdown"),
                }
            )
        return {
            "tool": "rag",
            "ok": observation.get("ok"),
            "input": tool_input,
            "scope_filters": scope_filters,
            "pipeline_latency": pipeline_latency,
            "answer": answer,
            "sources": sources,
            "table_contexts": table_contexts,
            "error_message": observation.get("error_message"),
            "latency_sec": observation.get("latency_sec"),
        }
    return observation


def answer_references_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect answer-level references from RAG observations for the chat bubble."""
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in steps:
        out = step.get("output") or {}
        if out.get("tool") != "rag":
            continue
        for source in out.get("sources") or []:
            key = str(source.get("chunk_id") or source.get("source_file") or source.get("header_path") or "")
            if not key or key in seen:
                continue
            refs.append(
                {
                    "kind": "chunk",
                    "chunk_id": source.get("chunk_id"),
                    "source_file": source.get("source_file") or "10-K filing",
                    "header_path": source.get("header_path") or "",
                    "excerpt": source.get("excerpt") or "",
                    "score": source.get("score"),
                }
            )
            seen.add(key)
            if len(refs) >= 4:
                return refs
    return refs


def _normalize_action_input(action: str, action_input: Any) -> tuple[str, dict[str, Any] | None]:
    if isinstance(action_input, dict):
        params = dict(action_input)
        if action == "send_email":
            display = (
                f"To: {params.get('to_email', '')}\n"
                f"Subject: {params.get('subject', '')}"
            )
        elif action == "rag":
            parts = [params.get("question") or ""]
            if params.get("ticker"):
                parts.append(f"ticker={params.get('ticker')}")
            if params.get("fiscal_year"):
                parts.append(f"fiscal_year={params.get('fiscal_year')}")
            display = "\n".join(p for p in parts if p)
        else:
            display = params.get("question") or json.dumps(params, ensure_ascii=False)
        return display.strip(), params
    text = str(action_input or "").strip()
    return text, {"question": text} if text else None


def format_trace_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Format one LangChain trace row for the UI."""
    action = item.get("action")
    if action == "final_answer":
        return None

    log = parse_agent_log(str(item.get("thought") or ""))
    raw_input = item.get("action_input")
    display_input, input_params = _normalize_action_input(action or "", raw_input)
    if not display_input:
        display_input = str(log.get("invoke_question") or "").strip()

    return {
        "step": item.get("step"),
        "reasoning": log.get("reasoning") or "",
        "tool": action,
        "input": display_input,
        "input_params": input_params,
        "output": compact_tool_output(item.get("observation")),
        "thought_before": "",
        "pending": item.get("pending", False),
    }


def build_scratchpad_prompt_faithful(query: str, trace: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Scratchpad as seen by the model (raw tool JSON, not UI-compacted)."""
    items: list[dict[str, Any]] = [
        {"role": "human", "kind": "question", "content": (query or "").strip()},
    ]
    for row in trace or []:
        action = row.get("action")
        if not action or action == "final_answer":
            continue
        step = row.get("step")
        log = parse_agent_log(str(row.get("thought") or ""))
        action_input = row.get("action_input")
        if isinstance(action_input, dict):
            action_input = action_input.get("question") or json.dumps(action_input, ensure_ascii=False)
        action_input = str(action_input or log.get("invoke_question") or "").strip()
        items.append(
            {
                "role": "assistant",
                "kind": "tool_call",
                "step": step,
                "tool": action,
                "input": action_input,
                "reasoning": log.get("reasoning") or "",
            }
        )
        items.append(
            {
                "role": "tool",
                "kind": "observation",
                "step": step,
                "tool": action,
                "content": row.get("observation"),
            }
        )
    return items


def build_scratchpad_display(query: str, trace: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Shape LangChain intermediate steps for the Memory UI (readable compaction)."""
    items: list[dict[str, Any]] = [
        {
            "role": "human",
            "kind": "question",
            "content": (query or "").strip(),
        }
    ]
    for row in trace or []:
        action = row.get("action")
        if not action or action == "final_answer":
            continue
        step = row.get("step")
        log = parse_agent_log(str(row.get("thought") or ""))
        action_input = row.get("action_input")
        if isinstance(action_input, dict):
            action_input = action_input.get("question") or json.dumps(action_input, ensure_ascii=False)
        action_input = str(action_input or log.get("invoke_question") or "").strip()
        items.append(
            {
                "role": "assistant",
                "kind": "tool_call",
                "step": step,
                "tool": action,
                "input": action_input,
                "reasoning": log.get("reasoning") or "",
            }
        )
        observation = row.get("observation")
        if isinstance(observation, dict):
            body = compact_tool_output(observation) or observation
        else:
            body = observation
        items.append(
            {
                "role": "tool",
                "kind": "observation",
                "step": step,
                "tool": action,
                "content": body,
            }
        )
    return items


def format_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Shape agent run result for the trace visualization UI."""
    steps: list[dict[str, Any]] = []
    for item in payload.get("trace") or []:
        formatted = format_trace_item(item)
        if formatted:
            steps.append(formatted)
    query = payload.get("query") or ""
    trace = payload.get("trace") or []
    return {
        "query": query,
        "answer": payload.get("answer"),
        "mode": payload.get("mode"),
        "max_steps": payload.get("max_steps"),
        "latency": payload.get("latency"),
        "steps": steps,
        "references": answer_references_from_steps(steps),
        "scratchpad": build_scratchpad_display(query, trace),
        "trace_raw": trace,
    }


def build_prompt_injection_record(
    *,
    query: str,
    memory_context: str,
    chat_history: list[dict[str, str]],
    trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Exact user-memory slices passed into run_langchain_agent for one turn."""
    return {
        "query": (query or "").strip(),
        "memory_context": (memory_context or "").strip(),
        "chat_history": chat_history or [],
        "scratchpad": build_scratchpad_prompt_faithful(query, trace),
        "system_note": (
            "Also sent every LLM call (not duplicated here): base agent instructions, "
            "SQLite schema, filing coverage, and tool catalog from build_langchain_system_prompt()."
        ),
    }

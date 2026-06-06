from __future__ import annotations

import inspect
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable


AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
PROJECT_ROOT = MAIN_ROOT.parent
SQL_DIR = MAIN_ROOT / "sql"
INFERENCE_DIR = MAIN_ROOT / "inference"
CHUNKING_DIR = MAIN_ROOT / "chunking"

for path in [str(SQL_DIR), str(INFERENCE_DIR), str(CHUNKING_DIR), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from text_to_sql import answer_sql_question  # noqa: E402
from filing_metadata import normalize_filter_values  # noqa: E402
from filing_scope import (  # noqa: E402
    comparative_table_retrieval_query,
    discover_indexed_filing_years,
    extract_metric_years,
    resolve_filing_year_filter,
    should_retry_with_newer_filing,
)
from rag_pipeline_config import (  # noqa: E402
    STUDIO_BM25_TOP_K,
    STUDIO_MAX_CONTEXT_CHUNKS,
    STUDIO_MAX_TABLE_CONTEXTS,
    STUDIO_RERANK_TOP_N,
    STUDIO_TABLE_SIMILARITY_THRESHOLD,
    STUDIO_TABLE_VECTOR_TOP_K,
    STUDIO_VECTOR_TOP_K,
)
from text_vector_rag_inference import (  # noqa: E402
    DEFAULT_ASSETS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_DB,
    DEFAULT_EMBED_MODEL,
    run_pipeline,
)

from email_delivery import send_answer_email  # noqa: E402


def _coerce_scope_filter(value: str | list[str] | None) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item or "").strip()]
        return cleaned or None
    text = str(value).strip()
    if not text:
        return None
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return parts if len(parts) > 1 else parts[0]
    return text


def _is_multi_filter(value: str | list[str] | None) -> bool:
    return isinstance(value, list) and len(value) > 1


INSUFFICIENT_ANSWER_MARKERS = (
    "does not support a confident answer",
    "does not explicitly",
    "does not mention",
    "not explicitly stated",
    "not found",
    "insufficient context",
    "cannot determine",
    "unable to determine",
    "not broken down",
    "does not break down",
    "does not allocate",
    "what's not available",
)


def _normalize_single_filing_year(value: str | list[str] | None) -> str | None:
    cleaned = normalize_filter_values(_coerce_scope_filter(value), kind="fiscal_year")
    if isinstance(cleaned, list):
        return cleaned[0] if len(cleaned) == 1 else None
    return cleaned


def _available_filing_years(db_path: Path | None) -> list[str]:
    return discover_indexed_filing_years(db_path)


def _looks_insufficient_answer(answer: str | None) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in INSUFFICIENT_ANSWER_MARKERS)


def _has_substantive_evidence(answer: str | None) -> bool:
    text = answer or ""
    lowered = text.lower()
    return (
        "::text_" in text
        or "chunk_id:" in lowered
        or "table_" in lowered
        or "disclosed" in lowered and len(text) > 300
    )


def _fallback_retrieval_query(question: str) -> str:
    return (
        "Rewrite this filing question as a retrieval query that preserves the same intent, "
        "company, fiscal period, entities, and requested metric/topic, but uses concrete "
        "10-K wording, synonyms, and related filing terms likely to appear in the document. "
        "Do not broaden the scope or answer the question. Original question: "
        f"{question.strip()} "
    )


def _run_rag_pipeline(
    question: str,
    *,
    retrieval_query: str | None = None,
    db_path: Path | None,
    assets_path: Path | None,
    table_db_path: Path | None,
    vector_top_k: int,
    rerank_top_n: int,
    max_context_chunks: int,
    chat_model: str,
    table_vector_top_k: int,
    table_similarity_threshold: float,
    max_table_contexts: int | None,
    ticker_filter: list[str] | None,
    fiscal_year_filter: list[str] | None,
) -> dict[str, Any]:
    kwargs = {
        "db_path": db_path or DEFAULT_DB,
        "vector_top_k": vector_top_k,
        "bm25_top_k": STUDIO_BM25_TOP_K,
        "rerank_top_n": rerank_top_n,
        "max_context_chunks": max_context_chunks,
        "chat_model": chat_model,
        "embed_model": DEFAULT_EMBED_MODEL,
        "assets_path": assets_path or DEFAULT_ASSETS,
        "table_db_path": table_db_path,
        "table_vector_top_k": table_vector_top_k,
        "table_similarity_threshold": table_similarity_threshold,
        "max_table_contexts": max_table_contexts,
        "ticker_filter": ticker_filter,
        "fiscal_year_filter": fiscal_year_filter,
    }
    if retrieval_query is not None and _supports_retrieval_query():
        kwargs["retrieval_query"] = retrieval_query
    if _run_pipeline_supports("include_section_neighbors"):
        kwargs["include_section_neighbors"] = True
    return run_pipeline(question, **kwargs)


def _run_pipeline_supports(name: str) -> bool:
    params = inspect.signature(run_pipeline).parameters
    return name in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _supports_retrieval_query() -> bool:
    return _run_pipeline_supports("retrieval_query")


TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "sql",
        "description": (
            "Query the read-only SQLite financial database for structured numbers, "
            "ratios, rankings, segment/geographic revenue, balance-sheet metrics, "
            "and year-over-year comparisons for Apple, Microsoft, or Alphabet."
        ),
        "when_to_use": [
            "The answer needs exact revenue, income, margin, assets, liabilities, cash, debt, or EPS.",
            "The question asks for highest/lowest, growth rate, ranking, percentage of total, or cross-company comparison.",
            "The question mentions FY2023, FY2024, FY2025, segment names, or geographic regions in the database.",
            "You need numeric facts before asking the filing for narrative explanation.",
        ],
        "when_not_to_use": [
            "The question only asks what management said, disclosed, or explained in the 10-K.",
            "The data is not in the schema (stock price, headcount, Azure-only revenue, market cap, dividends).",
            "You already have the needed numbers from a prior sql observation.",
        ],
        "parameters": {
            "question": {
                "type": "string",
                "required": True,
                "description": (
                    "A focused natural-language database question. Do not pass raw SQL. "
                    "Example: 'What was Apple Services revenue in FY2024 and FY2025?'"
                ),
            }
        },
        "returns": "status, generated_sql, result rows, row_count, and error_message if any.",
    },
    {
        "name": "rag",
        "description": (
            "Search Apple, Microsoft, or Alphabet 10-K filings with vector retrieval, "
            "reranking, section-aware context expansion, and table injection to answer "
            "narrative or filing-specific questions."
        ),
        "when_to_use": [
            "The question asks what the 10-K says, explains, discloses, or discusses.",
            "You need MD&A commentary, risk factors, strategy, regulatory issues, or performance drivers.",
            "You need filing text or table values after identifying a segment/metric from sql.",
            "The user asks for reasons, outlook, competition, or management narrative.",
        ],
        "when_not_to_use": [
            "The question can be answered entirely from the SQLite database.",
            "The question is unrelated to Apple, Microsoft, Alphabet, Salesforce, or their filings.",
            "A prior rag observation already contains the needed filing evidence.",
        ],
        "parameters": {
            "question": {
                "type": "string",
                "required": True,
                "description": (
                    "A single-intention filing question about one topic/section to retrieve. "
                    "It should be scoped to one company and one fiscal year; pass that scope "
                    "through ticker and fiscal_year instead of relying on question text. "
                    "For comparisons or multi-part asks, call rag once per independent "
                    "company/year/intention, then compare the returned answers. "
                    "Example: 'What drivers did management cite for Services growth?'"
                ),
            },
            "ticker": {
                "type": "string",
                "required": False,
                "description": (
                    "Optional metadata filter: AAPL, MSFT, GOOGL, or SALESFORCE when Salesforce "
                    "PDF chunks are present in the active RAG index/workspace. "
                    "Use when company scope is known (from the user or a prior sql/rag observation). "
                    "Pass exactly one ticker per rag call. For cross-company comparisons, "
                    "call rag separately for each ticker and synthesize after retrieval."
                ),
            },
            "fiscal_year": {
                "type": "string",
                "required": False,
                "description": (
                    "Optional metadata filter: FY2024 or FY2025 (2025 is also accepted). "
                    "Pass exactly one fiscal year per rag call — this selects which filing PDF to "
                    "search, not only the metric year inside tables. Newer filings often include "
                    "2–3 years of comparative performance (e.g., FY2025 PDF may contain FY2025, "
                    "FY2024, and FY2023 columns). If a FY2024 or FY2023 ask misses in the matching "
                    "year's filing, retry with fiscal_year=FY2025 and name the target year in the "
                    "question. For multi-year comparisons, call rag separately for each filing year "
                    "and synthesize after retrieval."
                ),
            },
        },
        "returns": "answer text, reranked chunks, table contexts, applied scope filters, and pipeline latency.",
    },
    {
        "name": "send_email",
        "description": (
            "Email the final research answer to the user as plain text. "
            "Use only after you have composed the answer the user wants delivered."
        ),
        "when_to_use": [
            "The user asks to email, send, or share the results to their inbox.",
            "You have a final answer ready and the user has provided a valid email address.",
        ],
        "when_not_to_use": [
            "You do not yet have the answer content to send.",
            "The user has not given an email address — ask them in chat first, do not guess.",
            "The user only wants a shorter summary in the chat (no email).",
        ],
        "parameters": {
            "to_email": {
                "type": "string",
                "required": True,
                "description": "Recipient address the user provided in the conversation.",
            },
            "subject": {
                "type": "string",
                "required": True,
                "description": "Short subject line describing the research question.",
            },
            "body": {
                "type": "string",
                "required": True,
                "description": "Full plain-text answer to email (not raw JSON).",
            },
        },
        "returns": "ok, status, to_email, subject, or error_message if send failed.",
    },
]


TOOL_DESCRIPTIONS = {
    tool["name"]: tool["description"] for tool in TOOL_SCHEMA
}


def format_tool_catalog() -> str:
    blocks = []
    for tool in TOOL_SCHEMA:
        when_use = "\n".join(f"- {item}" for item in tool.get("when_to_use", []))
        when_not = "\n".join(f"- {item}" for item in tool.get("when_not_to_use", []))
        params = tool.get("parameters", {})
        param_lines = "\n".join(
            f"- {name} ({spec.get('type', 'string')}): {spec.get('description', '')}"
            for name, spec in params.items()
        )
        blocks.append(
            f"""### Tool: {tool['name']}
Description: {tool['description']}

Use when:
{when_use}

Do not use when:
{when_not}

Parameters:
{param_lines}

Returns: {tool.get('returns', '')}"""
        )
    return "\n\n".join(blocks)


def run_sql_tool(question: str) -> dict[str, Any]:
    """Answer a structured financial question through the read-only SQL path."""
    start = time.perf_counter()
    try:
        result = answer_sql_question(question)
        return {
            "tool": "sql",
            "input": question,
            "ok": result.get("status") in {"success", "empty_result", "cannot_answer"},
            "latency_sec": round(time.perf_counter() - start, 3),
            **result,
        }
    except Exception as exc:
        return {
            "tool": "sql",
            "input": question,
            "ok": False,
            "status": "error",
            "latency_sec": round(time.perf_counter() - start, 3),
            "error_message": str(exc),
        }


def run_rag_tool(
    question: str,
    *,
    ticker: str | list[str] | None = None,
    fiscal_year: str | list[str] | None = None,
    db_path: Path | None = None,
    assets_path: Path | None = None,
    table_db_path: Path | None = None,
    vector_top_k: int = STUDIO_VECTOR_TOP_K,
    rerank_top_n: int = STUDIO_RERANK_TOP_N,
    max_context_chunks: int = STUDIO_MAX_CONTEXT_CHUNKS,
    chat_model: str = DEFAULT_CHAT_MODEL,
    table_vector_top_k: int = STUDIO_TABLE_VECTOR_TOP_K,
    table_similarity_threshold: float = STUDIO_TABLE_SIMILARITY_THRESHOLD,
    max_table_contexts: int | None = STUDIO_MAX_TABLE_CONTEXTS,
) -> dict[str, Any]:
    """Answer a filing/PDF question through vector RAG inference."""
    start = time.perf_counter()
    ticker_filter = normalize_filter_values(_coerce_scope_filter(ticker), kind="ticker")
    available_filing_years = _available_filing_years(db_path)
    requested_filing_year = _normalize_single_filing_year(fiscal_year)
    resolved_filing_year, filing_resolution = resolve_filing_year_filter(
        question,
        requested_filing_year,
        available_filing_years=available_filing_years,
    )
    fiscal_year_filter = [resolved_filing_year] if resolved_filing_year else None
    tool_input = {
        "question": question,
        "ticker": ticker_filter,
        "fiscal_year": fiscal_year_filter,
    }
    if filing_resolution:
        tool_input["filing_year_resolved"] = filing_resolution
    if _is_multi_filter(ticker_filter) or _is_multi_filter(fiscal_year_filter):
        return {
            "tool": "rag",
            "input": tool_input,
            "ok": False,
            "status": "needs_decomposition",
            "latency_sec": round(time.perf_counter() - start, 3),
            "error_message": (
                "RAG calls must be scoped to exactly one ticker and one fiscal year. "
                "Decompose comparison or multi-part requests into separate rag calls, "
                "then synthesize the returned answers."
            ),
        }
    try:
        initial_retrieval_query = None
        if (
            filing_resolution
            and filing_resolution.get("reason")
            in {"metric_year_not_indexed_as_filing", "metric_year_older_than_filing_scope"}
            and _supports_retrieval_query()
        ):
            initial_retrieval_query = comparative_table_retrieval_query(
                question,
                extract_metric_years(question),
            )
        result = _run_rag_pipeline(
            question,
            retrieval_query=initial_retrieval_query,
            db_path=db_path,
            assets_path=assets_path,
            table_db_path=table_db_path,
            vector_top_k=vector_top_k,
            rerank_top_n=rerank_top_n,
            max_context_chunks=max_context_chunks,
            chat_model=chat_model,
            table_vector_top_k=table_vector_top_k,
            table_similarity_threshold=table_similarity_threshold,
            max_table_contexts=max_table_contexts,
            ticker_filter=ticker_filter,
            fiscal_year_filter=fiscal_year_filter,
        )
        fallback_trace: list[dict[str, Any]] = list(result.get("fallback_trace") or [])
        if filing_resolution:
            fallback_trace.insert(0, filing_resolution)
        sufficiency_check = result.get("sufficiency_check")
        if isinstance(sufficiency_check, dict) and sufficiency_check.get("sufficient") is False:
            status = "insufficient_context"
        elif fallback_trace and (not _looks_insufficient_answer(result.get("answer")) or _has_substantive_evidence(result.get("answer"))):
            status = "fallback_success"
        elif fallback_trace:
            status = "insufficient_context"
        else:
            status = "success"
        if not fallback_trace and _looks_insufficient_answer(result.get("answer")) and _supports_retrieval_query():
            retry_question = _fallback_retrieval_query(question)
            retry_result = _run_rag_pipeline(
                question,
                retrieval_query=retry_question,
                db_path=db_path,
                assets_path=assets_path,
                table_db_path=table_db_path,
                vector_top_k=vector_top_k,
                rerank_top_n=rerank_top_n,
                max_context_chunks=max_context_chunks,
                chat_model=chat_model,
                table_vector_top_k=table_vector_top_k,
                table_similarity_threshold=table_similarity_threshold,
                max_table_contexts=max_table_contexts,
                ticker_filter=ticker_filter,
                fiscal_year_filter=fiscal_year_filter,
            )
            fallback_trace.append(
                {
                    "reason": "insufficient_context_retrieval_rewrite",
                    "original_question": question,
                    "retry_question": retry_question,
                    "changed": ["retrieval_formulation"],
                }
            )
            if not _looks_insufficient_answer(retry_result.get("answer")) or _has_substantive_evidence(retry_result.get("answer")):
                result = retry_result
                sufficiency_check = result.get("sufficiency_check")
                fallback_trace.extend(result.get("fallback_trace") or [])
                status = "fallback_success"
            else:
                status = "insufficient_context"
        if (
            _looks_insufficient_answer(result.get("answer"))
            and _supports_retrieval_query()
        ):
            current_filing = fiscal_year_filter[0] if fiscal_year_filter else None
            retry_filing, bump_to = should_retry_with_newer_filing(
                result.get("answer"),
                current_filing,
                question,
                available_filing_years=available_filing_years,
            )
            if retry_filing and bump_to:
                metric_years = extract_metric_years(question)
                retry_result = _run_rag_pipeline(
                    question,
                    retrieval_query=comparative_table_retrieval_query(question, metric_years),
                    db_path=db_path,
                    assets_path=assets_path,
                    table_db_path=table_db_path,
                    vector_top_k=vector_top_k,
                    rerank_top_n=rerank_top_n,
                    max_context_chunks=max_context_chunks,
                    chat_model=chat_model,
                    table_vector_top_k=table_vector_top_k,
                    table_similarity_threshold=table_similarity_threshold,
                    max_table_contexts=max_table_contexts,
                    ticker_filter=ticker_filter,
                    fiscal_year_filter=[bump_to],
                )
                fallback_trace.append(
                    {
                        "reason": "newer_filing_after_insufficient",
                        "metric_years": metric_years,
                        "requested_filing_year": current_filing,
                        "resolved_filing_year": bump_to,
                    }
                )
                if not _looks_insufficient_answer(retry_result.get("answer")) or _has_substantive_evidence(retry_result.get("answer")):
                    result = retry_result
                    fiscal_year_filter = [bump_to]
                    tool_input["fiscal_year"] = fiscal_year_filter
                    sufficiency_check = result.get("sufficiency_check")
                    fallback_trace.extend(result.get("fallback_trace") or [])
                    status = "fallback_success"
                elif status != "fallback_success":
                    status = "insufficient_context"
        return {
            "tool": "rag",
            "input": tool_input,
            "ok": True,
            "status": status,
            "fallback_trace": fallback_trace,
            "retrieval_confidence": result.get("retrieval_confidence"),
            "sufficiency_check": sufficiency_check,
            "latency_sec": round(time.perf_counter() - start, 3),
            "answer": result.get("answer"),
            "pipeline_latency": result.get("latency", {}),
            "reranked_top": result.get("reranked_top", []),
            "expanded_context": result.get("expanded_context", []),
            "table_contexts": result.get("table_contexts", []),
            "scope_filters": result.get("settings", {}),
        }
    except Exception as exc:
        return {
            "tool": "rag",
            "input": tool_input,
            "ok": False,
            "status": "error",
            "latency_sec": round(time.perf_counter() - start, 3),
            "error_message": str(exc),
        }


def run_send_email_tool(to_email: str, subject: str, body: str) -> dict[str, Any]:
    """Send the final answer to the user's inbox."""
    start = time.perf_counter()
    result = send_answer_email(to_email, subject, body)
    result["input"] = {"to_email": to_email, "subject": subject}
    result["latency_sec"] = round(time.perf_counter() - start, 3)
    return result


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "sql": run_sql_tool,
    "rag": run_rag_tool,
    "send_email": run_send_email_tool,
}


def execute_tool(name: str, question: str, **kwargs: Any) -> dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    if name == "send_email":
        return run_send_email_tool(
            kwargs.get("to_email", question),
            kwargs.get("subject", ""),
            kwargs.get("body", ""),
        )
    return TOOL_REGISTRY[name](question, **kwargs)

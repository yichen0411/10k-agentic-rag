from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SQL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SQL_DIR.parents[1]
DB_PATH = PROJECT_ROOT / "data" / "financials.db"
LOG_PATH = SQL_DIR / "sql_log.jsonl"

ALLOWED_TABLES = {
    "income_statements",
    "balance_sheets",
    "geographic_revenue",
    "segment_revenue",
    "companies",
}
FORBIDDEN_KEYWORDS = {
    "ALTER",
    "ANALYZE",
    "ATTACH",
    "BEGIN",
    "COMMIT",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "ROLLBACK",
    "SAVEPOINT",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
}

from financials_schema import build_text_to_sql_schema

TEXT_TO_SQL_RULES = """
Rules:
- Only generate SELECT statements.
- The database is strictly read-only. Never generate SQL that modifies schema,
  data, transactions, attached databases, or SQLite settings.
- Always filter company_ticker with values listed in the schema.
- Return only the raw SQL query. No explanation, no markdown, no backticks, no comments.
- Use SQLite syntax only. Do not use non-SQLite features such as QUALIFY,
  DATE_TRUNC, ILIKE, ARRAY_AGG, or proprietary SQL dialect functions.
- Do not use window functions inside aggregate functions. If a question asks
  for "most consistent", "volatility", "growth across years", or similar
  multi-step analysis, return the raw company/year values ordered by
  company_ticker and fiscal_year so Python can compute the final comparison.
- For queries returning row-level data, add LIMIT 1000.
- For queries using GROUP BY or aggregate functions (SUM, AVG, MAX, MIN, COUNT),
  do not add LIMIT unless the question asks for top N results.
- You may calculate simple ratios and percentages directly in SQLite.
  Use ROUND(..., 4) for decimal precision.
- SQLite integer division: revenue and other money columns are BIGINT. Always cast or
  multiply by 100.0 (or 1.0) before dividing, e.g. ROUND(100.0 * numerator / NULLIF(denominator, 0), 2).
  Never write (bigint_col / bigint_col) * 100 — that truncates to 0.
- For percentages and ROA/ROE, express as percent points in SQL (multiply by 100.0) unless
  the question explicitly asks for a decimal ratio.
- For return on assets (ROA), join income_statements.net_income with
  balance_sheets.total_assets on company_ticker and fiscal_year.
- For return on equity (ROE), join income_statements.net_income with
  balance_sheets.stockholders_equity on company_ticker and fiscal_year.
- For growth rates, CAGR, consistency, or year-over-year comparisons, fetch
  the relevant years and values; Python will calculate the final metric.
- Avoid division by zero: use NULLIF(denominator, 0) in division operations.
- Azure-only and YouTube-ad-only revenue are not in segment_revenue; return CANNOT_ANSWER for those asks.
- If the question requires data not in the schema，return exactly: CANNOT_ANSWER
""".strip()


def build_text_to_sql_system_prompt() -> str:
    return "\n\n".join(
        [
            "You are a Text-to-SQL generator for a financial research system.",
            "Convert the user's natural language question into one SQLite SELECT query.",
            "Database: financials.db",
            "SQL dialect: SQLite only.",
            build_text_to_sql_schema(),
            TEXT_TO_SQL_RULES,
        ]
    )


TEXT_TO_SQL_SYSTEM_PROMPT = build_text_to_sql_system_prompt()

def _load_local_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _strip_response(text: str) -> str:
    sql = (text or "").strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.I)
        sql = re.sub(r"\s*```\s*$", "", sql)
    return sql.strip().rstrip(";")


def _call_anthropic(system_prompt: str, user_prompt: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except Exception:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_SQL_MODEL", os.environ.get("ANTHROPIC_ROUTER_MODEL", "claude-haiku-4-5-20251001")),
        max_tokens=500,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


def _call_fireworks(system_prompt: str, user_prompt: str) -> str | None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        return None
    inference_dir = Path(__file__).resolve().parents[1] / "inference"
    if str(inference_dir) not in sys.path:
        sys.path.insert(0, str(inference_dir))
    from text_vector_rag_inference import (  # noqa: E402
        DEFAULT_FIREWORKS_CHAT_MODEL,
        call_fireworks_chat,
    )

    model = (
        os.environ.get("FW_SQL_MODEL")
        or os.environ.get("FW_ROUTER_MODEL")
        or os.environ.get("FW_CHAT_MODEL")
        or DEFAULT_FIREWORKS_CHAT_MODEL
    )
    return call_fireworks_chat(
        prompt=user_prompt,
        api_key=api_key,
        model=model,
        system=system_prompt,
        max_tokens=500,
    )


def _resolve_sql_llm_provider() -> str:
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("fireworks", "anthropic"):
        return explicit
    if os.environ.get("FIREWORKS_API_KEY"):
        return "fireworks"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "fireworks"


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    _load_local_env()
    provider = _resolve_sql_llm_provider()
    if provider == "fireworks":
        text = _call_fireworks(system_prompt, user_prompt)
    else:
        text = _call_anthropic(system_prompt, user_prompt)
    if text is None:
        raise RuntimeError("No LLM API key available for Text-to-SQL.")
    return _strip_response(text)


def _make_result(
    status: str,
    sql: str | None = None,
    result: list[dict[str, Any]] | None = None,
    row_count: int = 0,
    correction_used: bool = False,
    error_message: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    out = {
        "status": status,
        "sql": sql,
        "result": result,
        "row_count": row_count,
        "correction_used": correction_used,
        "error_message": error_message,
    }
    if message is not None:
        out["message"] = message
    return out


def _fallback_result() -> dict[str, Any]:
    return {
        "status": "fallback",
        "message": "Could not reliably answer from the database. Please refer to the financial statements directly.",
        "sql": None,
        "result": None,
        "row_count": 0,
        "correction_used": False,
        "error_message": None,
    }


def _log_call(question: str, generated_sql: str | None, result: dict[str, Any], latency_ms: int) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "generated_sql": generated_sql,
        "status": result.get("status"),
        "row_count": result.get("row_count", 0),
        "correction_used": result.get("correction_used", False),
        "error_message": result.get("error_message"),
        "latency_ms": latency_ms,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _strip_sql_string_literals(sql: str) -> str:
    """Remove quoted string contents before keyword checks."""
    without_single = re.sub(r"'(?:''|[^'])*'", "''", sql)
    return re.sub(r'"(?:""|[^"])*"', '""', without_single)


def _contains_forbidden_keyword(sql: str) -> bool:
    upper = _strip_sql_string_literals(sql).upper()
    return any(re.search(rf"\b{keyword}\b", upper) for keyword in FORBIDDEN_KEYWORDS)


def _first_word(sql: str) -> str:
    match = re.match(r"\s*([A-Za-z]+)", sql or "")
    return match.group(1).upper() if match else ""


def _extract_table_names(sql: str) -> set[str]:
    names = set()
    for match in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.I):
        names.add(match.group(1))
    return names


def _validate_sql(sql: str) -> str | None:
    if not sql:
        return "SQL is empty."
    if sql.strip().upper() == "CANNOT_ANSWER":
        return None
    if _contains_forbidden_keyword(sql):
        return "SQL contains a forbidden keyword."
    if "--" in sql or "/*" in sql or "*/" in sql:
        return "SQL comments are not allowed."
    if _first_word(sql) != "SELECT":
        return "Only SELECT statements are allowed."
    stripped = sql.strip()
    if ";" in stripped.rstrip(";"):
        return "Multiple SQL statements are not allowed."

    tables = _extract_table_names(sql)
    unknown = sorted(tables - ALLOWED_TABLES)
    if unknown:
        return f"Unknown table(s): {', '.join(unknown)}."
    if not tables:
        return "No table name found in SQL."
    return None


def _connect_readonly() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_readonly_authorizer)
    return conn


def _readonly_authorizer(action_code: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger: str | None) -> int:
    del arg1, arg2, db_name, trigger
    allowed_actions = {
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_SELECT,
    }
    if action_code in allowed_actions:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _execute_sql(sql: str) -> tuple[list[dict[str, Any]], int]:
    with _connect_readonly() as conn:
        rows = conn.execute(sql).fetchall()
    result = [dict(row) for row in rows]
    return result, len(result)


def _generate_sql(question: str) -> str:
    return _call_llm(TEXT_TO_SQL_SYSTEM_PROMPT, question)


def _correct_sql(question: str, failed_sql: str, error_message: str) -> str:
    user_prompt = (
        f"Original question:\n{question}\n\n"
        f"Failed SQL:\n{failed_sql}\n\n"
        f"Execution error:\n{error_message}\n\n"
        "Fix the SQLite query based on the error and the schema in the system prompt. "
        "Return only the corrected SQL. If it cannot be answered from the schema, return CANNOT_ANSWER."
    )
    return _call_llm(TEXT_TO_SQL_SYSTEM_PROMPT, user_prompt)


def answer_sql_question(question: str) -> dict[str, Any]:
    start = time.perf_counter()
    generated_sql: str | None = None
    final: dict[str, Any]
    max_attempts = 3

    try:
        generated_sql = _generate_sql(question)
        if generated_sql.upper() == "CANNOT_ANSWER":
            final = _make_result("cannot_answer", sql=None, result=None)
            return final

        current_sql = generated_sql
        correction_used = False
        last_error_message: str | None = None

        for attempt in range(max_attempts):
            validation_error = _validate_sql(current_sql)
            if validation_error:
                last_error_message = validation_error
                if attempt >= max_attempts - 1:
                    final = _make_result(
                        "error",
                        sql=current_sql,
                        result=None,
                        correction_used=correction_used,
                        error_message=validation_error,
                    )
                    return final
                corrected_sql = _correct_sql(question, current_sql, validation_error)
                correction_used = True
                if corrected_sql.upper() == "CANNOT_ANSWER":
                    final = _fallback_result()
                    final["error_message"] = validation_error
                    return final
                current_sql = corrected_sql
                continue

            try:
                result, row_count = _execute_sql(current_sql)
                if row_count == 0:
                    final = _make_result(
                        "empty_result",
                        sql=current_sql,
                        result=[],
                        row_count=0,
                        correction_used=correction_used,
                    )
                    return final
                if row_count > 1000:
                    final = _make_result(
                        "error",
                        sql=current_sql,
                        result=None,
                        row_count=row_count,
                        correction_used=correction_used,
                        error_message="Result has more than 1000 rows.",
                    )
                    return final
                final = _make_result(
                    "success",
                    sql=current_sql,
                    result=result,
                    row_count=row_count,
                    correction_used=correction_used,
                )
                return final
            except Exception as exc:
                last_error_message = str(exc)
                if attempt >= max_attempts - 1:
                    final = _fallback_result()
                    final["sql"] = current_sql
                    final["correction_used"] = correction_used
                    final["error_message"] = last_error_message
                    return final

                corrected_sql = _correct_sql(question, current_sql, last_error_message)
                correction_used = True
                if corrected_sql.upper() == "CANNOT_ANSWER":
                    final = _fallback_result()
                    final["error_message"] = last_error_message
                    return final
                current_sql = corrected_sql

        final = _fallback_result()
        final["sql"] = current_sql
        final["correction_used"] = correction_used
        final["error_message"] = last_error_message
        return final

    except Exception as exc:
        final = _fallback_result()
        final["error_message"] = str(exc)
        return final

    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log_call(question, generated_sql, locals().get("final", _fallback_result()), latency_ms)


def main() -> None:
    questions = [
        "What was Apple's total revenue in FY2025?",
        "Which company had the highest net income in FY2025?",
        "What was Apple's Greater China revenue in FY2024 and FY2025?",
        "Compare gross margins across all three companies in FY2025",
        "What was Microsoft's Azure revenue in FY2025?",
        "What is Apple's stock price today?",
        "What were Apple's iPhone and Services revenues in FY2025?",
        "Which company had the fastest revenue growth between FY2024 and FY2025?",
    ]
    for question in questions:
        print("=" * 80)
        print(question)
        print(json.dumps(answer_sql_question(question), indent=2))


if __name__ == "__main__":
    main()

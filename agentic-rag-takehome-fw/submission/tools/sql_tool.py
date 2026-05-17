"""
SQL tool: multi-step pipeline that converts a natural language question about
AAPL / MSFT / GOOGL financials into a SQL query, executes it, and returns a
synthesized prose answer.

Pipeline:
  classify_question → inspect_db → decompose_question → generate_sql
  → execute_with_retry → _synthesize_answer
"""

import sqlite3
import re
import json
import logging
from typing import Optional

from openai import OpenAI
from config import DB_PATH, FIREWORKS_API_KEY, FIREWORKS_BASE_URL, CHAT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema description (injected into every SQL-generation prompt)
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = """
Database: financials.db
Companies: AAPL (Apple Inc., FY ends September), MSFT (Microsoft, FY ends June),
           GOOGL (Alphabet Inc., FY ends December)
Fiscal years available: 2023, 2024, 2025
All monetary values are in USD (e.g. 391035000000 ≈ $391B).

Table routing:
- P&L (revenue, net income, gross profit, EPS, operating income): income_statements
- Balance sheet (assets, liabilities, equity, cash, debt, current ratio): balance_sheets
- Business segment revenue (iPhone, Mac, Services, Cloud, Azure, etc.): segment_revenue
- Geographic / regional revenue: geographic_revenue
  NOTE: SUM(all regions) = total revenue, so no join to income_statements needed.

Tables:

1. companies
   ticker TEXT PK | name TEXT | cik TEXT | sic TEXT | sector TEXT | fiscal_year_end TEXT

2. income_statements
   company_ticker TEXT | fiscal_year INT | period_start TEXT | period_end TEXT
   period_type TEXT ('FY', 'Q1'–'Q4') | revenue BIGINT | cost_of_revenue BIGINT
   gross_profit BIGINT | research_and_development BIGINT
   total_operating_expenses BIGINT | operating_income BIGINT | net_income BIGINT
   eps_basic REAL | eps_diluted REAL

3. balance_sheets
   company_ticker TEXT | fiscal_year INT | period_end TEXT | period_type TEXT
   total_assets BIGINT | total_liabilities BIGINT | stockholders_equity BIGINT
   cash_and_equivalents BIGINT | total_debt BIGINT | short_term_debt BIGINT
   accounts_receivable BIGINT | total_current_assets BIGINT
   total_current_liabilities BIGINT

4. segment_revenue
   company_ticker TEXT | fiscal_year INT | period_end TEXT | period_type TEXT
   segment_name TEXT | revenue BIGINT
   AAPL segments: iPhone, Mac, iPad, Services, Wearables Home and Accessories
   MSFT segments: Productivity and Business Processes, Intelligent Cloud, More Personal Computing
   GOOGL segments: Google Services, Google Cloud, Other Bets, Hedging gains (losses)

5. geographic_revenue
   company_ticker TEXT | fiscal_year INT | period_end TEXT | period_type TEXT
   region TEXT | revenue BIGINT
   AAPL regions: Americas, Europe, Greater China, Japan, Rest of Asia Pacific
   MSFT regions: United States, Other countries
   GOOGL regions: United States, Other Americas, EMEA, APAC, Hedging gains (losses)
"""

_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|TRUNCATE|PRAGMA|ATTACH)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client() -> OpenAI:
    return OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)


def _get_conn():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?i)^```sql\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_sql(text: str) -> str:
    """
    Robustly extract the first SELECT/WITH statement from LLM output.
    Handles leading explanation, markdown fences, and inline preambles.
    """
    # 1. SQL code fence
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if re.match(r"(?i)^(SELECT|WITH)\b", inner):
            return inner

    # 2. Scan lines for the first SELECT / WITH
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"(?i)^\s*(SELECT|WITH)\b", line):
            return "\n".join(lines[i:]).strip()

    # 3. Inline SELECT anywhere
    m = re.search(r"(?i)\b(SELECT\b[\s\S]*)", text)
    if m:
        return m.group(1).strip()

    return _strip_fences(text)


def _llm(
    prompt: str,
    client: OpenAI,
    max_tokens: int = 800,
    system: str = "",
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Step 0: run raw SQL (kept for direct use and retries)
# ---------------------------------------------------------------------------


def run_sql(sql: str) -> str:
    """Execute a read-only SELECT. Returns formatted table string or error string."""
    sql = sql.strip().rstrip(";").split(";")[0].strip()
    if _BLOCKED.search(sql):
        return "ERROR: Only SELECT statements are allowed."
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return "ERROR: Query must start with SELECT or WITH."
    try:
        conn = _get_conn()
        cur = conn.execute(sql)
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return "No results found."
        cols = [d[0] for d in cur.description]
        conn.close()
        header = " | ".join(cols)
        sep = "-" * len(header)
        lines = [header, sep] + [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"SQL ERROR: {e}"


# ---------------------------------------------------------------------------
# Step 1: classify_question
# ---------------------------------------------------------------------------


_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "set_complexity",
        "description": "Record whether the question needs one or multiple SQL queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "multi_part"],
                    "description": (
                        "simple: one SQL query is sufficient. "
                        "multi_part: needs multiple SQL queries (multiple tables, "
                        "entities, time periods, or a follow-up breakdown)."
                    ),
                }
            },
            "required": ["complexity"],
        },
    },
}


def classify_question(question: str, client: OpenAI) -> str:
    """
    Classify the question as 'simple' (one SQL) or 'multi_part' (needs multiple SQLs).
    Uses function-calling so the model's internal reasoning doesn't leak into output.
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Question: "{question}"\n\n'
                    "Does answering this require multiple SQL queries against different "
                    "tables or different groups of data that must be combined? "
                    "Call set_complexity with 'simple' or 'multi_part'."
                ),
            }
        ],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "set_complexity"}},
        max_tokens=100,
        temperature=0,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        return json.loads(msg.tool_calls[0].function.arguments).get("complexity", "simple")
    return "simple"


# ---------------------------------------------------------------------------
# Step 2: inspect_db
# ---------------------------------------------------------------------------


def inspect_db(question: str) -> str:
    """
    Query the DB for actual entity values referenced in the question.
    Returns a multi-line context string used to ground SQL generation.
    """
    conn = _get_conn()
    lines = []

    cur = conn.execute(
        "SELECT ticker, name, fiscal_year_end FROM companies ORDER BY ticker"
    )
    lines.append(
        "Companies: "
        + "; ".join(f"{r[0]} = {r[1]} (FY ends {r[2]})" for r in cur.fetchall())
    )

    cur = conn.execute(
        "SELECT DISTINCT company_ticker, fiscal_year "
        "FROM income_statements ORDER BY company_ticker, fiscal_year"
    )
    by_ticker: dict = {}
    for r in cur.fetchall():
        by_ticker.setdefault(r[0], []).append(str(r[1]))
    lines.append(
        "Fiscal years available: "
        + "; ".join(f"{t}: {', '.join(yrs)}" for t, yrs in by_ticker.items())
    )

    cur = conn.execute(
        "SELECT DISTINCT company_ticker, segment_name "
        "FROM segment_revenue ORDER BY company_ticker, segment_name"
    )
    by_ticker: dict = {}
    for r in cur.fetchall():
        by_ticker.setdefault(r[0], []).append(r[1])
    for ticker, names in by_ticker.items():
        lines.append(f"Segments ({ticker}): {', '.join(names)}")

    cur = conn.execute(
        "SELECT DISTINCT company_ticker, region "
        "FROM geographic_revenue ORDER BY company_ticker, region"
    )
    by_ticker = {}
    for r in cur.fetchall():
        by_ticker.setdefault(r[0], []).append(r[1])
    for ticker, names in by_ticker.items():
        lines.append(f"Geographic regions ({ticker}): {', '.join(names)}")

    conn.close()
    context = "\n".join(lines)
    logger.debug(f"[inspect_db]\n{context}")
    return context


# ---------------------------------------------------------------------------
# Step 3: decompose_question (multi_part / comparative only)
# ---------------------------------------------------------------------------


def decompose_question(question: str, client: OpenAI) -> list:
    """
    Break a complex question into atomic sub-questions, each answerable by
    exactly one SQL query. Each sub-question must be self-contained and
    cover every distinct piece of data the full question needs.
    Returns a list of sub-question strings.
    """
    text = _llm(
        f"""You have a financial question that requires multiple SQL queries to answer fully.
Identify every distinct piece of data needed to answer the question completely,
then write one self-contained sub-question per data fetch.

Rules:
- Each sub-question must be answerable by exactly one SQL query
- Together the sub-questions must cover ALL data the original question asks for
- Sub-questions must be independent (no sub-question depends on results of another)
- Be exhaustive: if the question asks for a breakdown, include a sub-question for it

Return a JSON array of strings ONLY. No explanation.

Question: {question}""",
        client,
        max_tokens=600,
    )
    try:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            sub_qs = json.loads(m.group())
            if sub_qs:
                logger.debug(f"[decompose] {sub_qs}")
                return sub_qs
    except (json.JSONDecodeError, TypeError):
        pass
    return [question]


# ---------------------------------------------------------------------------
# Step 4: generate_sql
# ---------------------------------------------------------------------------


_SQL_RULES = """Rules:
- Name all columns explicitly (no SELECT *)
- Cast to REAL before any division: CAST(a AS REAL) / b
- Return raw values; do NOT compute growth rates or percentages inside SQL
- Always filter period_type = 'FY' for annual data (unless a quarter is asked for)
- Use exact string values from DB context for segment_name, region, company_ticker
- Add ORDER BY / LIMIT only when the question explicitly asks for ranking or top-N
- Include ALL tables needed to fully answer the question; provide one query per table"""

# Tool accepts a list of SQL queries so the model can self-select which tables it needs
_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": (
            "Execute one or more SQLite SELECT statements against the financials database. "
            "Provide every query needed to fully answer the question — one per table or data focus."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "description": (
                        "List of SQLite SELECT statements. Include all queries required "
                        "to fetch every piece of data the question asks for."
                    ),
                    "items": {"type": "string"},
                }
            },
            "required": ["queries"],
        },
    },
}


def generate_sqls(question: str, db_context: str, client: OpenAI) -> list:
    """
    Ask the model to generate ALL SQL queries needed to fully answer the question.
    Uses function-calling so output is structured, not free text.
    Returns a list of SQL strings.
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{SCHEMA_DESCRIPTION}\n\n"
                    f"Actual DB values:\n{db_context}\n\n"
                    f"{_SQL_RULES}\n\n"
                    f"Question: {question}\n\n"
                    "Call execute_sql with all SELECT statements needed to fully answer this question."
                ),
            }
        ],
        tools=[_EXEC_TOOL],
        tool_choice={"type": "function", "function": {"name": "execute_sql"}},
        max_tokens=800,
        temperature=0,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        queries = args.get("queries", [])
        if isinstance(queries, list) and queries:
            return queries
    # Fallback: extract whatever SQL is in the text
    sql = _extract_sql(msg.content or "")
    return [sql] if sql else []


# ---------------------------------------------------------------------------
# Step 5: execute_with_retry
# ---------------------------------------------------------------------------


def execute_with_retry(
    sql: str, sub_question: str, db_context: str, client: OpenAI
) -> tuple:
    """
    Execute SQL; on error or empty/null result, retry once with a targeted fix.
    Returns (result_string, final_sql_used).
    """

    def _is_error(r: str) -> bool:
        return r.startswith(("SQL ERROR:", "ERROR:"))

    def _is_empty(r: str) -> bool:
        return r == "No results found."

    def _all_nulls(r: str) -> bool:
        data_lines = r.split("\n")[2:]  # skip header + sep
        return bool(data_lines) and all(
            re.match(r"^(None\s*\|\s*)*None\s*$", ln) for ln in data_lines
        )

    for attempt in range(2):
        result = run_sql(sql)
        logger.debug(f"[execute attempt {attempt + 1}] sql={sql!r}\nresult={result[:300]}")

        if not _is_error(result) and not _is_empty(result) and not _all_nulls(result):
            return result, sql

        if attempt == 1:
            break  # exhausted retries

        # Craft targeted fix hint
        if _is_error(result):
            hint = (
                f"The SQL failed with: {result}\n"
                "Fix the syntax or schema reference error."
            )
        elif _is_empty(result):
            hint = (
                "The SQL returned no rows. The WHERE conditions likely use wrong values. "
                "Cross-check the actual DB values in the context and correct the filters."
            )
        else:
            hint = (
                "The SQL returned only NULL values. "
                "Select different columns or fix the aggregation."
            )

        retry_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{hint}\n\n"
                        f"Original question: {sub_question}\n\n"
                        f"Failed SQL:\n{sql}\n\n"
                        f"{SCHEMA_DESCRIPTION}\n\n"
                        f"Actual DB values:\n{db_context}\n\n"
                        f"{_SQL_RULES}\n\n"
                        "Call execute_sql with the corrected SELECT statement."
                    ),
                }
            ],
            tools=[_EXEC_TOOL],
            tool_choice={"type": "function", "function": {"name": "execute_sql"}},
            max_tokens=600,
            temperature=0,
        )
        retry_msg = retry_resp.choices[0].message
        if retry_msg.tool_calls:
            args = json.loads(retry_msg.tool_calls[0].function.arguments)
            queries = args.get("queries", [])
            sql = queries[0] if queries else _extract_sql(retry_msg.content or "")
        else:
            sql = _extract_sql(retry_msg.content or "")

    return result, sql


# ---------------------------------------------------------------------------
# Final synthesis
# ---------------------------------------------------------------------------


def _synthesize_answer(question: str, sub_results: list, client: OpenAI) -> str:
    """Synthesize a prose answer from one or more SQL result sets."""
    parts = "\n\n".join(
        f"Sub-question: {r['sub_q']}\nSQL used: {r['sql']}\nResult:\n{r['result']}"
        for r in sub_results
    )
    return _llm(
        f"""You are a financial analyst. Using the SQL results below, answer the question.
Compute any required growth rates, percentages, or comparisons from the raw numbers.
Cite which table(s) and fiscal year(s) the data came from.

Original question: {question}

Query results:
{parts}

Give a clear, concise answer with exact numbers.""",
        client,
        max_tokens=1500,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def answer_sql_question(question: str, client: Optional[OpenAI] = None) -> str:
    """
    Pipeline:
      1. inspect_db     — pull actual DB values to ground SQL generation
      2. generate_sqls  — model decides which tables/queries are needed (function-calling)
      3. execute_with_retry — run each SQL, retry once on error/empty
      4. _synthesize_answer — combine all results into a prose answer
    """
    if client is None:
        client = _make_client()

    # 1. Inspect DB for actual values (always includes segment + geo metadata)
    db_context = inspect_db(question)

    # 2. Let the model decide which SQL queries are needed to fully answer the question
    sqls = generate_sqls(question, db_context, client)
    logger.debug(f"[generate_sqls] {sqls}")

    # 3. Execute each SQL (with one retry on failure)
    sub_results = []
    seen_results: set = set()
    for sql in sqls:
        result, final_sql = execute_with_retry(sql, question, db_context, client)
        if result not in seen_results:
            seen_results.add(result)
            sub_results.append({"sub_q": sql, "sql": final_sql, "result": result})

    # 4. Synthesize
    return _synthesize_answer(question, sub_results, client)


# ---------------------------------------------------------------------------
# Tool schema for the agent (takes a natural-language question, not raw SQL)
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_sql",
        "description": (
            "Answer a natural language question about financial data by querying "
            "the financials database. Use this for any question about revenue, "
            "net income, gross profit, EPS, operating income, balance-sheet metrics, "
            "segment revenue, or geographic revenue for AAPL, MSFT, or GOOGL.\n\n"
            "The tool internally classifies the question, inspects the database for "
            "actual values, decomposes complex questions, generates SQL, and retries "
            "on errors before returning a synthesized answer.\n\n"
            + SCHEMA_DESCRIPTION
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Natural language question about AAPL, MSFT, or GOOGL "
                        "financial data (FY2023–2025)."
                    ),
                }
            },
            "required": ["question"],
        },
    },
}
"""Structured SQLite schema text for Text-to-SQL and agent routing."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "financials.db"

COMPANY_TICKERS = ("AAPL", "MSFT", "GOOGL")
FISCAL_YEARS = (2023, 2024, 2025)
PERIOD_TYPES = ("FY",)


def _quote_values(values: Iterable[str]) -> str:
    return " | ".join(f"'{v}'" if not v[0].isdigit() else str(v) for v in values)


def _load_enum_maps(db_path: Path = DB_PATH) -> dict[str, dict[str, tuple[str, ...]]]:
    if not db_path.is_file():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        segment: dict[str, tuple[str, ...]] = {}
        for ticker, name in conn.execute(
            "SELECT company_ticker, segment_name FROM segment_revenue "
            "GROUP BY company_ticker, segment_name ORDER BY company_ticker, segment_name"
        ):
            segment.setdefault(ticker, []).append(name)
        region: dict[str, tuple[str, ...]] = {}
        for ticker, name in conn.execute(
            "SELECT company_ticker, region FROM geographic_revenue "
            "GROUP BY company_ticker, region ORDER BY company_ticker, region"
        ):
            region.setdefault(ticker, []).append(name)
        return {
            "segment_name": {k: tuple(v) for k, v in segment.items()},
            "region": {k: tuple(v) for k, v in region.items()},
        }
    finally:
        conn.close()


@lru_cache(maxsize=1)
def enum_maps() -> dict[str, dict[str, tuple[str, ...]]]:
    return _load_enum_maps()


def _conditional_value_lines(field: str, by_ticker: dict[str, tuple[str, ...]]) -> list[str]:
    lines = [f"- {field} TEXT"]
    for ticker in COMPANY_TICKERS:
        values = by_ticker.get(ticker, ())
        if values:
            lines.append(f"  when company_ticker = '{ticker}': {_quote_values(values)}")
    return lines


def _schema_tables(*, sql_detail: bool, enums: dict[str, dict[str, tuple[str, ...]]]) -> str:
    tickers = _quote_values(COMPANY_TICKERS)
    years = _quote_values(str(y) for y in FISCAL_YEARS)
    period = _quote_values(PERIOD_TYPES)
    region_lines = _conditional_value_lines("region", enums.get("region", {}))
    segment_lines = _conditional_value_lines("segment_name", enums.get("segment_name", {}))

    if sql_detail:
        income_cols = f"""- id INTEGER
- company_ticker TEXT ({tickers})
- fiscal_year INTEGER ({years})
- period_start TEXT
- period_end TEXT
- period_type TEXT ({period})
- revenue BIGINT
- cost_of_revenue BIGINT
- gross_profit BIGINT
- research_and_development BIGINT
- total_operating_expenses BIGINT
- operating_income BIGINT
- net_income BIGINT
- eps_basic REAL
- eps_diluted REAL"""
        balance_cols = f"""- id INTEGER
- company_ticker TEXT ({tickers})
- fiscal_year INTEGER ({years})
- period_end TEXT
- period_type TEXT ({period})
- total_assets BIGINT
- total_liabilities BIGINT
- stockholders_equity BIGINT
- cash_and_equivalents BIGINT
- total_debt BIGINT
- short_term_debt BIGINT
- accounts_receivable BIGINT
- total_current_assets BIGINT
- total_current_liabilities BIGINT"""
        geo_prefix = f"""- id INTEGER
- company_ticker TEXT ({tickers})
- fiscal_year INTEGER ({years})
- period_end TEXT
- period_type TEXT ({period})"""
        seg_prefix = geo_prefix
        revenue_col = "- revenue BIGINT"
        companies_cols = f"""- ticker TEXT ({tickers})
- name TEXT
- cik TEXT
- sic TEXT
- sector TEXT
- fiscal_year_end INTEGER"""
    else:
        income_cols = f"""- company_ticker TEXT ({tickers})
- fiscal_year INTEGER ({years})
- period_type TEXT ({period})
- revenue, cost_of_revenue, gross_profit
- research_and_development, total_operating_expenses, operating_income, net_income
- eps_basic, eps_diluted"""
        balance_cols = f"""- company_ticker TEXT ({tickers})
- fiscal_year INTEGER ({years})
- period_type TEXT ({period})
- total_assets, total_liabilities, stockholders_equity
- cash_and_equivalents, total_debt, short_term_debt
- accounts_receivable, total_current_assets, total_current_liabilities"""
        geo_prefix = f"""- company_ticker TEXT ({tickers})
- fiscal_year INTEGER ({years})
- period_type TEXT ({period})"""
        seg_prefix = geo_prefix
        revenue_col = "- revenue"
        companies_cols = f"- ticker TEXT ({tickers}), name, sector, fiscal_year_end"

    return f"""
Table: income_statements
Columns:
{income_cols}

Table: balance_sheets
Columns:
{balance_cols}

Table: geographic_revenue
Columns:
{geo_prefix}
{chr(10).join(region_lines)}
{revenue_col}

Table: segment_revenue
Columns:
{seg_prefix}
{chr(10).join(segment_lines)}
{revenue_col}

Table: companies
Columns:
{companies_cols}
""".strip()


def build_text_to_sql_schema(db_path: Path = DB_PATH) -> str:
    enums = enum_maps() if db_path.is_file() else {}
    tables = _schema_tables(sql_detail=True, enums=enums)
    return f"""
Schema:

{tables}

Coverage:
- Companies: AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet)
- fiscal_year values: {_quote_values(str(y) for y in FISCAL_YEARS)} — year the fiscal period ends
- period_type values: {_quote_values(PERIOD_TYPES)} only
- Monetary columns are USD BIGINT unless noted REAL
""".strip()


def build_agent_db_schema(db_path: Path = DB_PATH) -> str:
    enums = enum_maps() if db_path.is_file() else {}
    tables = _schema_tables(sql_detail=False, enums=enums)
    return f"""
Database: financials.db
SQL dialect: SQLite only (read-only).

{tables}

Coverage:
- Companies: Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL)
- Years: FY2023, FY2024, FY2025
- All monetary values are USD
- geographic_revenue holds revenue only — not regional operating expenses
- Azure-only or YouTube-ad-only revenue is NOT in segment_revenue
- Salesforce is NOT covered by this SQLite database. Do not use sql for Salesforce/CRM questions.
""".strip()

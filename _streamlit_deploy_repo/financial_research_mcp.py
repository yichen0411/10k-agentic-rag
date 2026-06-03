"""MCP server exposing the financial research agent as callable tools.

Run with:
    python financial_research_mcp.py

The server speaks MCP over stdio, so clients such as Cursor or Claude Desktop can
start it as a local command and call the tools below.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "main" / "agent"

for path in (ROOT, AGENT_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

load_dotenv(ROOT / ".env")

from chunk_studio.agent_bridge import global_index_status, run_trace_global  # noqa: E402
from tools import run_rag_tool, run_sql_tool  # noqa: E402


mcp = FastMCP("financial-research-agent")


def _trim_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep MCP responses readable while preserving enough trace detail."""
    return {
        "answer": payload.get("answer"),
        "session_id": payload.get("session_id"),
        "file_label": payload.get("file_label"),
        "steps": payload.get("steps", []),
        "langsmith": payload.get("langsmith", {}),
        "rag_scope": payload.get("rag_scope"),
    }


@mcp.tool()
def list_available_filings() -> dict[str, Any]:
    """List filings and index health for the global 10-K RAG corpus."""
    status = global_index_status()
    return {
        "ready": status.get("ready"),
        "filings": status.get("filings", []),
        "filing_count": status.get("filing_count", 0),
        "text_chunks": (status.get("text") or {}).get("row_count", 0),
        "table_chunks": (status.get("table") or {}).get("row_count", 0),
        "text_db": status.get("text_db"),
        "table_db": status.get("table_db"),
        "assets_exists": status.get("assets_exists"),
    }


@mcp.tool()
def ask_financial_db(question: str) -> dict[str, Any]:
    """Answer a structured financial question using the local SQLite database."""
    return run_sql_tool(question)


@mcp.tool()
def ask_10k_rag(
    question: str,
    ticker: str | None = None,
    fiscal_year: str | None = None,
) -> dict[str, Any]:
    """Answer a filing-specific question using 10-K vector retrieval and reranking."""
    return run_rag_tool(question, ticker=ticker, fiscal_year=fiscal_year)


@mcp.tool()
def run_financial_research_agent(
    query: str,
    max_steps: int = 6,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the full SQL + RAG financial research agent and return answer plus trace."""
    payload = run_trace_global(query=query, max_steps=max_steps, session_id=session_id)
    return _trim_agent_payload(payload)


if __name__ == "__main__":
    mcp.run()

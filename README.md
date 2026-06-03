# 10-K Agentic RAG (0525_redo + Chunk Studio)

Local stack for Apple / Microsoft / Alphabet: **Chunk Studio** (PDF chunking + UI) and **LangChain agent** (SQL + 10-K RAG + email).

## Layout

| Path | Role |
|------|------|
| `0525_redo/agent/` | LangChain agent, tools, prompts, **session memory** |
| `0525_redo/chunking/` | Sectioning, assets, RAG chunk build (used by Chunk Studio) |
| `0525_redo/inference/` | Vector RAG answer pipeline |
| `0525_redo/sql/` | Text-to-SQL over `data/financials.db` |
| `chunk_studio/` | Web UI: chunk browser + `/agent` Q&A |
| `data/` | SQLite DB, PDFs, chunk_studio workspaces |

## Setup

```bash
cd agentic-rag-takehome-fw
python3 -m venv .venv
source .venv/bin/activate
pip install -r starter/requirements.txt
cp .env.example .env
# Set ANTHROPIC_API_KEY (required). Optional SMTP_* for send_email.
python scripts/prepare_data.py --data-dir data
```

## Run Chunk Studio + Agent

```bash
# Use the same Python that has project deps (e.g. conda); plain `uvicorn` may hit the wrong interpreter.
python -m uvicorn chunk_studio.server:app --host 127.0.0.1 --port 8010
```

- http://127.0.0.1:8010/ — chunk browser (process PDFs with **Build embeddings**)
- http://127.0.0.1:8010/agent — LangChain agent (SQL + workspace RAG + optional email)

## CLI (no UI)

```bash
python 0525_redo/agent/agent.py "Which segment grew closest to MSFT revenue growth?"
python 0525_redo/agent/agent.py "..." --json --max-steps 6
```

## MCP Server

Expose the same financial research capabilities to MCP clients such as Cursor or
Claude Desktop:

```bash
/usr/local/bin/python3.10 -m venv .venv-mcp
.venv-mcp/bin/python -m pip install -r starter/requirements.txt
.venv-mcp/bin/python financial_research_mcp.py
```

Example local MCP config:

```json
{
  "mcpServers": {
    "financial-research-agent": {
      "command": "/Users/xinyichen/Downloads/fireworks_takehome/agentic-rag-takehome-fw/.venv-mcp/bin/python",
      "args": [
        "/Users/xinyichen/Downloads/fireworks_takehome/agentic-rag-takehome-fw/financial_research_mcp.py"
      ]
    }
  }
}
```

Tools exposed:

- `list_available_filings` — show indexed filings and RAG index health.
- `ask_financial_db` — query the local financial SQLite database.
- `ask_10k_rag` — retrieve 10-K filing evidence with optional ticker/year filters.
- `run_financial_research_agent` — run the full SQL + RAG agent and return trace steps.

## Docs

- `0525_redo/agent/README.md` — routing, tools, memory, SMTP
- `0525_redo/CHUNKING_AND_INFERENCE_DESIGN.md` — chunking + RAG pipeline
- `chunk_studio/DESIGN.md` — Chunk Studio + agent API
- `0525_redo/agent/MEMORY_DESIGN.zh.md` — session memory architecture

## Tests

```bash
python3 0525_redo/agent/test_memory_multiturn.py
```

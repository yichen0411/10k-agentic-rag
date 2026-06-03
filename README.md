# 10-K Agentic RAG (main + Chunk Studio)

Local stack for Apple / Microsoft / Alphabet: **Chunk Studio** (PDF chunking + UI) and **LangChain agent** (SQL + 10-K RAG + email).

## Layout

| Path | Role |
|------|------|
| `main/agent/` | LangChain agent, tools, prompts, **session memory** |
| `main/chunking/` | Sectioning, assets, RAG chunk build (used by Chunk Studio) |
| `main/inference/` | Vector RAG answer pipeline |
| `main/sql/` | Text-to-SQL over `data/financials.db` |
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
python main/agent/agent.py "Which segment grew closest to MSFT revenue growth?"
python main/agent/agent.py "..." --json --max-steps 6
```

## Docs

- `main/agent/README.md` — routing, tools, memory, SMTP
- `main/CHUNKING_AND_INFERENCE_DESIGN.md` — chunking + RAG pipeline
- `chunk_studio/DESIGN.md` — Chunk Studio + agent API
- `main/agent/MEMORY_DESIGN.zh.md` — session memory architecture

## Tests

```bash
python3 main/agent/test_memory_multiturn.py
```

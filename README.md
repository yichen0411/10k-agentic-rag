# 10-K Agentic RAG

Local stack for **Chunk Studio** (PDF chunking UI) and a **LangChain agent** (SQL + 10-K RAG + optional email) over Apple, Microsoft, and Alphabet filings.

## Architecture

```mermaid
graph TD
  subgraph offline [Offline indexing]
    pdf[10-K PDF] --> sec[TOC-guided sectioning]
    sec --> chk[Text chunks and table assets]
    chk --> tv[Text vector DB]
    chk --> vlm[Optional VLM table parse]
    vlm --> sv[Table summary vector DB]
  end

  subgraph runtime [Runtime]
    user[User question] --> agent[LangChain tool-calling agent]
    agent --> sql[SQL tool]
    agent --> rag[RAG tool]
    agent --> email[send_email optional]
    sql --> db[financials.db]
    rag --> tv
    rag --> sv
    rag --> ast[assets.json markdown tables]
    sql --> obs[Tool observations]
    rag --> obs
    obs --> agent
    agent --> ans[Grounded answer]
  end
```

### Offline chunking pipeline

```mermaid
graph TD
  a[10-K PDF] --> b[Read text and layout]
  b --> c[Parse visible TOC]
  c --> d[Menu-guided Item matching]
  d --> e[Detect subsections]
  d --> f[Extract tables and images]
  e --> g[Build text RAG chunks]
  f --> g
  g --> h[Metadata header_path table_refs neighbors]
  h --> i[Text vector index]
  f --> j[VLM table parse]
  j --> k[Table summary index]
```

### Dual-path RAG inference

```mermaid
graph TD
  q[Question ticker fiscal year] --> f[Metadata filter]
  f --> e[Embed query once]
  e --> vt[Text vector search]
  e --> bm[BM25 text search]
  e --> ts[Table summary vector search]
  vt --> m[Merge text hits]
  bm --> m
  m --> r[Cross-encoder rerank]
  r --> x[Context expansion]
  ts --> th[Similarity threshold]
  th --> tc[Table hits]
  x --> ctx[Assemble context]
  tc --> ctx
  ctx --> llm[Answer model]
  llm --> out[Answer and citations]
```

## Layout

| Path | Role |
|------|------|
| `main/agent/` | LangChain agent, tools, prompts, session memory |
| `main/chunking/` | Sectioning, assets, RAG chunk build |
| `main/inference/` | Vector RAG answer pipeline |
| `main/sql/` | Text-to-SQL over `data/financials.db` |
| `chunk_studio/` | Web UI: chunk browser + `/agent` Q&A |
| `data/` | SQLite DB, PDFs, vector indexes |
| `scripts/` | Data download and DB preparation |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r starter/requirements.txt
cp .env.example .env
# Set ANTHROPIC_API_KEY (required). Optional SMTP_* for send_email.
python scripts/prepare_data.py --data-dir data
```

## Run Chunk Studio + Agent

```bash
python -m uvicorn chunk_studio.server:app --host 127.0.0.1 --port 8010
```

- http://127.0.0.1:8010/ — chunk browser (process PDFs with **Build embeddings**)
- http://127.0.0.1:8010/agent — LangChain agent (SQL + workspace RAG + optional email)

## CLI (no UI)

```bash
python main/agent/agent.py "Which segment grew closest to MSFT revenue growth?"
python main/agent/agent.py "..." --json --max-steps 6
```

## Tests

```bash
python3 main/agent/test_memory_multiturn.py
python3 main/chunking/test_chunking_heuristics.py
python3 main/agent/test_filing_scope.py
```

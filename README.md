# 10-K Agentic RAG

Local stack for **Chunk Studio** (PDF chunking UI) and a **LangChain agent** (SQL + 10-K RAG + optional email) over Apple, Microsoft, and Alphabet filings.

## Architecture

```mermaid
flowchart TD
  subgraph Offline["Offline indexing"]
    PDF[10-K PDF] --> SEC[TOC-guided sectioning]
    SEC --> CHK[Text chunks + table assets]
    CHK --> TV[(Text vector DB)]
    CHK --> VLM[Optional VLM table parse]
    VLM --> SV[(Table summary vector DB)]
  end

  subgraph Runtime["Runtime"]
    U[User question] --> A[LangChain tool-calling agent]
    A --> SQL[SQL tool]
    A --> RAG[RAG tool]
    A --> EM[send_email optional]
    SQL --> DB[(financials.db)]
    RAG --> TV
    RAG --> SV
    RAG --> AST[assets.json markdown tables]
    SQL --> OBS[Tool observations]
    RAG --> OBS
    OBS --> A
    A --> ANS[Grounded answer]
  end
```

### Offline chunking pipeline

```mermaid
flowchart TD
  A[10-K PDF] --> B[Read text + layout]
  B --> C[Parse visible TOC]
  C --> D[Menu-guided Item matching]
  D --> E[Detect subsections]
  D --> F[Extract tables / images]
  E --> G[Build text RAG chunks]
  F --> G
  G --> H[Metadata: header_path, table_refs, neighbors]
  H --> I[Text vector index]
  F --> J[VLM table parse]
  J --> K[Table summary index]
```

### Dual-path RAG inference

```mermaid
flowchart TD
  Q[Question + ticker + fiscal year] --> F[Metadata filter]
  F --> E[Embed query once]
  E --> VT[Text vector search]
  E --> BM[BM25 text search]
  E --> TS[Table summary vector search]
  VT --> M[Merge text hits]
  BM --> M
  M --> R[Cross-encoder rerank]
  R --> X[Context expansion]
  TS --> TH[Similarity threshold]
  TH --> TC[Table hits]
  X --> CTX[Assemble context]
  TC --> CTX
  CTX --> LLM[Answer model]
  LLM --> OUT[Answer + citations]
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

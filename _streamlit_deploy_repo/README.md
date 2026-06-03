# 10-K Agentic RAG

Code for a local 10-K agentic RAG prototype with chunk inspection UI and agent Q&A.

This repo includes a local chunk inspection UI, a tool-calling agent, and separate retrieval paths for narrative 10-K text and table evidence.

## Design

More detail is in [DESIGN.md](DESIGN.md), but the core design is:

```mermaid
flowchart TD
  PDF[10-K PDFs] --> PARSE[PDF text, layout, table, image parsing]
  PARSE --> SECTION[TOC-guided sectioning]
  SECTION --> SUBSECTION[Layout-based subsection detection]
  SUBSECTION --> CHUNKS[Clean narrative chunks]
  PARSE --> TABLES[Table / image assets]
  TABLES --> VLM[Optional VLM table parse]
  CHUNKS --> TEXT_INDEX[(Text vector index)]
  VLM --> TABLE_INDEX[(Table summary vector index)]

  USER[User question] --> AGENT[Tool-calling agent]
  AGENT --> SQL[SQL tool]
  AGENT --> RAG[RAG tool]
  SQL --> ANSWER[Grounded answer]
  RAG --> ANSWER
  TEXT_INDEX --> RAG
  TABLE_INDEX --> RAG
  TABLES --> RAG
```

### Chunking

The offline pipeline keeps 10-K structure instead of using generic fixed-size PDF chunks.

```mermaid
flowchart TD
  PDF[10-K PDF] --> READ[Read text spans, words, fonts, positions]
  READ --> TOC[Parse visible TOC]
  TOC --> SECTION[Find real Part / Item sections in body order]
  SECTION --> SUB[Detect subsection headings from layout signals]
  SUB --> TEXT_UNITS[Build section preamble + subsection text units]
  SECTION --> ASSETS[Extract tables and images]
  TEXT_UNITS --> CHUNKS[Build clean narrative chunks]
  ASSETS --> CHUNKS
  CHUNKS --> META[Attach section path, table refs, pages, neighbors]
  META --> INDEX[Text vector DB]
```

### RAG

RAG uses text and table evidence separately, then assembles them into one grounded context.

```mermaid
flowchart TD
  Q[Question + ticker / fiscal year] --> FILTER[Metadata hard filter]
  FILTER --> EMBED[Embed retrieval query]
  EMBED --> TEXT_VEC[Text vector search]
  EMBED --> BM25[BM25 search]
  EMBED --> TABLE_VEC[Table summary search]
  TEXT_VEC --> MERGE[Merge text hits]
  BM25 --> MERGE
  MERGE --> RERANK[Cross-encoder rerank]
  RERANK --> EXPAND[Expand with preamble + neighbors]
  TABLE_VEC --> TABLES[Load VLM table markdown]
  EXPAND --> CONTEXT[Final evidence context]
  TABLES --> CONTEXT
  CONTEXT --> LLM[Answer model]
  LLM --> OUT[Answer with chunk/table citations]
```

### Agent Routing

- SQL handles exact metrics, ratios, rankings, and structured financial comparisons.
- RAG handles filing narrative, MD&A, risks, strategy, and table-backed evidence.
- Hybrid questions run SQL first, then use scoped RAG for the filing explanation.

## Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r starter/requirements.txt
python -m chunk_studio.server
```

Open `http://127.0.0.1:8010/` for Chunks and `http://127.0.0.1:8010/agent` for Agent Q&A.

Set API keys in a local `.env` file. Do not commit `.env`.

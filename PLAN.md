# Agentic RAG Build Plan (Revised)

## Environment Facts (post-setup.sh)

- Python: 3.9.18 (miniconda3 base, via uv venv)
- `.venv` already has: `openai 2.37.0`, `pymupdf 1.26.5`, `numpy 2.0.2`, `pandas 2.3.3`, `pydantic 2.x`, `requests`, `httpx`, `tqdm`
- Missing (must add): `fastapi`, `uvicorn`, `python-dotenv`
- Fireworks API key: in `.env` as `FIREWORKS_API_KEY`
- Confirmed working chat model: `accounts/fireworks/models/deepseek-v4-pro`
- Embedding model: `nomic-ai/nomic-embed-text-v1.5` via Fireworks — **confirmed working, 768-dim vectors**

---

## Architecture

```
User (CLI or HTTP POST /api/chat)
        │
        ▼
  Agent Loop  (function-calling, multi-step ReAct)
  LLM: accounts/fireworks/models/deepseek-v4-pro
        │
   ┌────┴────┐
   ▼         ▼
SQL Tool   PDF Tool
sqlite3    PyMuPDF + BM25 (or Fireworks nomic embeddings)
```

---

## Project Structure (what we build)

```
submission/
├── main.py              # FastAPI server + CLI --interactive entry point
├── agent.py             # Multi-step function-calling agent loop
├── tools/
│   ├── sql_tool.py      # Text-to-SQL over financials.db
│   └── pdf_tool.py      # Chunk retrieval from 10-K PDFs
├── indexing/
│   └── build_index.py   # One-time: chunk PDFs, build BM25/vector index, save to disk
├── config.py            # Paths, env vars, model names
├── requirements.txt     # fastapi, uvicorn, python-dotenv (everything else already in .venv)
├── dev_answers.json     # Answers to 10 dev questions (generated last)
└── README.md            # Exact run instructions for reviewer
```

---

## Phase-by-Phase Breakdown

### Phase 1 — Bootstrap (~20 min)
- `config.py`: load `.env`, define DB path, PDF dir, model names
- `requirements.txt`: add `fastapi`, `uvicorn`, `python-dotenv`
- `main.py` skeleton: FastAPI app with `POST /api/chat`, plus `--interactive` CLI mode
- Verify `openai` client connects to Fireworks with `deepseek-v4-pro`

### Phase 2 — SQL Tool (production-grade, multi-step pipeline)

`tools/sql_tool.py` is built as an explicit pipeline — each step is a separate, independently testable function. Before writing any code, thoroughly inspect the database: all tables, columns, data types, and sample rows. This inspection informs every step below.

#### Step 1 — Question Classification (`classify_question`)
Classify the incoming question before doing anything else:
- `"simple"` — single metric, single entity, single time period
- `"multi_part"` — multiple metrics, entities, or time periods in one question

#### Step 2 — Pre-SQL Inspection (`inspect_db`)
Before generating SQL, query the database to resolve any ambiguous references in the question. Do not assume — verify:
- What values actually exist for any text or categorical fields referenced in the question
- What time periods or ranges are available
- Which table and column best represents the concept being asked about
- How the question's natural language maps to exact database values

Output of this step: a resolved, unambiguous restatement of the question using actual database values.

#### Step 3 — Decomposition (`decompose_question`, only for `multi_part` or `comparative`)
Break the question into atomic sub-questions, each answerable by exactly one SQL query. Run each independently, then combine results in Python for any post-processing (arithmetic, ranking, comparison). Do not answer multi-part questions in a single SQL query.

#### Step 4 — SQL Generation (`generate_sql`)
Generate SQL using the resolved question from Step 2.
- Never use `SELECT *` — name columns explicitly
- Cast to `REAL` before any division to avoid integer truncation
- Do post-processing arithmetic (growth rates, differences, rankings) in Python after fetching raw values — not inside SQL
- Add `ORDER BY` / `LIMIT` only when the question asks for ranking or top-N

#### Step 5 — Execution and Error Handling (`execute_with_retry`)
Execute the SQL and handle errors by type, with at most one retry per error type:
- Syntax error → retry once, passing the error message back to fix syntax only
- Wrong table or column → retry once, re-inspecting the schema
- Empty result → retry once, re-examining `WHERE` conditions against actual values from Step 2
- All NULLs → retry once, re-checking column selection
- If still failing after one retry: return a clear explanation of what was attempted and why it failed

#### General rules
- Log the output of each step for debuggability
- Final answers cite which table(s) and time period(s) the data came from
- Block all non-SELECT statements; open the DB read-only via SQLite URI mode
- Each step's logic is independent and testable in isolation

### Phase 3 — PDF Indexing + Retrieval (~45 min)
- `indexing/build_index.py`:
  - Extract text per page via PyMuPDF (`fitz`)
  - Chunk into ~400-token windows with 50-token overlap (count with `len(text.split())` for simplicity)
  - Embed each chunk via Fireworks `nomic-ai/nomic-embed-text-v1.5` (768-dim, confirmed working)
  - Save to `data/index/`: chunk embeddings as `embeddings.npy`, metadata (ticker, year, page, text) as `chunks.json`
- `tools/pdf_tool.py`:
  - Accept query + optional ticker/year filter
  - Embed query with same model, cosine similarity over numpy matrix, return top-5 chunks
  - Each result includes `[SOURCE: AAPL_FY2025, p.12]` citation

### Phase 4 — Agent Loop (~45 min)
- `agent.py`:
  - System prompt: role, available tools (JSON schemas for `query_sql` and `search_pdf`), instructions to cite sources
  - Loop: call LLM → if tool_call, execute tool → append result → call LLM again → repeat until final answer
  - Max 6 iterations to prevent runaway loops
  - Handles all 3 tiers: single-tool (Tier 1), multi-tool sequential (Tier 2), multi-step reasoning (Tier 3)

### Phase 5 — HTTP API + CLI (~20 min)
- `POST /api/chat` → `{"question": "..."}` → `{"answer": "..."}`
- `python main.py --interactive` → REPL loop
- `python main.py` → starts FastAPI on port 8000

### Phase 6 — Index Build + Smoke Test (~15 min)
- `python indexing/build_index.py` — run once, writes to `data/index/`
- Test each tier manually with one question

### Phase 7 — Eval + dev_answers.json (~30 min)
- Run all 10 dev questions through the live agent
- Compare to gold answers: exact numeric match for q_001, q_008, q_011, q_018; prose review for the rest
- Write `dev_answers.json`

### Phase 8 — Report + README (~30 min)
- `README.md` in submission: exact steps (`uv venv`, `uv pip install`, `python indexing/build_index.py`, `python main.py`)
- 1–2 page report: design, routing, retrieval, evaluation, trade-offs, future improvements

---

## Key Design Decisions

| Decision          | Choice                                        | Reason                                                        |
|-------------------|-----------------------------------------------|---------------------------------------------------------------|
| LLM               | `deepseek-v4-pro`                             | Confirmed accessible on this API key, strong reasoning        |
| PDF retrieval     | Fireworks `nomic-embed-text-v1.5` + cosine similarity | Confirmed accessible; better semantic matching than BM25 for narrative text |
| SQL generation    | LLM with full schema in system prompt         | Handles arbitrary queries without hardcoding                  |
| Agent style       | OpenAI-style function calling (openai 2.37.0) | Native to both OpenAI SDK and Fireworks; clean tool interface |
| Server            | FastAPI + uvicorn                             | Minimal, matches `POST /api/chat` spec exactly                |
| Index storage     | numpy `.npy` + JSON metadata on disk          | No extra infra, deterministic, fast to load                   |

---

## Packages to Add (beyond what setup.sh installed)

```
fastapi
uvicorn[standard]
python-dotenv
```

Install via: `uv pip install fastapi "uvicorn[standard]" python-dotenv`

---

## Out of Scope

- Persistent vector DB (e.g. Chroma, Pinecone) — numpy index is sufficient for 6 PDFs
- SSE streaming — JSON response (`{"answer": "..."}`) is explicitly allowed by spec
- Auth / rate limiting — local PoC
- Multi-turn conversation — stateless per-request

---

## Risks / Unknowns

1. **deepseek-v4-pro function calling**: Need to confirm it returns `tool_calls` in OpenAI format; if not, fall back to prompt-based JSON routing.
2. **Python 3.9 vs 3.11**: setup.sh created a 3.9 venv (miniconda base). Should work fine for all packages we need.
3. **Embedding API rate limits**: Building the index embeds ~hundreds of chunks; may need to batch requests with a short sleep if rate-limited.

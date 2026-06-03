# Financial Research Agent

LangChain `create_tool_calling_agent` + `AgentExecutor` over **sql**, **rag**, and **send_email** tools. Chunk Studio `/agent` uses this stack via `chunk_studio/agent_bridge.py`.

## Files

| File | Role |
|------|------|
| `tools.py` | Tool wrappers + `TOOL_SCHEMA` |
| `email_delivery.py` | SMTP for `send_email` |
| `system_prompt.py` | DB schema, filing coverage, LangChain instructions |
| `trace_format.py` | LangChain trace → stream/JSON step payloads |
| `langchain_agent.py` | Agent executor |
| `langsmith_tracing.py` | Optional LangSmith env + run metadata |
| `agent_memory.py` | Session memory (storage, compression, retrieval) |
| `agent.py` | CLI entry point |
| `MEMORY_DESIGN.zh.md` | Memory architecture notes |

## Prompt stack (per turn)

```text
system (instructions + DB/filing schema + TOOL_SCHEMA + optional ## Session memory)
  → chat_history (recent user/assistant only, from SQLite messages)
  → human (current question)
  → agent_scratchpad (this turn’s tool calls + JSON observations only)
```

No separate router classifier. Each iteration: model chooses tool or final answer until done or `max_iterations` (default **6** in Chunk Studio stream).

## Tool error handling

| Layer | Behavior |
|-------|----------|
| **SQL tool** | `text_to_sql.py`: validate SQL + up to **3** LLM `_correct_sql` retries on execution errors |
| **RAG** | `tools.py` `try/except` → JSON with `ok: false`, `error_message` (does not crash the executor). If the first answer looks insufficient, `run_rag_tool()` retries once with a retrieval-focused rewrite while preserving the original user question for answer generation. |
| **Email** | `tools.py` `try/except` → JSON with `ok: false`, `error_message` |
| **Agent** | `handle_parsing_errors=True`; observations include `ok` / `status` / `fallback_trace`. Retry or switch-tool decisions across agent steps are still **prompt-driven** (`system_prompt.py`). |
| **HTTP** | `agent_bridge.py` wraps the run; uncaught exceptions → NDJSON `type: error` |

The model supplies:

- `question` — topic/section wording for semantic retrieval
- `ticker` / `fiscal_year` — optional metadata hard filters (see `RAG_SCOPE_RULES` in `system_prompt.py`)

Scope filtering is applied in `run_pipeline` → `load_chunks()`; putting company/year only in `question` does **not** replace the structured filter params.

### RAG insufficient-answer fallback

There are two fallback layers:

1. **Tool-level fallback** in `run_rag_tool()`: if the answer contains insufficient-context markers such as "not found", "cannot determine", or "does not support a confident answer", the tool retries the same user question with a separate `retrieval_query`. The retry query is designed for search only, so the final answer still addresses the original question.
2. **Agent-policy fallback** in `system_prompt.py`: if a previous RAG observation only says an exact phrase was not found or context was insufficient, the agent should not treat that as enough evidence. It should call RAG again with the same `ticker` / `fiscal_year` and a more concrete single-intent question using filing terms likely to appear in the document, such as components/includes, growth drivers, margins, risks, business descriptions, revenue, operating results, or disclosed reasons.

Example trajectory:

```text
User asks: What does Apple's FY2025 10-K say about the strategic importance of Services?
RAG observation: exact phrase "strategic importance of Services" was not found.
Fallback RAG call: question="What does the 10-K say about Services business growth, revenue contribution, and strategic role in Apple's business?", ticker="AAPL", fiscal_year="FY2025"
```

## Tool-policy evaluation

`eval_tool_policy_100.py` evaluates the agent's next-step tool policy with mocked tools. Each entry is one independent turn: question + existing observations + expected next action. Multi-step workflows are represented as multiple entries, one per turn.

The eval tracks:

- action choice: call a tool vs. answer now
- tool selection: expected tool calls vs. actual tool calls
- argument quality: SQL entity/year/metric recall; RAG ticker/year/single-scope/entity/evidence-term checks
- trajectory behavior: stop after sufficient evidence, SQL-to-RAG dependency satisfaction, RAG decomposition, and missing follow-up rate

The fallback subset was expanded after discovering that previous cases were too homogeneous. The key policy being tested is: an insufficient RAG observation is not a final answer; the agent should reformulate and call RAG again.

### Eval limitations

Tool-policy eval data is hard to generate robustly because the "correct" next tool call is often not unique. Reasonable agents may split one SQL query into multiple SQL calls, use different but valid RAG wording, or choose a different order for independent calls. The current scorer uses string/entity/term matching, so it can miss semantically correct calls and can also over-credit calls that include expected words without being a good retrieval query.

Fallback cases are especially hard to synthesize because they depend on realistic failure observations. If every case says "exact phrase not found", the eval only measures one narrow behavior. A better fallback suite should include diverse failure modes: exact phrase miss, abstract wording miss, too-narrow retrieval, synonym mismatch, table/numeric wording mismatch, wrong section retrieval, and SQL-to-RAG follow-up misses.

## Future Work

### Pre-answer retrieval fallback calibration

A potential future improvement is to decide whether to rewrite and retry the RAG query before the final answer-generation call, using retrieval signals such as top-3 text rerank scores and top table similarity. Early experiments suggest this gate must be conservative:

- Supported single-intent text questions usually have high top-1 rerank scores, but a correct chunk can still appear in the top 3 with a borderline score around `0.59`.
- Supported table questions can have low text rerank scores while the correct table similarity is strong, so text-only thresholds would incorrectly trigger fallback.
- Unsupported-but-Microsoft-relevant questions can still receive high rerank scores because retrieved chunks may be semantically related without containing the exact missing disclosure.

Because of this, retrieval thresholds should only trigger a query rewrite for clearly weak retrieval, not decide whether the document contains an answer. A safer first-pass rule would be based on the maximum top-3 text rerank score plus top table similarity, for example: fallback only when all top text rerank scores are low and no table candidate clears the table threshold.

This needs a better eval set before being enabled as a hard gate: supported text single-intent questions, supported table single-intent questions, and Microsoft/FY2025-relevant but unsupported single-intent questions. The unsupported set should avoid obviously unrelated industries and instead target plausible but undisclosed Microsoft facts such as product usage counts, churn, regional Azure details, OpenAI revenue-share terms, GPU purchase counts, or customer-segment breakdowns.

## HTTP entry (Chunk Studio)

`chunk_studio/agent_bridge.py` → `POST /api/agent/trace/stream` (all six filings).  
Uses merged indexes: `data/index/text_chunks/vectors.db`, `data/index/table_summaries/vectors.db`, `data/index/merged_assets.json`.  
Per-file workspace indexes remain for Chunk Studio processing only.

## Session memory

SQLite `data/agent_memory.db`. Pipeline detail: `MEMORY_DESIGN.zh.md`.

## LangSmith (optional)

Env-gated tracing in `langsmith_tracing.py`; does not change tool or memory logic.

```bash
# .env
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=fireworks-agentic-rag   # optional
```

Each run tags `session_id` / `file_id` in metadata. JSON/CLI payload may include `langsmith.url` for the root trace.

Disable without removing the key: `LANGSMITH_TRACING=false`.

## CLI

```bash
python 0525_redo/agent/agent.py "Your question here"
python 0525_redo/agent/agent.py "..." --json --max-steps 6
```

## Related

- `../CHUNKING_AND_INFERENCE_DESIGN.md`
- `../../chunk_studio/DESIGN.md`
- `MEMORY_DESIGN.zh.md`

## Memory test

```bash
python 0525_redo/agent/test_memory_multiturn.py
```

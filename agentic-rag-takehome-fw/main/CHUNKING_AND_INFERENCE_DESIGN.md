# Chunking and Inference Design

> 中文版：`CHUNKING_AND_INFERENCE_DESIGN.zh.md`

This document describes the current PDF chunking, asset extraction, vector indexing, retrieval, reranking, context expansion, and inference design for the financial RAG system. The current implementation was developed and validated primarily on Apple FY2025 10-K, but the design is intended to generalize across Apple, Microsoft, and Alphabet 10-K filings.

The core goal is to make 10-K text retrievable at the right granularity while preserving enough structural metadata to recover context during inference. The design separates narrative text retrieval from table value retrieval, because financial tables behave differently from prose and should not be silently mixed into text chunks.

For the Chunk Studio product layer — visual table region detection, crop rendering, UI flow, and the column-alignment detector used at API time — see `chunk_studio/DESIGN.md`.

## High-Level Workflow

The pipeline has six stages:

1. Parse the visible 10-K table of contents and use it as a menu for section discovery.
2. Locate main Item sections in the body using menu-guided sequential matching.
3. Use font/style signals to detect smaller subsections inside each Item section.
4. Extract tables/images as separate assets and attach them to the nearest section/subsection.
5. Build text-only RAG chunks with header paths, table references, and inference expansion metadata.
6. Run inference from a separate inference layer using **parallel text + table-summary retrieval**, text-only reranking, table threshold filtering, context expansion, VLM table markdown injection, and answer generation.

Code is split by responsibility:

- `main/chunking/` contains PDF parsing, sectioning, asset extraction, text chunk construction, vector indexing, offline VLM table parsing, table-summary vector indexing, and retrieval hit-rate evaluation.
- `main/inference/` contains runtime RAG inference and RAGAS-style answer evaluation.
- `chunk_studio/` wraps the same pipeline for upload, visualization, VLM parse inspection, and Q&A.

The current Apple FY2025 outputs are:

- Menu-guided sections: `main/chunking/AAPL_FY2025_menu_guided_sections.json`
- Section-linked assets: `main/chunking/AAPL_FY2025_section_assets.json`
- Text-only RAG chunks: `main/chunking/AAPL_FY2025_rag_chunks.json`
- Text vector DB: `data/index/text_chunks/vectors.db`
- Table summary vector DB: `data/index/table_summaries/vectors.db` (after VLM parse + `build_table_vector_db.py`)
- Offline VLM table parse: `main/chunking/vlm_table_parse.py`
- Table summary index builder: `main/chunking/build_table_vector_db.py`
- Inference script: `main/inference/text_vector_rag_inference.py`
- Mixed inference eval: `main/inference/run_mixed_inference_eval.py`
- RAGAS-style mixed eval: `main/inference/eval_msft_mixed_ragas_style.py`
- MSFT mixed test set: `main/common/msft_fy2025_mixed_15_inference_test.json`
- MSFT table test set: `main/common/msft_fy2025_parsed_table_test_questions.json`
- MSFT table eval: `main/inference/eval_msft_table_test_questions.py`
- Inference evaluation script: `main/inference/eval_single_questions_ragas_style.py`

Current Apple FY2025 counts:

- Main sections: 23
- Subsections: 179
- Raw tables detected: 43
- Tables after merge: 42
- Images detected: 1
- Text chunks: 161
- Table chunks in text vector DB: 0
- Text chunks with table references: 19
- Text chunks with image references: 1

## Model Providers and Environment

The recommended runtime split keeps **Fireworks for embeddings only** and **Anthropic for all chat steps** (rerank, answer, RAGAS judge, agent/sql chat):

| Stage | Provider | Default model |
| --- | --- | --- |
| Offline index build + per-query vector search | Fireworks | `nomic-ai/nomic-embed-text-v1.5` |
| Text rerank | Anthropic | `claude-haiku-4-5-20251001` |
| Answer generation | Anthropic | `claude-sonnet-4-20250514` |
| RAGAS judge / agent loop | Anthropic | configured chat model |

When `ANTHROPIC_API_KEY` is set, `call_chat()` never routes to Fireworks, even if a caller passes a Fireworks model id such as `accounts/fireworks/models/qwen3-8b`. `FIREWORKS_API_KEY` is required only for embedding.

Configure via repo-root `.env`:

```bash
FIREWORKS_API_KEY=...          # embeddings only
ANTHROPIC_API_KEY=...
ANTHROPIC_RERANK_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-20250514
```

`load_env_file()` overwrites shell environment variables with values from `.env`, so local key changes take effect after editing the file and reloading.

Offline text/table vectors are already stored in SQLite. At inference time, Fireworks is called **once per question** to embed the query; rerank and answer do not use Fireworks when Anthropic is configured.

## Why The Design Is Menu-Guided

Early attempts based on page ranges and generic Item-heading regex were too brittle. SEC 10-K PDFs often contain repeated Item references in footers, TOCs, cross-references, audit reports, and page headers. Page offsets can also be inconsistent between the PDF page number and the printed 10-K page number.

The current design starts with the visible table of contents because it gives an ordered menu of expected section names. Instead of trusting page numbers, the parser uses the TOC as a sequence of expected Items and then searches through the document body in order. This makes the sectioning more robust across companies and filing styles.

The important principle is: the TOC is used as an ordered semantic menu, not as a page coordinate source.

## Section Detection

The section detection process uses three layers:

1. Identify likely TOC pages near the beginning of the filing.
2. Parse Item entries and titles from those TOC pages.
3. Search the body text sequentially for the matching Item headings and title text.

The body search is character-offset based. Once a section heading is found, the section content is cut from that heading to the next detected heading. This avoids relying on page ranges for the final boundaries.

Each section receives a stable section reference such as:

- `Part I::Item 1`
- `Part I::Item 1A`
- `Part II::Item 7`
- `Part II::Item 8`

The section object keeps metadata like part, item, section title, page span for debugging, character count, and references to tables/images. Large raw text is not kept in the compact asset output, because the RAG-ready representation stores text as chunks instead.

## TOC and Section Corner Cases

The sectioning logic has to handle several PDF-specific problems:

- The visible TOC may include line wrapping, punctuation differences, or repeated dot leaders.
- Some body pages mention “Item 1A” or “Item 7” inside ordinary prose. These should not become section starts.
- Printed page numbers may not match PDF page indices.
- Some filings have short lines that look like TOC entries outside the TOC.
- Some sections have no meaningful preamble and immediately start with a subsection.
- Some sections, especially exhibits, contain list-like content that visually resembles tables.

The menu-guided approach reduces false positives by enforcing order. If the TOC says the next section should be `Item 7`, the matcher does not accept an unrelated `Item 1A` mention later in a paragraph as a new section boundary.

The TOC page detector is intentionally conservative. It scans early pages and requires evidence such as multiple Item entries and page-number-like patterns. This avoids wrongly labeling body pages as TOC pages.

## Subsection Detection

After main Item sections are cut, the pipeline detects smaller subsections using font/style features from the PDF:

- Font size
- Boldness
- Italicization
- Left indentation
- Line length
- Whether the line looks like a standalone heading
- Whether the text ends like a sentence
- Whether the line starts with common paragraph words

This produces subsection paths such as:

- `Business > Products > iPhone`
- `Risk Factors > conflict, terrorism, natural disasters...`
- `Management’s Discussion and Analysis > Gross Margin > Services`
- `Financial Statements and Supplementary Data > Note 4 – Financial Instruments > Cash, Cash Equivalents and Marketable Securities`

Subsections are stored as materialized text units, not just heading labels. Each subsection chunk has a path, title, level, page, and text content.

## Subsection Corner Cases

Subsection detection is heuristic and has some known limitations:

- Some true headings are visually subtle and may look like ordinary prose.
- Some italicized paragraphs look heading-like but are actually explanatory notes.
- Very short headings such as “Services” can be confused with table row labels if table text is still present.
- Long Risk Factor headings can span multiple lines and are hard to distinguish from normal paragraphs.
- Exhibit sections contain many title-like lines that are not semantically useful for retrieval.
- Financial statement notes often have nested headings and tables interleaved tightly.

The current approach is designed to prefer useful retrieval boundaries over perfect visual reconstruction. Some headings may be missed if they do not have strong font/style signals. Some low-value list items may become chunks, especially in exhibit sections.

## Table Extraction and Attachment

Tables are extracted separately from text. The current asset extractor uses PyMuPDF table detection to find bounding boxes, raw rows, page locations, and rough complexity metadata.

Each table is attached to:

- A main section reference
- A subsection reference when possible
- Page start/end
- Bounding box or per-page bounding boxes
- Raw rows
- Raw text representation
- Cross-page merge metadata if applicable

Tables are not embedded as normal text chunks in the text vector DB. Instead, text chunks keep table references, such as `table_004` or `table_group_041`. During inference, table evidence can enter context through two independent paths:

1. **Text chunk refs** — if an expanded text anchor chunk has `table_refs`, load the corresponding table markdown.
2. **Table summary vector retrieval** — if a VLM table summary passes the similarity threshold, load its markdown directly without going through text rerank.

This separation exists because table values should be handled as structured or semi-structured data. Mixing raw table rows directly into prose chunks makes embeddings noisy and causes text retrieval to behave like table lookup.

## Offline VLM Table Parse (Stage 1)

For tables that need reliable numeric QA, the preferred evidence is no longer flattened `raw_rows`. The offline path is:

1. Render a table crop PNG from the PDF bbox (with vertical padding).
2. Collect ±2 sentence narrative context before/after the table from the PDF text layer.
3. Call a VLM (Anthropic or Fireworks fallback) to produce:
   - `markdown` — full GitHub-flavored table
   - `summary` — one-sentence retrieval summary
4. Store the result on each table in `assets.json` under `vlm_parse`.

Script: `main/chunking/vlm_table_parse.py`

CLI filters useful for eval/debug:

- `--min-page 3` — skip TOC / pre-body pages
- `--exclude-item Item 5` — skip stock repurchase tables, etc.
- `--max-tables N` — batch size control

Chunk Studio exposes parsed results in the right-column **Parses** tab and serves crop images via `/api/files/{id}/tables/{table_id}/parse-image.png`.

## Table Summary Vector Index (Stage 2)

After VLM parse succeeds, embed each table's `vlm_parse.summary` into a **separate** vector DB:

- Global path: `data/index/table_summaries/vectors.db`
- Per-workspace path (Chunk Studio): `{workspace}/index/table_vectors.db`

Script: `main/chunking/build_table_vector_db.py`

Important: text chunks and table summaries live in different DBs and are queried independently at inference time. They are **not merged into one candidate pool**.

## Cross-Page Table Handling

Some financial tables span multiple pages. The asset extraction pipeline runs:

```text
find_tables()
  -> same-page fragment merge
  -> synthesize_page_bottom_header_tables()   # headers missed by find_tables()
  -> supplement_header_only_continuations()   # synthetic top-of-page data rows
  -> merge_connected_tables()                 # scored cross-page merge
```

### Normal merge candidates

- Previous table near page bottom (`page_height - bbox.y1 < 95pt`).
- Next table starts near top of following page (`bbox.y1 < 130pt`).
- Compatible column counts or widths.
- Continuation does not look like an unrelated financial statement.

### Header-only degenerate tables (MSFT Item 5 dividend case)

The hardest representative case:

```text
page 32 bottom:
  Declaration Date | Record Date | Payment Date | Dividend Per Share | Amount
  (often missed entirely by find_tables())

page 33 top:
  June 10, 2025 | August 21, 2025 | September 11, 2025 | $0.83 | $6,170
  (often detected, but easily assigned to the wrong Item)
```

Handling strategy:

1. Mark header-only, page-bottom tables as `pending_merge=True` (including synthesized `table_header_band` assets).
2. Synthesize bottom header bands from the PDF text layer when `find_tables()` misses them.
3. Lower merge threshold for header-only predecessors; allow column-count mismatch between header and data halves.
4. Keep subsection/section attribution on the header half after merge.
5. Assign tables with `y0 < 150pt` to the previous page's section when content clearly continues across the page break.

Merge logic remains conservative for normal tables. Header-only cross-page continuation is an explicit exception because geometry is strong evidence.

Merged tables receive a group id such as `table_group_003`. Evaluation should treat merged groups and their source fragments as semantically equivalent when appropriate.

### Subsection detection: centered ALL CAPS headings

For headings like `MARKET AND STOCKHOLDERS`, scoring uses line center (not left x0), uppercase ratio, and vertical gaps. Score >= 4 makes a line a subsection candidate.

## Table Corner Cases

Tables are the hardest part of the current system. Important corner cases include:

- PyMuPDF may detect only part of a table.
- A table title may be outside the detected table bounding box.
- Repeated headers can appear on continuation pages.
- Some financial statements are visually tables but semantically broad documents.
- Some rows are hierarchical, where indentation matters.
- Parenthetical negative values can be represented inconsistently.
- A table may be attached to the right section but the wrong subsection if headings are near page boundaries.
- Some “tables” in exhibit sections are really lists of exhibits.
- A merged table id may differ from the table id referenced by a nearby text chunk.
- A text question can be generated from a table-leading sentence but require table values that are not present in text.

The current design handles these by keeping table references and raw table text. It does not yet fully normalize tables into row-level or value-level records. For high-quality table QA, a future layer should parse tables into normalized records or use a table-specific answer prompt.

## Image Extraction

Images are extracted as separate assets and attached to sections/subsections when they pass size filters. Small decorative or noisy image blocks are filtered out. Image chunks are not embedded in the current text vector DB. Text chunks keep `image_refs` so that a future image/VLM path can retrieve and process relevant figures if needed.

## RAG Text Chunk Construction

The final RAG chunks are text-only. Each chunk contains:

- A stable chunk id
- Source file
- Section reference
- Subsection reference
- Header path
- Text content
- Token count
- Table references
- Image references
- Split metadata
- Inference expansion metadata

The text content includes the header path at the beginning. This improves semantic retrieval because short chunks like `iPhone` or `Services` otherwise lack enough standalone context.

Chunk text intentionally removes:

- Table rows and flattened numeric table runs
- PDF page markers
- Repeated footers
- Repeated Item headings
- Redundant section labels already represented by the header path

The current target is roughly 100-500 tokens per chunk. Some chunks are shorter when the source subsection itself is short. The system does not force short chunks to merge at index time, because merging can blur precise section boundaries. Instead, short chunks are handled at inference time through context expansion.

## Text Chunk Corner Cases

The text chunk builder has to handle several tricky cases:

- Some subsections are naturally short, such as product descriptions.
- Some section preambles are important background but do not directly answer most questions.
- Some subsection text includes table-leading sentences like “The following table shows...”
- Some table values remain in flattened PDF text unless explicitly stripped.
- Some financial statement pages contain table captions and statement titles that look like prose.
- Exhibit sections can produce long, list-like text chunks.
- Removing too much table-like text can harm questions that ask about statement names or table descriptions.
- Keeping too much table-like text harms text embeddings.

The current compromise is text-only retrieval with table references. If a chunk references a table, table content can be added during inference, but the table rows are not embedded as normal prose.

## Inference Expansion Metadata

The RAG chunks now include metadata specifically for inference-time context expansion. This is separate from chunk ownership metadata.

Ownership metadata tells us where the chunk belongs:

- `section_ref_id`
- `subsection_ref_id`
- `section_title`
- `header_path`
- `text_unit_id`
- `text_unit_kind`

Expansion metadata tells us what to add at answer time:

- Section preamble references
- Previous/next chunk in the same split text unit
- Previous/next chunk in the same section
- Whether cross-section expansion is allowed
- Expansion scope

The key design choice is that expansion happens only within the same section. Cross-section expansion is disabled by default. This prevents retrieval from accidentally pulling unrelated sections into the answer context.

During final prompt formatting, selected anchor chunks are kept in full, and section preambles are kept in full. Adjacent previous/next chunks are shortened to the first two and last two sentences. This preserves boundary context while avoiding large prompt growth from neighboring subsections.

## Why Section Preamble References Exist

Some Items have a section-level introduction followed by many subsections. For example, `Risk Factors` may begin with a general risk disclosure before listing individual risk factors. A short risk subsection may be difficult to interpret without that preamble.

Instead of duplicating the preamble into every chunk, each subsection chunk stores a reference to the section preamble chunk. During inference, if a subsection is selected, the preamble can be added once.

This avoids bloating the vector DB while preserving useful context for generation.

## Why Same-Text-Unit Links Exist

When a single subsection is longer than the target chunk size, it is split into multiple chunks. Those chunks are fragments of the same original logical text unit. If retrieval hits one fragment, inference should often include the previous and next fragment.

The same-text-unit links support this. They are especially important when a sentence-aware splitter cuts a long subsection into several chunks but the answer needs surrounding context.

## Why Same-Section Neighbor Links Exist

Some questions naturally require nearby subsections in the same Item section. For example:

- Product announcement context followed by tariff discussion in MD&A
- Commercial paper followed by term debt
- Share repurchase followed by shares of common stock
- Risk factor preamble followed by specific risk factor chunks

The same-section neighbor links let inference add nearby chunks, but only within the same Item section. This is intentionally different from cross-section expansion, which remains disabled.

## Vector Indexing

### Text vector DB

Only text chunks are embedded into the text vector DB:

- Global: `data/index/text_chunks/vectors.db`
- Chunk Studio workspace: `{workspace}/index/vectors.db`

### Table summary vector DB

Only successful VLM table summaries are embedded into the table summary vector DB:

- Global: `data/index/table_summaries/vectors.db`
- Chunk Studio workspace: `{workspace}/index/table_vectors.db`

The current embedding model is:

- `nomic-ai/nomic-embed-text-v1.5`

The embedding dimension is 768. Embeddings are stored as float32 blobs in SQLite, along with chunk content and metadata JSON. The vector search computes cosine similarity in Python over the stored embeddings.

Each row also stores filing identity columns parsed from the source filename:

- `ticker` — `AAPL`, `MSFT`, or `GOOGL`
- `fiscal_year` — `FY2024` or `FY2025`
- `source_file` — e.g. `MSFT_FY2025_10-K.pdf`

Chunk Studio uploads copy the PDF to `source.pdf` on disk, but indexing resolves the canonical filename from workspace `metadata.json` (`original_filename`) via `main/chunking/filing_metadata.py`. Existing workspaces can be patched with `main/chunking/patch_workspace_source_files.py`.

Multiple filings can be merged into one global DB (`data/index/text_chunks/vectors.db`) by passing multiple `*_rag_chunks.json` files to `build_text_vector_db.py`.

Fireworks is used for embeddings. In this environment, Python urllib requests to Fireworks were blocked by Cloudflare with 403 errors, while curl worked with the same key and payload. For that reason, the embedding helper currently calls Fireworks through curl.

## Retrieval And Inference Flow

The current inference path is a **dual-path design**: text and table summaries are retrieved in parallel, reranked/filtered separately, then assembled into one answer context.

### Step 0 — Optional metadata scope filter (multi-filing)

Before vector/BM25 search, `load_chunks()` can restrict candidates with SQL `WHERE` on `ticker` and `fiscal_year`:

```python
run_pipeline(
    query,
    ticker_filter="MSFT",          # or ["MSFT", "GOOGL"]
    fiscal_year_filter="FY2025",   # or "2025"
)
```

CLI flags: `--ticker MSFT --fiscal-year FY2025`.

When filters are omitted, all indexed filings participate (~1500 text chunks for six 10-Ks). When both are set, the candidate pool narrows to one filing (~250 chunks), which greatly reduces cross-company Item-section collisions.

The agent passes scope through structured `rag` tool parameters (`ticker`, `fiscal_year`); the system prompt defines when to pass, omit, or multi-pass tickers. Hard filtering is implemented in code, not by parsing company names out of the question string alone.

### Step 1 — Parallel similarity search

One query embedding is computed once, then used against two independent DBs:

```text
query embedding
    ↓              ↓
text DB          table summary DB
top 10           top 5
```

The two result sets are **not merged and do not compete** with each other.

### Step 2 — Text-side rerank

Only the text top 10 candidates are sent to the reranker. The reranker selects **top 3 text anchor chunks**.

Table summaries do **not** participate in rerank. Rerank is text-only.

Implementation: `rerank_text_chunks()` in `text_vector_rag_inference.py`.

### Step 3 — Table-side threshold filter

From table top 5, keep only hits with cosine similarity ≥ threshold. Default threshold:

- `table_similarity_threshold = 0.75`

Hits below threshold are discarded. This prevents weak table matches from polluting the prompt.

Threshold should be calibrated on a small labeled query set. On MSFT FY2025 table tests, most target tables scored 0.76–0.86; one EPS table (`table_006_merged`) scored 0.712 and was filtered at 0.75 but still answered correctly via text chunk refs.

On the MSFT FY2025 mixed 15-question set (10 table + 5 text), threshold 0.75 was often too strict: only about 1/10 table questions passed the threshold, while about 6/10 passed at 0.70. For production MSFT-style filings, start closer to **0.65–0.70** unless a labeled eval set confirms 0.75 is safe.

### Step 4 — Context assembly

For each **text anchor chunk** (rerank top 3):

- add section preamble (full text)
- add previous/next chunk within the same section (trimmed to first 2 + last 2 sentences in the final prompt)
- inspect `table_refs` and load corresponding table markdown

For each **table summary hit** that passed threshold:

- load corresponding VLM `markdown` directly

Dedup rule: the same `table_id` appears at most once in the final table context.

Final prompt order:

1. section preambles
2. text anchor / neighbor chunks
3. table markdown blocks, each prefixed as:

```text
[Table: table_005 | Item 7 > Management’s Discussion ...]
```

Fallback order for table body:

1. `vlm_parse.markdown` when `vlm_parse.status == success`
2. flattened `raw_rows`
3. `raw_text`

### Step 5 — Answer generation

Send assembled text + table context to the answer model. When `ANTHROPIC_API_KEY` is set, answer generation uses Anthropic (default Sonnet). Rerank uses a separate, faster Anthropic model (default Haiku). Fireworks chat is not used in this configuration.

Table contexts are highest-priority evidence for numeric/table-value questions.

This design intentionally separates:

- **text anchors** — chosen by rerank
- **table evidence** — chosen by table-summary vector search + threshold, plus optional `table_refs` from text anchors

Retrieval finds the best anchors on each side. Inference stitches them together at answer time.

## Reranking

Reranking applies **only to text chunks**.

The reranker receives the text top 10 vector candidates and chooses the top 3 text chunks most useful for answering the query. Table summaries never enter this step.

Default rerank model when `ANTHROPIC_API_KEY` is set: **Haiku** (`ANTHROPIC_RERANK_MODEL`). This is much faster than earlier Fireworks rerank runs (~3s vs ~15s per question on MSFT mixed evals) while keeping answer quality acceptable when answer generation uses Sonnet.

The reranker is especially useful for:

- Choosing between similar Risk Factor chunks
- Choosing the right financial note among several related notes
- Selecting a table-leading subsection over a generic section preamble
- Avoiding irrelevant but semantically broad chunks

The reranked top 3 text chunks are anchors for text-side expansion. They are not the only evidence used at answer time, because table markdown may enter separately through the table threshold path.

If the rerank model returns invalid JSON, the pipeline falls back to vector order for text anchors.

## Table Context During Inference

Table evidence enters the answer context through two paths:

### Path A — Table summary retrieval + threshold

1. Retrieve top 5 table summaries from the table vector DB.
2. Keep hits with similarity ≥ `table_similarity_threshold` (default 0.75).
3. Load VLM markdown for those table ids.

This path does not depend on text rerank. It fixes the earlier failure mode where table summaries were retrieved correctly but dropped because rerank selected only text chunks.

### Path B — Text chunk `table_refs`

If any expanded text anchor chunk references `table_004`, load that table's markdown as secondary evidence. This remains useful when:

- the target table summary score is slightly below threshold
- the question is naturally anchored to narrative text that cites a nearby table

Both paths dedupe on `table_id`.

Each table block in the final prompt uses:

```text
[Table: {table_id} | {section/header path}]
markdown:
...
```

The answer prompt treats table contexts as highest-priority evidence for numeric/table-value questions.

Current limitation: if VLM parse has not been run for a table, the pipeline falls back to flattened `raw_rows` / `raw_text`, which is weaker for complex layouts. For production table QA, run `vlm_table_parse.py` before building the table summary index.

## Query Decomposition

Query decomposition was considered for multi-chunk questions. Based on evaluation, the recommendation is:

- For ordinary single-topic questions, use direct retrieval.
- For explicit multi-intent questions, decomposition can help.
- Regardless of decomposition, use context expansion after retrieval.

The main reason is that many multi-chunk failures are not pure embedding failures. Some generated evaluation groups include adjacent chunks that are not truly needed. In those cases, decomposition would not help much. For genuine multi-hop questions, such as one part about commercial paper and another part about share repurchases, decomposition would likely improve recall.

## Evaluation Summary

Several sanity evaluations were run.

Single-chunk retrieval, 100 generated questions:

- Hit@1: 80%
- Hit@3: 94%
- Hit@5: 97%
- Hit@10: 97%

Cross-chunk retrieval, mixed groups of 2-5 chunks:

- Any target hit@10: 98%
- Mean target recall@10: 65%
- All targets hit@10: 30%

For group size 2:

- Mean target recall@10: 87.5%
- All targets hit@10: 75%
- Any target hit@10: 100%

For group size 3:

- Mean target recall@10: 63.6%
- All targets hit@10: 27.3%
- Any target hit@10: 100%

RAGAS-style mixed inference evaluation, 30 text questions plus 5 table questions:

- Target in vector top 10: 97.1%
- Target in rerank top 3: 94.3%
- Target in expanded context: 94.3%
- Target table in context: 80%
- Mean faithfulness: 4.89 / 5
- Mean answer relevance: 4.69 / 5
- Mean context precision: 3.91 / 5
- Mean reference coverage: 4.54 / 5
- Mean end-to-end latency: 6.75 seconds
- Max latency: 12.97 seconds

Text questions performed better than table questions. Table questions had lower answer relevance and reference coverage because the model sometimes failed to read the table raw text correctly, even when the target table was present.

After adding adjacent-context trimming, a non-table 30-question RAGAS-style run produced:

- Target in vector top 10: 100%
- Target in rerank top 3: 96.7%
- Target in expanded context: 96.7%
- Mean faithfulness: 4.90 / 5
- Mean answer relevance: 4.83 / 5
- Mean context precision: 3.97 / 5
- Mean reference coverage: 4.67 / 5
- Mean end-to-end latency: 7.15 seconds
- Max latency: 15.28 seconds
- Pass rate: 93.3%

A separate 10-question table-focused run produced:

- Target table in context: 80%
- Mean faithfulness: 4.40 / 5
- Mean answer relevance: 4.90 / 5
- Mean context precision: 3.70 / 5
- Mean reference coverage: 4.60 / 5
- Mean end-to-end latency: 11.49 seconds
- Pass rate: 70%
- Borderline rate: 10%
- Fail rate: 20%

### MSFT FY2025 parsed-table regression (10 questions)

Test set: `main/common/msft_fy2025_parsed_table_test_questions.json`

Tables under test: first 10 VLM-parsed Item 7 MD&A tables in the MSFT Chunk Studio workspace (pages 33–40).

#### Old merged rerank pipeline

Settings: text top 10 + table top 5 merged into one rerank pool, rerank top 3.

| Metric | Result |
| --- | --- |
| Table vector hit@5 | 10/10 |
| Target table in rerank top 3 | 1/10 |
| Target table in final context | 6/10 |
| Answer correctness (manual check) | 8/10 |

Main failure mode: table summaries were retrieved but rerank selected only text chunks. Many answers still looked correct because MD&A narrative text or `table_refs` happened to contain the same numbers.

Dividend questions (`table_005`) failed because rerank chose dividend-related text with wrong `table_refs` and never injected the small dividend table markdown.

#### New dual-path pipeline

Settings: text top 10 rerank top 3 + table top 5 threshold 0.75, independent paths.

| Metric | Result |
| --- | --- |
| Table vector hit@5 | 10/10 |
| Target table passes threshold | 9/10 |
| Target table in final context | 10/10 |
| Answer correctness (manual check) | 10/10 |

Notes:

- `table_006_merged` for diluted EPS scored 0.703 and was filtered at threshold 0.75, but still answered correctly via text chunk refs.
- Mean end-to-end latency on this run was about 22 seconds with Fireworks chat; rerank dominated latency.
- Automated numeric matcher in the eval script under-reported correctness because it did not normalize “million” units consistently.

Eval outputs:

- `main/inference/msft_fy2025_table_test_eval.json` — old pipeline
- `main/inference/msft_fy2025_table_test_eval_v2.json` — dual-path pipeline
- corresponding `.jsonl` logs with per-question pipeline paths and latency

### MSFT FY2025 mixed inference (15 questions: 10 table + 5 text)

Test set: `main/common/msft_fy2025_mixed_15_inference_test.json`

Workspace: `data/chunk_studio/1779921176-msft-fy2025-10-k-8d505c867d/`

Representative 5-question runs with Haiku rerank + Sonnet answer:

| Setting | Numeric correctness | Mean latency |
| --- | --- | --- |
| Haiku rerank + Haiku answer | 4/5 | ~26s |
| Haiku rerank + Sonnet answer | 3/5 | ~9.4s (RAGAS replay) |

Notes:

- Lower latency came mainly from faster Haiku rerank and shorter Sonnet answers, not from skipping retrieval.
- RAGAS-style eval can replay cached vector hits when Fireworks embedding is unavailable; use `--no-replay` for full end-to-end latency and retrieval measurement.
- Main table failures at threshold 0.75 were threshold/filter issues, not rerank misses.

Eval artifacts:

- `main/inference/msft_fy2025_mixed_5_anthropic_eval.json`
- `main/inference/msft_fy2025_mixed_5_ragas_style_eval.json`
- `main/inference/msft_fy2025_mixed_15_inference_results.json`

## Inference Latency

The current inference path records latency for:

- Loading chunks and table assets
- Vector search (shared query embedding + text/table DB lookup)
- Text rerank
- Context expansion
- Final answer generation
- Total time

On the MSFT 10-question dual-path eval with Fireworks chat, mean end-to-end latency was about 22 seconds and rerank was usually the slowest stage.

With Anthropic Haiku rerank + Sonnet answer on MSFT mixed questions, rerank dropped to roughly 2–4 seconds and total latency to roughly 8–12 seconds on short eval batches, depending on context size and replay mode.

Latency could be reduced by:

- Skipping rerank for very high-confidence text top1 retrieval.
- Using a smaller/faster rerank model.
- Reducing context expansion when the selected chunks are already long.
- Trimming adjacent chunk context, which is now implemented for previous/next neighbors.
- Caching embeddings for repeated questions.
- Caching table lookups and chunk metadata in memory for a server process.
- Running VLM parse offline so inference never waits on image parsing.

## Known Weaknesses

The current design works well for text retrieval and much better for parsed-table QA than the old merged rerank path, but some weaknesses remain:

- Table QA quality depends on offline VLM parse coverage; unparsed tables still fall back to weak flattened text.
- Unparsed or weakly linked tables can still enter context through text-chunk `table_refs`, even when table-summary retrieval fails or is below threshold.
- `table_similarity_threshold` needs calibration; too high drops useful tables, too low adds noise.
- Some table-leading text questions require table content but may be evaluated as text questions.
- Cross-chunk generated eval questions can be noisy if groups are random adjacent chunks.
- Exhibit sections produce low-value list chunks.
- Some section/subsection boundaries are imperfect due to font detection limitations.
- Text rerank top 3 may exclude useful secondary narrative evidence for broad questions.
- Same-section neighbor expansion can add noise if a section has many loosely related subsections.
- Adjacent chunk trimming reduces prompt size but can remove details if the answer depends on the middle of a neighboring chunk.
- `table_refs` attachment can point to the wrong parsed table for small/isolated tables (for example dividend tables linked to unrelated nearby tables).
- Derived table questions, such as percentages, fail when the denominator table is not retrieved or linked through expansion.
- Section preambles are helpful, but not every section has a meaningful preamble.

## Recommended Production Inference Policy

The recommended inference policy is:

1. Embed the query once.
2. Retrieve text top 10 and table-summary top 5 **in parallel**.
3. Rerank **text only** to select top 3 anchor chunks.
4. Filter table-summary hits with `table_similarity_threshold` (start at 0.70 for MSFT-style filings; calibrate on labeled queries).
5. Expand text anchor chunks conservatively:
   - Always include section preamble references.
   - Include same-text-unit previous/next chunks, but trim adjacent chunks in the final prompt.
   - Include same-section previous/next chunks when enabled, but trim adjacent chunks in the final prompt.
   - Never cross section boundaries automatically.
6. Load table markdown for:
   - all threshold-passing table-summary hits, and
   - any `table_refs` attached to expanded text chunks.
7. Dedupe table ids and format tables as `[Table: id | section]`.
8. If the query has multiple independent clauses, consider query decomposition before retrieval.

This policy keeps text and table retrieval independent, avoids table summaries being dropped by text rerank, and still allows narrative context through text anchors.

## Future Improvements

The most valuable next improvements are:

- Calibrate `table_similarity_threshold` per filing / embedding model on labeled table questions.
- Improve `table_refs` linking for small/isolated tables (dividend tables, note tables near page breaks).
- Normalize table rows into structured records for deterministic arithmetic checks.
- Improve evaluation generation so cross-chunk questions only use coherent same-section groups.
- Add query decomposition for explicit multi-hop questions.
- Improve subsection detection for subtle headings and financial notes.
- Add a context budgeter that chooses preamble, same-text-unit neighbors, same-section neighbors, and tables based on token budget.
- Wire VLM parse + table summary indexing into the Chunk Studio process pipeline by default.
- Add server-side caching for low-latency repeated inference.
- For visually complex tables, optionally pass table crop images to a multimodal answer step in addition to VLM markdown.

## Agent Layer (orchestration above inference)

The agent **does not replace** the RAG pipeline in the sections above; it calls tools that wrap SQL and `run_pipeline`.

### End-to-end (one user question)

```text
[Offline]
  PDF → chunking (sections, text chunks, assets, optional VLM tables)
      → embed → data/index/text_chunks/vectors.db (+ table summary DB)
      → Chunk Studio workspace: {file}/index/vectors.db, assets.json

[Per turn — agent]
  session_id → build_agent_memory → chat_history + memory_context
  AgentExecutor loop (max_iterations ≈ 6):
    LLM → tool sql | rag | send_email | final answer
      sql:  NL question → TEXT_TO_SQL LLM → validate → execute → up to 3× correct_sql on error
      rag:  optional ticker/fiscal_year filter → embed → vector top10 + BM25 top10 + table top5
            → text rerank top3 → threshold tables → expand neighbors → answer_query LLM
      observations as JSON (ok, status, rows / answer / scope_filters / error_message) → scratchpad
  append_turn → SQLite memory; prune → fold overflow into sessions.summary (truncate, not LLM)
```

| Entry | Code |
|-------|------|
| HTTP | `chunk_studio/agent_bridge.py` |
| CLI | `main/agent/agent.py` → `langchain_agent.run_langchain_agent` |

**Routing:** No separate classifier — each step is LLM tool-calling (`system_prompt.py` + `TOOL_SCHEMA` + scratchpad + memory).

**RAG scope:** The `rag` tool accepts optional `ticker` and `fiscal_year` parameters. `system_prompt.py` (`RAG_SCOPE_RULES`) tells the model when to pass them (after sql, user-specified scope, cross-company comparisons). Retrieval applies the filter in `load_chunks()`; the question string carries topic/section wording only.

**Memory:** `agent_memory.py` / `MEMORY_DESIGN.zh.md`.

**Errors:** SQL retries in `text_to_sql.py`; RAG/email exceptions → `ok: false` JSON; agent retry/switch tool is **prompt-only**. See `agent/README.md`.

**SQL empty:** `empty_result` — agent may reformulate or call rag (prompt).

**RAG weak evidence:** model may say insufficient context; retrieval can still return low-score chunks.

**Not implemented in agent/RAG path:** retrieval-time query decomposition/rewrite (eval scripts generate questions separately).

## Bottom Line

The system uses precise, metadata-rich text chunks for retrieval and delays context stitching until inference. This is the right tradeoff for 10-K filings: small chunks retrieve better, while metadata-driven expansion restores the surrounding context needed for answer generation.

The strongest part of the current system is text retrieval and text-grounded answering. Table QA improved materially once the pipeline stopped forcing table summaries through text rerank. The recommended production shape is:

```text
text path:   optional ticker/year filter -> vector top10 + BM25 top10 -> text rerank top3 -> expand neighbors/refs
table path:  optional ticker/year filter -> summary vector top5 -> threshold filter -> inject VLM markdown
answer:      preamble + text + [Table: id | section] markdown blocks
agent rag:   rag(question, ticker?, fiscal_year?) -> run_pipeline filters -> dual path above
```

Robust table QA still depends on offline VLM parse quality, threshold calibration, and correct `table_refs`, but the dual-path design matches how financial 10-K evidence actually behaves: narrative and tables are related, but should not compete in the same rerank pool.

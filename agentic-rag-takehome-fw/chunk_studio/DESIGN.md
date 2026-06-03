# Chunk Studio Design

> 中文版：`DESIGN.zh.md`

This document describes the current Chunk Studio product design: upload a 10-K PDF, generate sections/chunks/assets, visualize tables and figures, and (optionally) run Q&A over the chunk result.

The core product goal is **correct visual table regions**, not perfect cell parsing. Financial 10-K tables often carry semantics through layout (indentation, bold headers, subtotals, gray bands, cross-page continuation). For this product, the screenshot must be right first; parsed rows/cells are secondary metadata.

Implementation lives primarily in:

- `chunk_studio/server.py` — FastAPI backend, visual table detector, crop rendering
- `chunk_studio/static/index.html` — single-page UI
- `0525_redo/chunking/` — original sectioning, asset extraction, chunk building pipeline reused by Chunk Studio

For the broader RAG chunking and inference design, see `0525_redo/CHUNKING_AND_INFERENCE_DESIGN.md`.

---

## High-Level Architecture

```text
Upload PDF
  -> workspace/{file_id}/
       source.pdf
       sections.json
       assets.json
       chunks.json
       metadata.json
       index/vectors.db   (optional)

UI
  -> list files / activity log / hierarchical chunk tree
  -> hierarchical table + figure tree
  -> table crop preview (PNG)
  -> optional Q&A
```

Chunk Studio is a thin product layer around the existing `0525_redo` pipeline:

1. **Sectioning** — menu-guided TOC + body heading detection (`toc_guided_section_probe.py`)
2. **Asset extraction** — PyMuPDF `find_tables()` + image extraction (`section_asset_extractor.py`)
3. **Chunk building** — text-only RAG chunks with table/image refs (`rag_chunk_builder.py`)
4. **Visual table serving** — column-alignment detector at API time (`chunk_studio/server.py`)
5. **Offline VLM table parse** — table crop → markdown + summary (`0525_redo/chunking/vlm_table_parse.py`)
6. **Q&A inference** — dual-path text + table-summary retrieval (`0525_redo/inference/text_vector_rag_inference.py`)

Important split:

| Stage | Responsibility |
| --- | --- |
| Offline process | sections, raw assets, chunks |
| API-time visual layer | table region detection, merge, crop rendering for UI |
| Offline VLM parse | table crop PNG → markdown + summary stored in `assets.json` |
| Q&A | dual-path retrieval: text rerank + table threshold |

The visual detector runs when `/api/files/{id}/assets` or table crop endpoints are called. It does **not** require re-processing the PDF after detector changes.

VLM parse is an offline/batch step (CLI or future process hook). Parsed results are inspectable in the right-column **Parses** tab.

---

## Processing Flow

When the user clicks **Process**, the backend runs three steps and writes progress into `metadata.json`:

| Step | Status key | What happens |
| --- | --- | --- |
| Reading structure | `sectioning` | Parse visible TOC, locate Item sections, detect subsections |
| Finding assets | `extracting_assets` | Run `build_asset_payload()` for tables/images |
| Writing chunks | `building_chunks` | Run `build_rag_payload()` for text chunks |

Default Chunk Studio processing does **not** build embeddings. Q&A is optional and requires vector index build.

Workspace layout:

```text
data/chunk_studio/{file_id}/
  source.pdf
  sections.json
  assets.json
  chunks.json
  metadata.json
  index/vectors.db        # only if embeddings enabled
```

---

## Design Principle: Visual-First Tables

### Why `find_tables()` alone is not enough

PyMuPDF `page.find_tables()` is useful but brittle on 10-K financial tables because:

- PDF has no real table semantics — only words, coordinates, lines, rectangles
- Many financial tables have no full grid lines
- Columns are aligned by whitespace, not borders
- One logical table is often detected as many one-row fragments
- Cross-page continuation has no explicit PDF link
- Section/subsection attachment can split one visual table across many metadata buckets
- Wrapped labels, indentation, bold subheaders, and subtotals break naive row/cell grouping

So the old flow was:

```text
find_tables() -> fragmented bboxes -> merge heuristics -> crop
```

That produced too many small crops and missed cross-page context.

The current flow is:

```text
words + geometry -> visual table region -> crop
find_tables() -> metadata / fallback only
```

### Why visual inference is the right long-term direction

Financial tables are not always flat matrices. Layout often encodes dependency:

- indentation = parent/child row hierarchy
- bold = section header or subtotal
- gray background = grouped block
- horizontal rules = subtotal boundaries
- blank spacing = logical breaks
- cross-page layout = continuation of the same statement

For complex table Q&A, the reliable evidence is the **table image**, not flattened cell text. Parsed cells help retrieval and rough grounding; final reasoning should use table crops plus nearby narrative context, ideally with a multimodal model at answer time.

Chunk Studio currently uses **geometry heuristics only** for detection. It does **not** run VLM on every page during processing.

---

## Table Detection: Three Layers

Chunk Studio currently combines three layers.

### Layer 1 — Raw extraction (`section_asset_extractor.py`)

During process:

- PyMuPDF `find_tables()` extracts table fragments
- Same-page fragment merge
- Cross-page connected-table merge
- Tables/images attached to nearest section/subsection

This produces `assets.json` with entries like `table_001`, `table_group_137`, etc.

### Layer 2 — Parsed-table visual merge (API fallback)

If column-alignment detection fails, the API falls back to merging parsed fragments by geometry:

**Same-page merge (`visual_table_group`)**

- same page
- similar left edge and width
- similar column count
- small vertical gap between fragments
- does **not** require same subsection (subsection attachment is unreliable for financial rows)

**Cross-page merge on parsed groups**

- previous crop ends near page bottom
- next fragment starts near page top
- compatible width/columns
- absorbs continuation rows on the next page

### Layer 3 — Column-alignment visual regions (primary, API-time)

This is the main detector used by the UI today.

It does **not** call any model. It uses PyMuPDF word coordinates only.

---

## Column-Alignment Detector

### Input

For each PDF page:

```python
page.get_text("words")
```

Each word has `(x0, y0, x1, y1, text, ...)`.

Example simplified rows:

```text
row 1: Assets                     $ 128,335   95,466
       left_anchor=42             anchors=[456, 510, 570]

row 2: Accounts receivable                 43,052   56,924
       left_anchor=42             anchors=[510, 570]

row 3: Inventory                           1,234    1,100
       left_anchor=42             anchors=[510, 570]
```

### Step 1 — Cluster words into visual rows

- sort words by vertical center, then x
- group words whose y-centers differ by <= 3.5pt into one row

### Step 2 — Compute row features

For each row:

- `bbox`
- `text`
- `numeric_anchors`: right edges (`x1`) of number-like tokens, rounded to 6pt
- `left_anchor`: rounded x0 of first token
- `numeric_count`
- `is_tableish`

A row is table-ish if:

- it has >= 2 numeric anchors, or
- it has >= 4 words, >= 1 numeric anchor, and horizontal spread > 220pt

Number-like tokens match patterns such as:

- `128,335`
- `(2,625)`
- `$`
- `%`
- `-`, `—`

### Step 3 — Find consecutive aligned runs

Scan consecutive table-ish rows. Extend the run while `_same_alignment_run()` is true:

- vertical gap between rows: `-3 .. 42` pt
- numeric anchor overlap >= 2, or
- anchor overlap >= 1 and left anchors within 10pt

Accept the run if:

- length >= 3 rows, or
- length >= 2 rows and max numeric count >= 3

### Step 4 — Expand region vertically

After a run is found, expand up/down:

**Upward expansion** for header/unit rows if:

- gap <= 34pt
- row is wide (> 180pt)
- text looks header-ish: `(in millions)`, `year`, `june`, `september`, `ended`, or contains numbers

**Downward expansion** while next row:

- gap <= 34pt
- still table-ish or numeric

Final bbox uses full page width:

```text
x0 = 0
x1 = page width
y0 = first used row top - 18pt
y1 = last used row bottom + 18pt
```

### Step 5 — Merge regions on same page

Merge adjacent regions on the same page if:

- vertical gap <= 24pt
- numeric anchor overlap >= 2

### Step 6 — Merge cross-page visual regions

Link region A on page N to region B on page N+1 if:

- A ends near page bottom (`bbox.y1 > 620`)
- B starts near page top (`bbox.y0 < 180`)
- numeric anchor overlap >= 2
- at most one continuation page is absorbed

This avoids swallowing multiple unrelated pages into one giant region.

### Step 7 — Combine with parsed-table fallback

For each detected visual region:

- find overlapping parsed visual groups (`overlap > 0.25`)
- inherit section metadata from the best overlapping parsed table
- emit as `visual_region_XXX`

Then append any old parsed visual groups not already covered (`overlap > 0.72`).

If the detector throws, API falls back to parsed-table visual groups only.

---

## Detection Signal Priority

| Signal | Role |
| --- | --- |
| Repeated numeric column right edges across consecutive rows | **Primary** |
| Stable label left edge | Strong secondary |
| Stable row rhythm / small vertical gaps | Strong secondary |
| Multi-column numeric density | Secondary |
| Header/unit rows above table | Expansion helper |
| Cross-page bottom/top proximity + same anchors | Continuation helper |
| Horizontal/vertical lines, gray fills | Possible future bonus only |
| Subsection metadata | **Not** used for visual merge |
| Numeric density alone | Too weak by itself |
| `find_tables()` bbox alone | Too fragmented for UI crops |

The key insight: **column alignment consistency across consecutive rows is the most reliable table signal in 10-K PDFs**.

---

## Output Schema

### Single-page visual region

```json
{
  "table_id": "visual_region_031",
  "asset_type": "visual_table_region",
  "page_start": 64,
  "page_end": 64,
  "row_count": 16,
  "col_count": 5,
  "bbox": [0.0, 189.7, 612.0, 450.6],
  "anchors": [510, 570],
  "source_table_ids": ["visual_table_029"],
  "section_title": "Financial Statements and Supplementary Data",
  "header_path": ["Financial Statements and Supplementary Data", "..."]
}
```

### Cross-page parsed group

```json
{
  "table_id": "table_group_137",
  "asset_type": "table_group",
  "page_start": 63,
  "page_end": 64,
  "row_count": 4,
  "col_count": 9,
  "bbox_by_page": [
    {"page": 63, "bbox": [42.0, 717.75, 570.0, 729.0]},
    {"page": 64, "bbox": [42.0, 27.75, 570.0, 84.0]}
  ],
  "source_table_ids": ["table_138", "table_139", "table_140", "table_141"]
}
```

### API summary shape returned to UI

```json
{
  "table_id": "visual_region_031",
  "asset_type": "visual_table_region",
  "page_start": 64,
  "page_end": 64,
  "row_count": 16,
  "col_count": 5,
  "header_path": ["Financial Statements and Supplementary Data", "..."],
  "crops": [
    {"crop_idx": 0, "page": 64, "bbox": [0.0, 189.7, 612.0, 450.6]}
  ]
}
```

Assets API also reports:

```json
{
  "counts": {
    "visual_tables": 101,
    "visual_detector": "column_alignment"
  }
}
```

---

## Table Crop Rendering

Endpoint:

```text
GET /api/files/{file_id}/tables/{table_id}/crops/{crop_idx}.png
```

Rendering rules:

- use PyMuPDF pixmap clip from source PDF
- prefer readable page slices over tight detector boxes
- scale: `Matrix(2.5, 2.5)`
- PNG responses are no-cache

### Single-page / single-crop tables

```text
clip = full page width
y0 = bbox.top - 76/96pt padding
y1 = bbox.bottom + 76/96pt padding
```

### Multi-page tables

For `bbox_by_page` with multiple crops:

- first page crop: from `bbox.top - 220pt` to page bottom
- last page crop: from page top to `bbox.bottom + 180pt`
- middle pages: full page

This is why cross-page tables like page 63 bottom + page 64 top now show useful context instead of one-row slivers.

---

## UI Design Notes

Current UI goals:

- clean, non-flashy layout
- hierarchical chunk and asset trees
- show only the current tree label, not full repeated path strings
- chunk text expands inline in the tree leaf
- right column shows table crop previews, not HTML tables
- draggable column resizer for the preview pane
- activity log with stepper/progress during processing
- selecting a table updates preview without collapsing the asset tree

Table preview behavior:

- each crop is a large scrollable/drag-pan image region
- image URLs include cache-bust query param after reload
- no custom scroll slider bar; native/trackpad scrolling only

Q&A uses the dual-path inference pipeline documented in `0525_redo/CHUNKING_AND_INFERENCE_DESIGN.md`:

```text
text path:   top10 -> text rerank top3 -> expand preamble/neighbors/refs
table path:  summary top5 -> similarity threshold (default 0.75) -> VLM markdown
answer:      preamble + text + [Table: id | section] blocks
```

Model providers when `.env` is configured for the recommended split:

| Stage | Provider |
| --- | --- |
| Query embedding | Fireworks |
| Text rerank | Anthropic (default Haiku) |
| Answer | Anthropic (default Sonnet) |

Fireworks is not used for chat when `ANTHROPIC_API_KEY` is set.

Chunk Studio `/api/files/{id}/ask` parameters include:

- `vector_top_k` (default 10)
- `rerank_top_n` (default 3, text only)
- `table_vector_top_k` (default 5)
- `table_similarity_threshold` (default 0.75)

Requires workspace indexes:

- `{workspace}/index/vectors.db` — text chunks
- `{workspace}/index/table_vectors.db` — VLM table summaries

Vector indexing is built during Process when `FIREWORKS_API_KEY` is present. Table summary indexing is currently a separate build step after VLM parse.

---

## What Uses VLM Today

**Table region detection does not use VLM.** Detection is 100% local geometry over PyMuPDF words.

**Offline table parse does use VLM.** For selected tables in `assets.json`, Chunk Studio can render a crop PNG and call a VLM to produce:

- `vlm_parse.markdown`
- `vlm_parse.summary`

These fields power:

- the **Parses** inspector tab
- the table summary vector DB used at Q&A time

Scripts:

- `0525_redo/chunking/vlm_table_parse.py`
- `0525_redo/chunking/build_table_vector_db.py`

Recommended use of multimodal models by stage:

| Stage | Use VLM? |
| --- | --- |
| Detect table regions | No — too slow/expensive per page |
| Render table crops | No |
| Offline parse table crops to markdown | Yes — batch/offline |
| Retrieve candidate tables | No — embed `vlm_parse.summary`, not images |
| Answer complex table questions | Yes — inject VLM markdown; optional future step: pass crop image to multimodal answer model |
| Recover hierarchy/indent semantics | Yes — VLM markdown preserves layout better than flattened rows |

Recommended inference pattern:

```text
retrieval -> text top10 + table-summary top5 (parallel)
text       -> rerank top3 anchors + neighbor expansion
table      -> threshold filter -> VLM markdown
answer     -> preamble + text + [Table: id | section]
citation   -> chunk_id / table_id
```

---

## Known Limitations and Corner Cases

The current detector is much better than parser-first cropping, but it is still heuristic. The product criterion remains: **the screenshot must be correct**. The sections below document what merge logic exists today and which corner cases still fail.

### What merge exists today

There are three layers, not zero:

| Layer | Where | What it merges |
| --- | --- | --- |
| Parsed same-page merge | `section_asset_extractor.py` | One-row `find_tables()` fragments on the same page |
| Parsed cross-page merge | `section_asset_extractor.py` | Adjacent pages when table touches bottom/top and columns/width match |
| Visual region merge | `chunk_studio/server.py` | Column-aligned row runs; adjacent regions on same page; cross-page regions with matching numeric anchors |

These layers are **conservative**. They improve many 10-K tables but do not guarantee one logical table becomes one UI entry.

### How boundaries are found (recap)

Boundaries are **not** based on grid lines or VLM. They come from:

```text
words -> visual rows -> numeric column anchors -> consecutive aligned runs -> expanded bbox
```

A region is accepted when a run has:

- >= 3 aligned table-ish rows, or
- >= 2 rows with >= 3 numeric tokens total

Same-page region merge requires:

- vertical gap <= 24pt
- numeric anchor overlap >= 2

Cross-page visual region merge requires:

- previous region near page bottom (`y1 > 620`)
- next region near page top (`y0 < 180`)
- numeric anchor overlap >= 2
- at most one continuation page absorbed

This means the boundary is really a **block of numerically aligned rows**, not the full semantic extent of a financial statement table.

### Corner cases that still fail

#### 1. One logical table split into two UI entries

Still happens when:

- a blank line, prose sentence, or non-numeric subheader sits between two table blocks
- vertical gap between row runs exceeds ~42pt
- the second block has different numeric anchors (subtotal row, indent shift, column count change)
- two detected regions on the same page are > 24pt apart
- a new `visual_region_*` is emitted while an older `visual_table_*` / `table_group_*` fallback is also kept because coverage is incomplete

Symptom in UI: two adjacent table entries that visually belong to one statement.

#### 2. Cross-page header-only tail + full table body on next page

This is the weakest case today.

Example:

```text
page N bottom:
  "Designated as Hedging Instruments"
  "Foreign exchange contracts purchased"   (no values)

page N+1 top:
  full numeric rows for the same table
```

Why it fails:

- header/category rows often have **zero or one numeric anchor**
- they may not qualify as `is_tableish`
- even if detected, cross-page merge currently requires **anchor overlap >= 2**
- header-only tails cannot satisfy that rule
- parsed `find_tables()` may miss the header row entirely and only detect page N+1 body rows

Symptom in UI: page N header/context missing, or header and body appear as separate unrelated tables.

There is **no dedicated header-only continuation rule yet**.

#### 3. Category header at page bottom, data rows on next page

Related to case 2, common in derivative / disclosure tables:

```text
page N bottom:
  "Not Designated as Hedging Instruments"

page N+1 top:
  "Foreign exchange contracts purchased" 15,214 | 7,167
  ...
```

If the category header is the last thing on page N and the first data rows start on page N+1, merge depends on lucky fragment detection. The visual crop may show page-bottom context, but the table may still be represented as separate entries or an incomplete group.

#### 4. Tables with very few numeric columns

Mostly text columns, ratio tables, or label-heavy tables with sparse numbers:

- may never reach `is_tableish`
- may produce no visual region
- may survive only as tiny parsed fragments

#### 5. Numeric-heavy prose mistaken for tables

Pages with many aligned numbers outside a real table (footnotes, inline statistics, list-like prose) can occasionally produce false-positive regions.

#### 6. Multi-page tables beyond one continuation page

Cross-page merge is intentionally capped at **one continuation page** for visual regions. A table spanning 3+ pages may appear as multiple linked or unlinked regions.

#### 7. Hierarchy encoded by indentation

Parent/child row relationships, bold subtotals, gray bands, and wrapped labels are preserved in the **crop image**, but not recovered as structured hierarchy. Flat row/column metadata still loses semantic dependency.

#### 8. Section/subsection attachment noise

Underlying extractor may attach each financial row to a different subsection. Visual merge intentionally ignores subsection for cropping, but metadata `header_path` can still look wrong even when the screenshot is right.

#### 9. Duplicate or overlapping table entries

Because the API combines:

- new `visual_region_*` detector output, and
- uncovered fallback parsed groups

the table list can contain overlapping entries for nearby or identical content.

### Cross-page handling today

Cross-page is handled in two ways:

**Detection / grouping**

- parsed `table_group_*` via bottom/top geometry + column count
- visual region cross-page merge via bottom/top geometry + anchor overlap
- parsed-group tail merge that absorbs next-page continuation rows

**Rendering**

Even when bbox is tight, crops add context:

- first page: from `bbox.top - 220pt` to page bottom
- next page: from page top to `bbox.bottom + 180pt`

So cross-page **screenshots** can look reasonable even when **grouping** is incomplete. Grouping and cropping are not the same problem.

### Example: improved but not solved

The derivative-table case that motivated recent work:

- page 63 bottom: header + first rows
- page 64 top: continuation rows

Current status:

- better than one-row sliver crops
- `table_group_137` can merge some continuation rows
- still not guaranteed to unify header-only tails with full next-page bodies
- still possible to see split entries depending on anchors and detection luck

### Planned fixes for the above

Highest-value next rules:

1. **Header-only cross-page continuation** — partially implemented: cross-page merge now bridges when page tail/head contain no prose; continuation crops prepend `header_crop` from the first page.
2. **Same-page gap bridging** — implemented: adjacent regions merge when the gap contains only table-internal lines (subtotal, category subheader, column titles) or no text at all.
3. **Dedup in UI/API** — suppress fallback parsed groups when a visual region already covers >= 72% overlap.
4. **Persist visual regions at process time** — avoid recomputing slightly different results on each API call.

### Merge rules now implemented

**Same page**

Merge region A + region B when the vertical gap between them:

- has **no words at all**, or
- contains only **table-internal lines** such as subtotals, `Changes in Fair Value...`, `Total debt investments`, column header fragments, or
- still satisfies the older anchor-overlap heuristic for small gaps.

This fixes cases like **MSFT page 60**, where one fair-value table was previously split at an internal subtotal/subheader block.

**Cross page**

Merge when:

- previous region is near page bottom and next region is near page top, and
- page tail/head contain no prose, or anchors overlap, or the previous page ends with header-only lines and the next page starts with numeric rows.

**Header carry-over for parsing**

For `crop_idx > 0`, the PNG is rendered as:

```text
header band from first page/table top
+
continuation body crop
```

So downstream table-image parsing always sees column headers, even when the table body continues on a later page or later region.

---

## Validation Snapshot (MSFT FY2025)

On the MSFT workspace used during development:

| Metric | Value |
| --- | --- |
| Raw parsed tables | 97 |
| VLM-parsed tables (first MD&A batch) | 10 |
| Text chunks | 204 |
| Text vectors | 204 |
| Table summary vectors | 10 |
| Column-alignment visual tables served | 101 |

Table QA regression on 10 parsed-table questions:

| Pipeline | Answer correctness | Target table in context |
| --- | --- | --- |
| Old merged rerank | 8/10 | 6/10 |
| Dual-path + threshold 0.75 | 10/10 | 10/10 |

Eval artifacts:

- `0525_redo/inference/msft_fy2025_table_test_eval_v2.json`
- `0525_redo/inference/msft_fy2025_table_test_eval_v2.jsonl`

This is served live by `/api/files/{file_id}/assets` without re-processing for visual tables. VLM parse results live in `assets.json`.

---

## Agent Q&A (`/agent`)

Standalone page: **http://127.0.0.1:8010/agent** (also linked from Chunk Studio header and file cards).

LangChain agent (`0525_redo/agent/langchain_agent.py`) with tools:

| Tool | Role |
|------|------|
| `sql` | Structured numbers from `data/financials.db` (Text-to-SQL inside tool) |
| `rag` | Narrative/table evidence from **this workspace’s** `index/vectors.db` + `assets.json` |
| `send_email` | Optional SMTP delivery of the final answer |

### API

- Stream: `POST /api/files/{file_id}/agent/trace/stream` (NDJSON) via `agent_bridge.py`.
- Agent/memory pipeline: `0525_redo/agent/README.md`, `MEMORY_DESIGN.zh.md`.

### Prerequisites

- Filing `status=ready` and **`index/vectors.db` exists** (Process with **Build embeddings** checked).
- `.env`: `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY` (embeddings for RAG index at process time).

### Optional email (`.env`)

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=you@gmail.com
SMTP_USER=you@gmail.com
SMTP_PASSWORD=gmail_app_password
SMTP_USE_TLS=true
```

See `0525_redo/agent/README.md` for routing, empty SQL/RAG behavior, and Gmail App password notes.

### Routing note

Each agent step is chosen by the LLM from tool descriptions + prior observations (see `0525_redo/agent/README.md`).

---

## How To Run

From repo root:

```bash
cd agentic-rag-takehome-fw
uvicorn chunk_studio.server:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/          # Chunk Studio
http://127.0.0.1:8010/agent     # Agent Q&A
```

Upload a PDF, click **Process** (enable embeddings for Agent), then inspect chunks/assets/tables or open **Agent Q&A**.

Environment:

- `.env` at repo root: `FIREWORKS_API_KEY`, `ANTHROPIC_API_KEY`, optional SMTP vars for email tool
- optional: `ANTHROPIC_RERANK_MODEL`, `ANTHROPIC_CHAT_MODEL`
- default port: `8010`

---

## Recommended Next Steps

If continuing this product, the most valuable follow-ups are:

1. Move column-alignment detector out of `server.py` into `0525_redo/chunking/visual_table_detector.py`
2. Persist `visual_regions.json` during process instead of computing at API time
3. Attach visual regions to chunks by overlap/page/section instead of parsed fragment refs
4. Add a debug overlay mode in UI showing detected row anchors and region bboxes
5. Wire VLM parse + table summary indexing into the Process pipeline by default
6. Calibrate `table_similarity_threshold` on labeled table questions per filing (MSFT mixed eval suggests 0.65–0.70 may be safer than 0.75)
7. Keep parsed cell text as auxiliary metadata, not the primary table representation

The product success criterion should remain:

> **The screenshot of the table must be correct.**

Everything else is secondary.

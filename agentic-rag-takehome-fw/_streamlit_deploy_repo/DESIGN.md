# Design Overview

This project is a local 10-K research agent. The design separates offline document processing from runtime question answering so that the agent can answer with grounded evidence instead of relying on a single generic PDF-to-text pipeline.

## System Architecture

The system has three layers:

- Offline processing turns raw 10-K PDFs into section-aware chunks, table assets, and vector indexes.
- Runtime tools expose structured financial SQL and filing-grounded RAG as separate capabilities.
- The agent routes user questions to SQL, RAG, or a SQL-then-RAG chain and streams tool traces to the UI.

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
  SQL --> SYNTHESIS[Grounded final answer]
  RAG --> SYNTHESIS
  TEXT_INDEX --> RAG
  TABLE_INDEX --> RAG
  TABLES --> RAG
  SYNTHESIS --> UI[Agent UI + tool trace]
```

## Chunking Pipeline

Generic chunking is not enough for 10-Ks because Item headings repeat in the table of contents, tables can span pages, and a single Item section can contain many unrelated topics. The chunking pipeline preserves the filing hierarchy while keeping chunks small enough for precise retrieval.

```mermaid
flowchart TD
  PDF[10-K PDF] --> READ[Read text spans, words, fonts, positions]
  READ --> TOC_SCAN[Scan early pages for visible TOC]
  TOC_SCAN --> TOC[Parse TOC entries]
  TOC --> SECTION[Menu-guided body heading matching]
  SECTION --> ITEM[Build Part / Item sections]

  ITEM --> SUB[Detect subsection headings]
  SUB --> PREAMBLE[Split section preamble]
  SUB --> SUBCHUNKS[Build subsection text units]

  ITEM --> ASSET[Extract table / image assets]
  ASSET --> TABLE_PIPE[Table merge + metadata pipeline]
  TABLE_PIPE --> ASSET_JSON[assets.json]

  PREAMBLE --> CHUNK[Build clean text chunks]
  SUBCHUNKS --> CHUNK
  ASSET_JSON --> CHUNK
  CHUNK --> LINKS[Add refs and expansion links]
  LINKS --> TEXT_INDEX[Build text vector DB]

  ASSET_JSON --> VLM[Optional offline VLM table parse]
  VLM --> TABLE_SUM[Table markdown + summary]
  TABLE_SUM --> TABLE_INDEX[Build table summary vector DB]
```

Each text chunk carries:

- ticker and fiscal year
- section and subsection references
- header path
- page range
- table and image references
- neighboring chunk links for context expansion

## Section And Subsection Detection

Main Item sections are found by parsing the visible TOC and then matching expected Item headings in body order. This avoids accepting repeated `Item 7` references from the TOC, headers, footers, or narrative cross-references.

Subsections are detected with layout signals rather than regex alone:

- bold / italic
- font size
- x-position and indentation
- vertical gap before and after
- line length
- centered all-caps score
- filters for prose-like lines and page-number noise

```mermaid
flowchart TD
  A[Section text + PDF layout records] --> B[Filter records inside section]
  B --> C[Reject obvious noise]
  C --> D{Heading-like line?}
  D -- no --> SKIP[Skip as prose/noise]
  D -- yes --> E[Score visual/style signals]
  E --> F[bold / italic]
  E --> G[font size]
  E --> H[indentation]
  E --> I[vertical spacing]
  E --> J[centered all-caps]
  E --> K[prose filters]
  F --> DECIDE{Candidate accepted?}
  G --> DECIDE
  H --> DECIDE
  I --> DECIDE
  J --> DECIDE
  K --> DECIDE
  DECIDE -- no --> SKIP
  DECIDE -- yes --> LEVEL[Assign heading level]
  LEVEL --> STACK[Update heading stack]
  STACK --> PATH[Build subsection path]
  PATH --> OUT[subsection chunks + preamble]
```

## Text And Table Separation

Narrative text and financial tables are treated as separate evidence types:

- Text chunks remove flattened numeric table rows so embeddings are not dominated by row noise.
- Table markers remain in text chunks so runtime RAG can reconnect nearby tables.
- Tables are stored as assets and can be parsed into markdown plus summaries.
- Table summaries are embedded separately from narrative text.

```mermaid
flowchart TD
  A[Section + subsection text] --> B[Strip repeated headings]
  B --> C[Strip PDF noise / page markers]
  C --> D[Find flattened table starts]
  D --> E[Remove numeric table rows from text]
  E --> F[Insert TABLE markers]
  F --> G[Add header path to chunk text]
  G --> H[Sentence-aware split: about 100-500 tokens]
  H --> I[Attach metadata]
  I --> J[table_refs / table_anchors]
  I --> K[image_refs]
  I --> L[section_ref / subsection_ref]
  I --> M[inference expansion links]
  M --> N[same text-unit prev/next]
  M --> O[same section prev/next]
  M --> P[section preamble refs]
  I --> Q[Text vector index]
```

## RAG Pipeline

The RAG tool uses a dual retrieval path:

- Text retrieval combines vector search and BM25, then reranks text hits.
- Table retrieval searches embedded table summaries and applies a similarity threshold.
- Context assembly expands selected text chunks with preamble and neighbors.
- Table contexts are loaded from directly referenced tables and independently retrieved table hits.

```mermaid
flowchart TD
  A[RAG question + ticker + fiscal year] --> B[Apply metadata hard filter]
  B --> C[Embed retrieval query once]
  C --> D[Text vector search top K]
  C --> E[BM25 text search top K]
  C --> F[Table summary vector search top K]
  D --> G[Merge vector + BM25 text hits]
  E --> G
  G --> H[Text rerank]
  H --> I[Top text anchors]
  F --> J[Table similarity threshold]
  J --> K[Filtered table hits]
  I --> L[Expand context]
  L --> M[Section preamble]
  L --> N[Same text-unit neighbors]
  L --> O[Same section neighbors]
  I --> P[Collect table_refs from text chunks]
  K --> Q[Load VLM markdown tables]
  P --> Q
  Q --> R[Deduplicate table contexts]
  M --> S[Assemble final context]
  N --> S
  O --> S
  R --> S
  S --> T[Answer model]
  T --> U[Grounded answer + citations]
```

## Agent Routing

The agent is intentionally tool-driven:

- SQL is used for exact metrics, ratios, rankings, and year-over-year comparisons from structured financial data.
- RAG is used for what the filing says: strategy, risk, MD&A explanation, segment commentary, and table evidence.
- Hybrid questions use SQL first to establish numbers, then RAG to retrieve the filing explanation scoped to the relevant company and fiscal year.

```mermaid
flowchart TD
  U[User question] --> A[Agent receives prompt + tools]
  A --> R{Route question}
  R -- exact numbers / ratios / ranking --> SQL[SQL tool]
  R -- filing narrative / table evidence --> RAG[RAG tool]
  R -- hybrid question --> H[SQL first, then scoped RAG]
  SQL --> SQLOBS[SQL JSON observation]
  RAG --> RAGOBS[RAG JSON observation]
  H --> HOBS[Numeric + filing observations]
  SQLOBS --> D{Enough evidence?}
  RAGOBS --> D
  HOBS --> D
  D -- no --> RETRY[Retry / reformulate / call another tool]
  RETRY --> A
  D -- yes --> SYN[Synthesize grounded answer]
  SYN --> OUT[Final answer]
```

## Design Tradeoffs

- Heuristic layout detection is explainable and fast, but it is not a perfect visual reconstruction of the PDF.
- Small chunks improve retrieval precision, but they require metadata-driven context expansion at inference time.
- Separating text and table evidence improves retrieval quality, but requires explicit table references and table context loading.
- Tool routing reduces hallucination risk, but the agent needs clear tool policy and visible traces for debugging.

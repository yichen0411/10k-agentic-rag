# 10-K Agentic RAG 架构导图

> 用法：这些是可复制到支持 Mermaid 的 Markdown / slides 工具里的导图。每一块都对应 presentation 里的一个重点模块。

## 1. 全局系统导图

```mermaid
mindmap
  root((10-K Agentic RAG))
    Audience / Use Case
      Financial analyst
      Investment research
      10-K Q&A
      Cross-company comparison
    Offline Processing
      PDF parsing
      TOC-guided sectioning
      Subsection detection
      Table / image extraction
      Text chunking
      Vector indexes
    Runtime Tools
      SQL tool
        Structured financial DB
        Exact numbers
        Ratios / growth / rankings
      RAG tool
        Filing narrative
        Table evidence
        Text + table retrieval
      send_email
        Optional delivery action
    Agent Layer
      Tool routing
      SQL -> RAG dependency chain
      RAG decomposition
      Fallback / retry
      Final synthesis
    Memory
      Short-term chat history
      Long-term summary / semantic notes
      Persistent user preferences
    Evaluation
      Retrieval Hit@K
      Table QA regression
      RAGAS-style judge
      Tool-policy eval
      Corner-case benchmark
```

## 2. Offline Parsing 到 Chunking 导图

```mermaid
flowchart TD
  A[10-K PDF] --> B[Read visible text + layout records]
  B --> C[Detect TOC pages]
  C --> D[Parse visible TOC entries]
  D --> E[Menu-guided Item matching in body]
  E --> F[Build main sections: Part / Item]
  F --> G[Detect subsection headings]
  G --> H[Attach subsection chunks + section preamble]
  F --> I[Extract tables and images]
  I --> J[Attach assets to section / subsection]
  H --> K[Build clean text RAG chunks]
  J --> K
  K --> L[Add metadata: header_path, refs, neighbors]
  L --> M[Text vector DB]
  J --> N[Optional VLM table parse]
  N --> O[Table markdown + summary]
  O --> P[Table summary vector DB]
```

## 3. Section / Subsection Detection 导图

```mermaid
mindmap
  root((Section + Subsection Detection))
    Main Sectioning
      Problem
        Item headings repeat in TOC
        Item references appear in body
        PDF page numbers may not match printed pages
      Design
        Parse visible TOC
        Treat TOC as ordered semantic menu
        Search body in TOC order
        Build stable refs
          Part I::Item 1
          Part II::Item 7
      Tradeoff
        More robust than page matching
        Depends on usable visible TOC
    Subsection Detection
      Signals
        Bold
        Italic
        Font size
        X position / indentation
        Line length
        Sentence-ending filter
        Prose-start filter
        Vertical gaps
        Centered all-caps score
      Hierarchy
        Heading stack
        Level assignment
        Header path materialization
      Corner Cases
        All-caps centered headings
        Italic prose mistaken as heading
        TOC / page-number noise
        Long risk-factor titles
        Sections with no subsection
      Output
        section preamble
        subsection_chunks
        path
        char offsets
        page
```

## 4. Text Chunking 导图

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

## 5. Table Extraction / Merge / VLM 导图

```mermaid
flowchart TD
  A[PDF pages] --> B[PyMuPDF find_tables]
  B --> C[Raw table fragments]
  C --> D[Word-layer row rescan]
  D --> E[Recover missed subtotal / line-item rows]
  E --> F[Same-page fragment merge]
  F --> G[Collapse absorbed duplicate fragments]
  G --> H[Cross-page merge scoring]
  H --> I{Should merge?}
  I -- yes --> J[table_group with source_table_ids]
  I -- no --> K[Keep separate table]
  J --> L[Attach section / subsection metadata]
  K --> L
  L --> M[Render visual crop]
  M --> N[Offline VLM parse]
  N --> O[Markdown table]
  N --> P[One-sentence table summary]
  P --> Q[Table summary vector DB]
  O --> R[Used as table context in RAG]
```

## 6. Table Corner Cases 导图

```mermaid
mindmap
  root((Table Corner Cases))
    Same-page fragmentation
      Many one-line detections
      Merge by proximity / width / columns
      Drop absorbed fragments
    Cross-page continuation
      Previous table touches page bottom
      Next table starts near page top
      Compare columns / width / data rows
      Preserve merge_group_id
    Header-only bottom band
      Header on page N
      Data on page N+1
      Lower merge threshold
      Allow column mismatch
      Keep section from header page
    Wrong section assignment
      Page-top table may belong to previous page section
      Prefer previous section near page top
    Table rows missed by detector
      Rescan PDF word layer
      Expand bbox downward
      Recompute raw_rows and complexity
    Sparse / visual hierarchy
      Indentation carries meaning
      Bold / subtotal rows matter
      VLM markdown preserves more semantics
    Tradeoff
      Aggressive merge risks unrelated tables
      Conservative merge splits logical table
      VLM improves quality but adds cost
```

## 7. Indexing 导图

```mermaid
flowchart LR
  A[Text chunks] --> B[Embed with Fireworks]
  B --> C[(Text vectors.db)]
  C --> D[Stores content + metadata JSON]
  D --> E[ticker / fiscal_year / source_file]

  F[VLM table markdown + summary] --> G[Compose table summary text]
  G --> H[Embed with Fireworks]
  H --> I[(Table vectors.db)]
  I --> J[Stores table_id + section_ref + summary]

  K[Assets JSON] --> L[Table lookup]
  L --> M[Load markdown / raw_rows at runtime]
```

## 8. RAG Pipeline 导图

```mermaid
flowchart TD
  A[rag question + ticker + fiscal_year] --> B[Apply metadata hard filter]
  B --> C[Embed retrieval query once]
  C --> D[Text vector search top K]
  C --> E[BM25 text search top K]
  C --> F[Table summary vector search top K]
  D --> G[Merge vector + BM25 text hits]
  E --> G
  G --> H[Text rerank only]
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

## 9. RAG Fallback / Confidence 导图

```mermaid
flowchart TD
  A[Initial RAG answer / retrieval result] --> B{Looks insufficient?}
  B -- no --> C[Return success]
  B -- yes --> D[Rewrite retrieval query]
  D --> E[Use concrete 10-K terms]
  E --> F[Preserve original answer question]
  F --> G[Retry retrieval]
  G --> H{Better evidence?}
  H -- yes --> I[Return fallback_success]
  H -- no --> J[Return insufficient_context]

  K[Retrieval confidence] --> L[Text top rerank score]
  K --> M[Top table similarity]
  L --> N{Both weak?}
  M --> N
  N -- yes --> D
  N -- no --> C
```

## 10. SQL Tool 导图

```mermaid
flowchart TD
  A[sql natural-language question] --> B[Text-to-SQL LLM]
  B --> C{CANNOT_ANSWER?}
  C -- yes --> D[status: cannot_answer]
  C -- no --> E[Validate SQL]
  E --> F{Valid SELECT only?}
  F -- no --> G[status: error]
  F -- yes --> H[Read-only SQLite execution]
  H --> I{Execution succeeds?}
  I -- yes --> J{Rows?}
  J -- yes --> K[status: success + rows]
  J -- no --> L[status: empty_result]
  I -- no --> M[Correct SQL with error message]
  M --> N{Attempts < 3?}
  N -- yes --> E
  N -- no --> O[status: fallback]
```

## 11. SQL / RAG Routing 导图

```mermaid
mindmap
  root((Routing Policy))
    SQL
      Exact financial numbers
      Margins / ratios
      Growth rates
      Rankings
      Segment revenue
      Geographic revenue
      Balance sheet metrics
    RAG
      10-K narrative
      Management commentary
      Risks
      Strategy / competition
      MD&A drivers
      Filing table evidence
    SQL then RAG
      Identify entity with SQL
      Pass ticker/year/entity to RAG
      Retrieve management explanation
      Synthesize numeric + narrative answer
    RAG decomposition
      One ticker
      One fiscal year
      One factual intention
      Compare after separate retrievals
    Stop / answer
      Existing observations sufficient
      User asks to summarize previous answer
      User asks style-only follow-up
```

## 12. Agent Loop 导图

```mermaid
flowchart TD
  A[User question] --> B[Build memory context]
  B --> C[System prompt + tool catalog + chat history]
  C --> D[LangChain tool-calling agent]
  D --> E{Need tool?}
  E -- SQL --> F[Call sql tool]
  E -- RAG --> G[Call rag tool]
  E -- Email --> H[Call send_email]
  E -- No --> I[Final answer]
  F --> J[JSON observation]
  G --> J
  H --> J
  J --> K[Agent scratchpad]
  K --> D
  I --> L[Append turn to memory]
  L --> M[Return answer + trace]
```

## 13. Memory 导图

```mermaid
mindmap
  root((Session Memory))
    Short-term
      messages table
      recent user / assistant turns
      injected as chat_history
      resolves "that" / "above"
    Long-term summary
      old messages folded into sessions.summary
      capped length
      no LLM summarization currently
    Semantic notes
      Q/A snippets
      keyword-overlap retrieval
      max note count
      related follow-ups only
    Persistent facts
      memory_facts table
      user_email
      response_style
      output_language
      preferred_units
      explicit preferences
    Tool artifacts
      stored for debugging
      not normally injected
    Boundary
      memory is not filing evidence
      SQL remains source for numbers
      RAG remains source for filing claims
```

## 14. User Question 到 Answer 全链路导图

```mermaid
flowchart TD
  A[User asks question] --> B[Load session memory]
  B --> C[Agent receives prompt]
  C --> D{Question type?}

  D -- Structured numbers --> E[sql tool]
  E --> F[Validated read-only SQL]
  F --> G[Rows / status JSON]

  D -- Filing narrative / table --> H[rag tool]
  H --> I[Scope filter ticker/year]
  I --> J[Text + BM25 + table retrieval]
  J --> K[Text rerank + table threshold]
  K --> L[Context expansion + table injection]
  L --> M[RAG answer JSON]

  D -- Hybrid --> N[SQL first]
  N --> O[Use SQL result entity/year/metric]
  O --> P[RAG follow-up with concrete scope]
  P --> Q[Synthesize numeric + filing evidence]

  G --> R[Agent decides next step]
  M --> R
  Q --> R
  R --> S{Enough evidence?}
  S -- no --> T[Retry / reformulate / call another tool]
  T --> R
  S -- yes --> U[Final grounded answer]
  U --> V[Append memory]
  V --> W[Optional send_email if requested]
```

## 15. Evaluation 导图

```mermaid
mindmap
  root((Evaluation))
    Offline parsing eval
      Section sanity checks
      Subsection counts
      Heading hierarchy inspection
      Table layout hard cases
    Retrieval eval
      Generated chunk-grounded questions
      Hit@1 / Hit@3 / Hit@5 / Hit@10
      Cross-chunk recall
    Table QA eval
      Expected table in top K
      Threshold pass
      Final context hit
      Numeric correctness
      Failure diagnosis
    Answer eval
      Faithfulness
      Answer relevancy
      Context precision
      Reference coverage
      Latency
    Agent eval
      Next-step tool policy
      SQL vs RAG choice
      SQL -> RAG dependency
      RAG decomposition
      Fallback after insufficient evidence
    Future eval
      More hand-labeled corner cases
      Unsupported-but-plausible questions
      Visual table labels
      Full trajectory regression
```

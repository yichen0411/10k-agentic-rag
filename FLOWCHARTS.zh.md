# 10-K Agentic RAG 流程图

> 这份文件专门放可以复制到 slides 的 Mermaid flowchart。和 `ARCHITECTURE_MAPS.zh.md` 的区别是：这里尽量只用流程图，不用 mindmap。

## 1. End-to-End 全链路

```mermaid
flowchart TD
  U[User question] --> M[Load session memory]
  M --> A[Agent receives prompt + tools + memory]
  A --> R{Route question}

  R -- exact numbers / ratios / ranking --> SQL[SQL tool]
  R -- filing narrative / table evidence --> RAG[RAG tool]
  R -- hybrid question --> H[SQL first, then scoped RAG]
  R -- style follow-up --> S[Answer from chat history / memory]

  SQL --> SQLOBS[SQL JSON observation]
  RAG --> RAGOBS[RAG JSON observation]
  H --> HOBS[Numeric + filing observations]

  SQLOBS --> D{Enough evidence?}
  RAGOBS --> D
  HOBS --> D
  S --> OUT[Final answer]

  D -- no --> RE[Retry / reformulate / call another tool]
  RE --> A
  D -- yes --> SYN[Synthesize grounded answer]
  SYN --> OUT
  OUT --> MEM[Append turn to memory]
  MEM --> EMAIL{User asked to email?}
  EMAIL -- yes --> SEND[send_email tool]
  EMAIL -- no --> DONE[Done]
  SEND --> DONE
```

## 2. Offline PDF Processing 到 Index

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

## 3. TOC-Guided Sectioning

```mermaid
flowchart TD
  A[PDF text records] --> B[Collect early pages]
  B --> C{Looks like TOC page?}
  C -- no --> B
  C -- yes --> D[Extract Item entries from TOC]
  D --> E[Normalize heading text]
  E --> F[Create ordered expected Item menu]
  F --> G[Search body text in menu order]
  G --> H{Candidate matches expected next Item?}
  H -- no --> I[Ignore repeated / cross-reference Item mention]
  H -- yes --> J[Accept as section boundary]
  I --> G
  J --> K[Cut section until next accepted boundary]
  K --> L[Assign stable section_ref_id]
  L --> M[Part / Item hierarchy]
```

## 4. Subsection Detection

```mermaid
flowchart TD
  A[Section text + PDF layout records] --> B[Filter records inside section]
  B --> C[Reject obvious noise]
  C --> D{Heading-like line?}

  D -- no --> SKIP[Skip as prose/noise]
  D -- yes --> E[Score visual/style signals]

  E --> F[bold / italic]
  E --> G[font size]
  E --> H[x-position / indentation]
  E --> I[line length]
  E --> J[gap before / after]
  E --> K[centered all-caps score]
  E --> L[prose-start and sentence-ending filters]

  F --> DECIDE{Candidate accepted?}
  G --> DECIDE
  H --> DECIDE
  I --> DECIDE
  J --> DECIDE
  K --> DECIDE
  L --> DECIDE

  DECIDE -- no --> SKIP
  DECIDE -- yes --> LEVEL[Assign heading level]
  LEVEL --> STACK[Update heading stack]
  STACK --> PATH[Build subsection path]
  PATH --> CUT[Cut text until next subsection]
  CUT --> OUT[subsection_chunks + preamble_text]
```

## 5. Text Chunk Builder

```mermaid
flowchart TD
  A[Section preamble + subsection text units] --> B[Strip repeated headings]
  B --> C[Strip PDF noise]
  C --> D[Find table-related text starts]
  D --> E{Contains flattened table rows?}
  E -- yes --> F[Strip table rows from text]
  E -- no --> G[Keep narrative text]
  F --> H[Insert TABLE marker / table anchor]
  G --> I[Add header path prefix]
  H --> I
  I --> J[Sentence-aware split]
  J --> K[Create text chunk]
  K --> L[Attach metadata]
  L --> L1[section_ref_id]
  L --> L2[subsection_ref_id]
  L --> L3[header_path]
  L --> L4[table_refs / image_refs]
  L --> L5[split_index / split_count]
  L --> M[Annotate inference links]
  M --> N[same_text_unit prev/next]
  M --> O[same_section prev/next]
  M --> P[section_preamble_refs]
  M --> Q[Ready for embedding]
```

## 6. Table Extraction and Merge

```mermaid
flowchart TD
  PDF[PDF pages] --> RAW[PyMuPDF find_tables]
  RAW --> FRAG[Raw table fragments]
  FRAG --> RESCAN[Rescan word layer around bbox]
  RESCAN --> ROWS[Recover missed rows and expand bbox]
  ROWS --> SAME[Merge same-page fragments]
  SAME --> DEDUP[Collapse absorbed duplicate fragments]
  DEDUP --> HEADER[Detect header-only / synthetic header bands]
  HEADER --> CONT[Supplement page-top continuations]
  CONT --> CROSS[Score cross-page merge candidates]

  CROSS --> CHECK{Merge candidate?}
  CHECK -- yes --> MERGED[Create table_group]
  CHECK -- no --> SINGLE[Keep separate table]

  MERGED --> META[Attach section/subsection refs]
  SINGLE --> META
  META --> ANCHOR[Annotate text anchors]
  ANCHOR --> ASSETS[Write assets.json]
```

## 7. Header-Only Cross-Page Table Case

```mermaid
flowchart TD
  A[Page N bottom has header band only] --> B[Mark header_only / pending_merge]
  B --> C[Page N+1 top has data rows]
  C --> D[Check geometry]
  D --> D1[Prev touches page bottom]
  D --> D2[Next starts near page top]
  D1 --> E[Lower merge threshold]
  D2 --> E
  E --> F[Allow column mismatch]
  F --> G[Inherit section/subsection from header page]
  G --> H[Merge into table_group]
  H --> I[Use group id and source_table_ids]
  I --> J[VLM crop sees header + data context]
```

## 8. VLM Table Parse and Table Index

```mermaid
flowchart TD
  A[assets.json tables] --> B[Select tables for parse]
  B --> C[Render table crop PNG]
  C --> D[Collect nearby narrative context]
  D --> E[Build VLM prompt]
  E --> F[VLM parses image]
  F --> G[Markdown table]
  F --> H[One-sentence summary]
  G --> I[Store vlm_parse in assets.json]
  H --> I
  I --> J[Compose table summary embedding text]
  J --> K[Embed summary]
  K --> L[table_vectors.db]
  L --> M[Runtime table summary retrieval]
```

## 9. RAG Runtime

```mermaid
flowchart TD
  Q[Question + ticker + fiscal_year] --> FILTER[Apply metadata hard filter]
  FILTER --> EMBED[Embed retrieval query once]
  EMBED --> TVEC[Text vector search]
  EMBED --> BM25[BM25 search]
  EMBED --> TBLVEC[Table summary vector search]

  TVEC --> MERGE[Merge dense + sparse text hits]
  BM25 --> MERGE
  MERGE --> RERANK[Text-only rerank]
  RERANK --> ANCHOR[Top text anchor chunks]

  TBLVEC --> THRESH[Filter by table similarity threshold]
  THRESH --> TBLHITS[Filtered table hits]

  ANCHOR --> EXPAND[Expand text context]
  EXPAND --> PRE[Section preamble]
  EXPAND --> NEIGH[Same text-unit / same-section neighbors]
  EXPAND --> REFS[Collect table_refs]

  TBLHITS --> LOADT[Load table markdown]
  REFS --> LOADT
  LOADT --> DEDUP[Deduplicate table contexts]

  PRE --> CTX[Assemble context]
  NEIGH --> CTX
  DEDUP --> CTX
  CTX --> ANSWER[Answer model]
  ANSWER --> OBS[RAG observation JSON]
```

## 10. RAG Fallback

```mermaid
flowchart TD
  A[Initial retrieval + answer] --> B[Compute retrieval confidence]
  B --> C[Text top rerank score]
  B --> D[Top table similarity]
  C --> E{Both text and table weak?}
  D --> E

  E -- no --> F{Answer looks insufficient?}
  E -- yes --> G[Rewrite retrieval query]
  F -- no --> OK[status: success]
  F -- yes --> G

  G --> H[Use concrete filing terms]
  H --> I[Preserve original answer question]
  I --> J[Retry retrieval pipeline]
  J --> K{Retry improves evidence?}
  K -- yes --> L[status: fallback_success]
  K -- no --> M[status: insufficient_context]
```

## 11. SQL Tool

```mermaid
flowchart TD
  A[Natural-language SQL question] --> B[LLM generates SQLite SQL]
  B --> C{CANNOT_ANSWER?}
  C -- yes --> CA[Return cannot_answer]
  C -- no --> V[Validate SQL]

  V --> V1[Only SELECT]
  V --> V2[No forbidden keywords]
  V --> V3[No comments / multi-statements]
  V --> V4[Only allowed tables]

  V1 --> OKV{Validation OK?}
  V2 --> OKV
  V3 --> OKV
  V4 --> OKV

  OKV -- no --> ERR[Return error]
  OKV -- yes --> EXE[Read-only SQLite execution]
  EXE --> SUCCESS{Execution succeeds?}

  SUCCESS -- yes --> ROWS{Rows returned?}
  ROWS -- yes --> OUT[Return success + rows]
  ROWS -- no --> EMPTY[Return empty_result]

  SUCCESS -- no --> CORR[Ask LLM to correct SQL]
  CORR --> RETRY{Retry attempts left?}
  RETRY -- yes --> V
  RETRY -- no --> FALL[Return fallback]
```

## 12. Agent Routing

```mermaid
flowchart TD
  A[Agent sees user question + memory] --> B{Needs exact structured numbers?}
  B -- yes --> SQL[Call SQL]
  B -- no --> C{Needs filing narrative/table evidence?}
  C -- yes --> RAG[Call RAG]
  C -- no --> D{Follow-up / style change?}
  D -- yes --> HIST[Use chat_history / memory]
  D -- no --> LIMIT[Explain limitation]

  SQL --> E{Need filing explanation too?}
  E -- yes --> SCOPE[Extract ticker/year/entity/metric from SQL rows]
  SCOPE --> RAG2[Call scoped single-intent RAG]
  E -- no --> SYN[Synthesize answer]

  RAG --> F{RAG says insufficient?}
  F -- yes --> RETRY[Retry with concrete filing terms]
  F -- no --> SYN
  RAG2 --> SYN
  RETRY --> SYN
  HIST --> SYN
  LIMIT --> SYN
```

## 13. Memory Update

```mermaid
flowchart TD
  A[New user + assistant turn] --> B[Append messages]
  A --> C[Extract episodic facts]
  C --> C1[user_email]
  C --> C2[response_style]
  C --> C3[output_language]
  C --> C4[preferred_units]
  A --> D[Write semantic Q/A note]
  A --> E[Store tool artifacts for debugging]

  B --> F{Short-term window exceeded?}
  F -- no --> G[Keep recent messages verbatim]
  F -- yes --> H[Fold old messages into session summary]
  H --> I[Cap summary length]
  D --> J[Cap semantic notes]
  E --> K[Cap tool artifacts]

  G --> L[Next turn memory context]
  I --> L
  J --> L
  C1 --> L
  C2 --> L
  C3 --> L
  C4 --> L
```

## 14. Evaluation Pipeline

```mermaid
flowchart TD
  A[Build / collect eval cases] --> B{Eval layer}

  B -- Text retrieval --> C[Chunk-grounded questions]
  C --> C1[Measure Hit@1 / Hit@3 / Hit@5 / Hit@10]

  B -- Table QA --> D[Expected table questions]
  D --> D1[Check table vector hit]
  D --> D2[Check threshold pass]
  D --> D3[Check final context hit]
  D --> D4[Check numeric correctness]

  B -- RAG answer --> E[RAGAS-style judge]
  E --> E1[Faithfulness]
  E --> E2[Answer relevancy]
  E --> E3[Context precision]
  E --> E4[Reference coverage]

  B -- Agent policy --> F[Next-step tool-policy cases]
  F --> F1[SQL vs RAG]
  F --> F2[SQL -> RAG dependency]
  F --> F3[RAG decomposition]
  F --> F4[Fallback behavior]

  C1 --> G[Analyze failures]
  D4 --> G
  E4 --> G
  F4 --> G
  G --> H[Add corner case regression]
  H --> A
```


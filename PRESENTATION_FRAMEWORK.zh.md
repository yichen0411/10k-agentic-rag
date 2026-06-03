# 10-K Agentic RAG Presentation Framework

> 目标：这不是演讲稿，而是 25 分钟 presentation 的结构化材料框架。每个 bullet 都可以转成 slide 或 speaker notes。

## 0. Presentation 目标与听众

- **Project audience / customer**
  - 主要用户：需要快速研究 10-K 的金融分析师、投资研究人员、企业研究人员。
  - 典型场景：
    - 想问“Apple Services 增长由什么驱动？”
    - 想比较 Apple / Microsoft / Alphabet 的财务表现。
    - 想从 10-K 表格里找具体数值，而不是人工翻 PDF。
    - 想把检索结果、财务数字和管理层叙述结合成一个 grounded answer。
  - 面试场景下的客户：take-home evaluator / interviewer，关注系统是否能把 PDF parsing、RAG、agent workflow 和 evaluation 做成完整闭环。

- **Project team and my role**
  - Solo project / take-home implementation。
  - 我负责 end-to-end：
    - Offline PDF parsing and chunking。
    - Table extraction and visual table handling。
    - Text and table RAG pipeline。
    - SQL tool and text-to-SQL guardrails。
    - Agent routing / workflow orchestration。
    - Session memory design。
    - Eval set construction, corner-case testing, metrics, and iteration。
    - Chunk Studio UI and `/agent` Q&A integration。

- **Business problem / objectives**
  - 10-K 信息密度高、PDF 结构复杂、表格多、跨页多，人工查找和验证很慢。
  - 普通 RAG 的问题：
    - PDF chunking 容易丢章节语义。
    - 表格直接扁平化进文本 chunk 会污染 embedding。
    - 只做 text retrieval 很难稳定回答表格数值问题。
    - Agent 如果没有明确 routing policy，容易把 SQL 和 filing evidence 混用。
  - 项目目标：
    - 让用户用自然语言问 Apple / Microsoft / Alphabet 的财务和 filing 问题。
    - 对结构化数字使用 SQL，对 10-K 叙述和表格证据使用 RAG。
    - 保持答案 grounded：能区分“SQL calculated”和“filing disclosed”。
    - 在技术上解决 10-K 的章节、subsection、表格、跨页和多轮 follow-up。

### 10-K 用例与瓶颈（Use cases & bottleneck）

- **设计原则**
  - 单意图、单数据源的问题可以稳定支持。
  - 需要「先 SQL 再 RAG」「多工具链式推理」「跨文档运维手册式串联」的复杂问题，当前架构**不能可靠支持**。
  - 瓶颈不是「答不出来」，而是：**路由与证据链路过长、歧义大，容易在错误工具之间跳转或编造中间步骤**。

- **✅ 支持：仅 SQL（结构化库）**
  - 问题特征：答案只依赖 `financials.db` 里的结构化字段（收入、分部、地理收入、资产负债表科目等），不需要读 10-K 正文或运维手册叙述。
  - 10-K 示例（与演示对齐）：
    - 「FY2025 哪家公司的总收入最高？」
    - 「Apple FY2025 Services 分部收入是多少？」
    - 「Compare Apple and Microsoft FY2025 revenue growth.」
  - 类比（非 10-K 域，说明同一模式）：
    - 「When was the last time the equipment got repaired?」→ 仅查维修记录表，不查手册。

- **✅ 支持：仅 RAG（单份 filing 证据）**
  - 问题特征：答案来自**一份** 10-K 内的叙述、风险因素、MD&A、表格摘要；`ticker` + `fiscal_year` 明确，且**一次 rag 调用只问一个意图**。
  - 10-K 示例：
    - 「What risk factors related to AI did Salesforce disclose in FY2025?」
    - 「What did Microsoft say in FY2025 MD&A about cloud demand?」
    - 「Summarize Alphabet FY2025 risk factors related to competition.」
  - 类比：
    - 「What should I do if equipment xxx is overheating?」→ 只查单份维护手册/规程，不查维修工单表。

- **❌ 不支持：复杂 / 多步链式问题**
  - 问题特征：需要**多个异构数据源**或**多个意图串联**，且中间步骤没有单独、可验证的 ground truth。
  - 典型失败模式：`sql` → `sql` → `rag`，或「先算趋势/对比，再查是否做过某运维动作，再从指南里推荐下一步」。
  - 10-K 示例（应明确拒绝或拆成多轮，而不是硬答）：
    - 「For the segment with the worst operating margin trend, check whether it has been cleaned recently and recommend the next maintenance step from the guide.」
    - 「Which segment grew slowest, and what did management say about it, and cite the filing?」
  - 类比（用户给的反例）：
    - 「For the heat exchanger showing the worst pressure-drop trend, check whether it has been cleaned recently and recommend the next maintenance step from the guide.」
      - 隐含：`sql`（趋势/最差）→ `sql`（是否清洗）→ `rag`（手册推荐）——**超出当前单步工具契约**。
  - **演示 / 面试时怎么说**
    - 先展示两个 ✅，再展示一个 ❌，强调：**不是模型不会答，而是产品边界是「单 SQL / 单 RAG」**。
    - 若必须演示复杂题：说明应拆成多轮对话，或拆成多个单意图子问题，而不是指望一次 agent run 完成全链。

- **Timeframe / project plan**
  - Phase 1: Offline parsing and chunking
    - TOC-guided sectioning。
    - Font / layout based subsection detection。
    - Table / image extraction。
    - Text-only RAG chunks with metadata。
  - Phase 2: RAG inference
    - Text vector index。
    - Table VLM parsing and table summary index。
    - Dual-path text + table retrieval。
    - Rerank, context expansion, answer generation。
  - Phase 3: Tool layer
    - Build reliable `sql` tool.
    - Build reliable `rag` tool.
    - Define structured inputs and stable JSON outputs。
    - Add fallback and logging。
  - Phase 4: Agent workflow
    - First version: AI workflow over existing SQL/RAG tools。
    - Later version: LangChain tool-calling agent。
    - Add routing policy, decomposition rules, memory, trace streaming。
  - Phase 5: Evaluation and iteration
    - Retrieval hit-rate eval。
    - Table-specific eval。
    - RAGAS-style answer eval。
    - Agent tool-policy eval。
    - Corner-case benchmark for table layout。

## 1. System Overview

- **High-level architecture**
  - Offline:
    - 10-K PDF -> section detection -> subsection detection -> table/image extraction -> text chunks -> embeddings。
    - Optional table path: table crop -> VLM markdown + summary -> table summary embeddings。
  - Runtime:
    - SQL tool answers structured financial database questions。
    - RAG tool answers filing text/table questions。
    - Agent decides whether to call SQL, RAG, both, or email。
    - Memory provides short-term conversation context and longer-term user/session facts。

- **Three major technical blocks for presentation**
  - Block A: Offline parsing, chunking, table processing。
  - Block B: SQL + RAG tools and routing workflow。
  - Block C: Agent, memory, observability, and evaluation。

- **Core design principle**
  - Treat narrative text and financial tables differently.
  - Text is best retrieved as section-aware semantic chunks.
  - Tables are best handled as separate evidence objects with image/VLM-derived markdown and summaries.
  - Agent workflow should not invent evidence; it should choose the right tool and synthesize only from tool observations.

## 2. Block A: Offline Parsing, Chunking, and Table Handling

### 2.1 Offline Parsing Goal

- Convert noisy 10-K PDFs into retrieval-ready evidence.
- Preserve enough structure for grounding:
  - company / ticker。
  - fiscal year。
  - Part / Item section。
  - subsection path。
  - page range。
  - table references。
  - image references。
  - neighboring chunk links。

- Avoid making every chunk too large.
  - Large chunks keep context but hurt retrieval precision。
  - Small chunks retrieve precisely but need metadata-driven expansion。
  - Final design uses small/medium text chunks plus inference-time expansion。

### 2.2 Section Detection

- **Problem**
  - SEC PDFs contain repeated `Item 1A`, `Item 7`, etc. in:
    - table of contents。
    - headers / footers。
    - cross references。
    - narrative paragraphs。
  - Printed page numbers and PDF page indexes may not match。

- **Approach**
  - Parse visible TOC from early pages。
  - Treat TOC as an ordered semantic menu, not as authoritative page coordinates。
  - Search body text in TOC order to locate real Item headings。
  - Build stable section refs such as:
    - `Part I::Item 1`
    - `Part I::Item 1A`
    - `Part II::Item 7`
    - `Part II::Item 8`

- **Tradeoff**
  - More robust than page-number matching。
  - Still assumes the visible TOC is usable。
  - If a filing has unusual TOC formatting, section detection may need custom handling。

### 2.3 Subsection Detection

- **Why subsection matters**
  - 10-K Item sections are too broad for precise RAG。
  - `Item 7` can contain many different topics: revenue, margins, liquidity, segment performance, risk commentary。
  - Subsection paths make chunks self-describing and improve retrieval。

- **Approach**
  - After main Item sections are located, scan PDF text records inside each section。
  - Use layout/font features to identify subsection heading candidates:
    - bold。
    - italic。
    - font size。
    - x-position / left indentation。
    - line length。
    - whether line ends like a sentence。
    - whether line starts with prose words like “the”, “we”, “our”。
    - vertical gap before / after。
    - centered all-caps heading score。
  - Assign hierarchy levels:
    - Note headings and larger headings as higher-level nodes。
    - bold / centered headings as middle-level nodes。
    - italic headings as lower-level nodes。
  - Maintain a heading stack to build paths:
    - `Management's Discussion and Analysis > Revenue`
    - `Risk Factors > Macroeconomic and industry risks`
    - `Financial Statements > Note 4 - Financial Instruments`

- **Corner cases handled**
  - **All-caps centered headings**
    - Some 10-K headings are not bold, but are visually centered and all caps。
    - Added `centered_heading_score` using center alignment, uppercase ratio, and vertical spacing。
  - **Italic prose vs italic heading**
    - Italic often marks subheadings, but can also be explanatory prose。
    - Filter by length, sentence ending, numeric prefix, and bad starts like “in”, “and”, “the”。
  - **TOC / Item / page number noise**
    - Exclude Item headings, Part headings, table headings, page numbers, date-only lines。
  - **Long risk-factor titles**
    - Allow moderately long heading-like text but reject sentence-like lines with verbs。
  - **Sections with no detected subsection**
    - Keep full Item body as section preamble so content is not lost。

- **Tradeoff**
  - Heuristic heading detection is explainable and fast。
  - It is not a perfect visual reconstruction of the PDF。
  - The objective is retrieval usefulness, not exact document layout fidelity。
  - Some subtle headings may be missed; some low-value list lines may become chunks。

- **What did not fully work**
  - Relying only on regex for `Item` headings was too brittle。
  - Relying only on font bold/italic missed centered all-caps headings。
  - Treating visual hierarchy as semantic hierarchy is imperfect, especially in financial notes。

- **Future improvements**
  - Add a labeled heading/subsection eval set。
  - Add visual debugging overlay for heading candidates。
  - Improve handling of financial statement notes and subtle subheadings。
  - Potentially combine heuristics with a lightweight layout model, but keep deterministic guardrails。

### 2.4 Text Chunking

- **Goal**
  - Build clean text chunks for semantic retrieval without polluting them with flattened table rows。

- **Approach**
  - Each chunk includes:
    - source file。
    - section ref。
    - subsection ref。
    - header path。
    - text。
    - token count。
    - table refs。
    - image refs。
    - split index。
    - inference expansion metadata。
  - Chunk text starts with header context so short chunks like “Services” are not ambiguous。
  - Target roughly 100-500 tokens per chunk。
  - Remove:
    - repeated headings。
    - PDF noise。
    - page markers。
    - flattened numeric table rows。
    - known table headers。
  - Insert `[[TABLE:...]]` markers where table content was stripped so the RAG layer can reconnect table evidence later。

- **Context expansion metadata**
  - Same text unit previous/next chunk。
  - Same section previous/next chunk。
  - Section preamble refs。
  - Cross-section expansion disabled by default。

- **Tradeoff**
  - Keeping chunks small improves retrieval precision。
  - Removing table rows improves text embeddings。
  - But answer generation sometimes needs neighboring context or table values。
  - Solved by adding inference-time expansion and table injection instead of making chunks huge。

### 2.5 Table Extraction and Processing

- **Why tables are the hardest part**
  - 10-K tables are not real structured tables in PDF; they are positioned words, lines, and rectangles。
  - Many tables have no grid。
  - Semantic hierarchy is encoded visually:
    - indentation。
    - bold subtotal rows。
    - blank spacing。
    - repeated headers。
    - page breaks。
  - Tables often span pages, and PyMuPDF can split them into fragments。

- **Layered approach**
  - Layer 1: PyMuPDF `find_tables()` to get initial table fragments。
  - Layer 2: Word-layer rescanning to recover rows missed by `find_tables()`。
  - Layer 3: Same-page fragment merge。
  - Layer 4: Cross-page connected table merge。
  - Layer 5: Synthetic header-band / continuation handling。
  - Layer 6: Offline VLM parse of table crop into markdown and summary。
  - Layer 7: Table summary vector index for table-specific retrieval。

- **What is stored for each table**
  - table id。
  - page start/end。
  - bbox or bbox by page。
  - raw rows / raw text。
  - section/subsection refs。
  - source table ids if merged。
  - complexity metadata。
  - VLM parse:
    - markdown。
    - summary。
    - parse status。

- **Corner cases handled**
  - **Same-page fragments**
    - Multiple one-line detections can belong to one logical table。
    - Merge based on page, vertical proximity, width/columns, and table-like rows。
  - **Cross-page continuation**
    - Merge if previous table touches page bottom and next table starts near page top。
    - Score by bottom/top geometry, column count, width, data rows, and continuation flags。
  - **Header-only page bottom + data on next page**
    - Detect header-only tables。
    - Lower merge threshold for header-only cases。
    - Allow header/data column mismatch。
    - Preserve section/subsection from header page rather than incorrectly assigning to next section。
  - **Page-top continuation assigned to wrong section**
    - Prefer previous page section when a table starts very near page top。
  - **Rows missed by PyMuPDF**
    - Rescan word layer below the detected bbox to recover subtotal/line-item rows。
  - **Duplicate / absorbed fragments**
    - Drop fragments fully contained inside a larger rescanned table。
  - **Visual table cropping**
    - Chunk Studio focuses on correct screenshot / crop, because visual evidence is often more reliable than flattened cells。

- **Tradeoffs**
  - Deterministic heuristics are fast, explainable, and cheap。
  - They require many corner-case rules。
  - Aggressive merge can incorrectly combine unrelated financial statements。
  - Conservative merge can split one logical table into multiple entries。
  - VLM parse improves table QA but adds cost and offline processing complexity。

- **What did not fully work**
  - Directly embedding raw table rows inside text chunks caused noisy retrieval。
  - A single combined rerank pool for text chunks and table summaries did not work well:
    - table summaries could be retrieved in top 5 but dropped by text reranking。
    - old table pipeline had target table in final context only `6/10` in one MSFT table test。
  - PyMuPDF alone is insufficient for 10-K hard cases:
    - table layout benchmark showed baseline custom pipeline score `0.95` vs PyMuPDF `0.662` and PyMuPDF4LLM `0.513` on selected hard cases。

- **Future improvements**
  - Persist visual regions at processing time rather than recomputing at API time。
  - Calibrate table similarity threshold per filing / per company。
  - Parse tables into structured row/value records for deterministic arithmetic。
  - Improve table-to-chunk linking near page breaks。
  - Add table-level labeled eval set covering:
    - cross-page tables。
    - header-only continuations。
    - sparse numeric tables。
    - tables with indentation hierarchy。
    - derivative / fair value / revenue schedules。
  - For complex table questions, optionally pass table crop image to a multimodal answer model。

## 3. Block B: SQL, RAG, and Routing Workflow

### 3.1 Why Split SQL and RAG

- **SQL tool**
  - Best for exact structured numbers:
    - revenue。
    - net income。
    - margins。
    - EPS。
    - assets / liabilities。
    - segment revenue。
    - geographic revenue。
    - rankings and growth calculations。

- **RAG tool**
  - Best for filing evidence:
    - MD&A commentary。
    - risk factors。
    - management explanations。
    - strategy / competition / regulatory discussion。
    - table values not in the structured database。

- **Hybrid questions**
  - Often require SQL first, then RAG:
    - SQL identifies the fastest-growing segment。
    - RAG retrieves what management said about that segment。
  - Important design rule:
    - SQL output should feed concrete entity/year/metric into the RAG question。
    - Do not ask RAG with vague wording if SQL already identified the entity。

### 3.2 Step 1: Build SQL and RAG Tools First

- Before building the agent, make tools reliable as standalone components。
- Tool layer goals:
  - clear input schema。
  - stable output JSON。
  - error statuses instead of crashes。
  - fallback behavior。
  - logs for debugging and eval。

### 3.3 SQL Tool Design

- **Input**
  - Natural language `question` only。
  - Agent is not allowed to write raw SQL directly。
  - The SQL generator receives schema and rules。

- **Output**
  - `status`:
    - `success`
    - `empty_result`
    - `cannot_answer`
    - `error`
    - `fallback`
  - generated SQL。
  - result rows。
  - row count。
  - correction flag。
  - error message。
  - latency log。

- **Guardrails**
  - SQLite only。
  - Read-only connection。
  - Only `SELECT` statements allowed。
  - Forbidden keywords:
    - `DROP`
    - `DELETE`
    - `INSERT`
    - `UPDATE`
    - `ALTER`
    - `PRAGMA`
    - transaction commands。
  - Reject comments and multiple statements。
  - Only allow known tables。
  - Limit large row-level queries。
  - Return `CANNOT_ANSWER` for unavailable metrics such as Azure-only revenue or stock price。

- **Fallback**
  - If generated SQL fails execution:
    - Ask LLM to correct SQL using original question, failed SQL, and error message。
    - Retry up to 3 times。
  - If correction fails:
    - return fallback JSON rather than throwing。

- **SQL eval**
  - Test direct numeric questions。
  - Test unavailable metrics。
  - Test growth / margin / ranking questions。
  - Test whether SQL returns raw values when Python or final answer should compute multi-step comparison。
  - In agent policy eval, SQL cases check whether the agent chooses SQL and preserves required entities, years, and metrics。

- **Corner cases**
  - Fiscal year means the year the fiscal period ends, not calendar start。
  - Segment names differ by company。
  - Some desired business metrics are not in the database。
  - Growth questions may need multiple years of raw data。
  - SQL may answer numbers, but cannot explain management reasons。

- **Tradeoffs**
  - Natural-language SQL input is easy for agent and user。
  - It requires strong validation because LLM-generated SQL can be wrong。
  - Keeping SQL tool limited to read-only structured data makes it safer and easier to evaluate。

- **Future improvements**
  - Add a larger golden SQL eval set。
  - Add deterministic post-processing for growth, CAGR, ranking, and margins。
  - Add schema-aware query planner for multi-step numeric questions。
  - Add unit tests for SQL validation and correction paths。

### 3.4 RAG Tool Design

- **Input**
  - `question`: one standalone, single-intention filing question。
  - `ticker`: optional but should be passed when known。
  - `fiscal_year`: optional but should be passed when known。
  - Important: ticker/year should be structured parameters, not only text inside the question。

- **Output**
  - answer。
  - status:
    - `success`
    - `fallback_success`
    - `insufficient_context`
    - `needs_decomposition`
    - `error`
  - reranked top text chunks。
  - expanded context。
  - table contexts。
  - scope filters。
  - retrieval confidence。
  - sufficiency check。
  - latency。
  - fallback trace。

- **Runtime retrieval pipeline**
  - Apply metadata filter by ticker and fiscal year。
  - Embed query once。
  - Parallel retrieval:
    - text vector top K。
    - BM25 top K。
    - table summary vector top K。
  - Merge vector + BM25 text hits。
  - Rerank text hits only。
  - Filter table summary hits by similarity threshold。
  - Expand text context:
    - selected chunk。
    - section preamble。
    - same text-unit neighbors。
    - same section neighbors。
  - Inject table markdown from:
    - table summary vector hits。
    - text chunk `table_refs` / `table_anchors`。
  - Generate answer using only provided text/table context。

- **Why dual-path RAG**
  - Text and table evidence should not compete in a single rerank pool。
  - Text rerank is good for narrative context。
  - Table summary similarity is better for table evidence。
  - Final answer should see both。

- **Fallback**
  - If answer looks insufficient:
    - rewrite retrieval query using concrete 10-K terms。
    - preserve original answer question。
    - retry retrieval。
  - If retrieval confidence is weak:
    - rewrite retrieval query before answer generation。
  - If multi-ticker or multi-year filters are passed:
    - return `needs_decomposition` and require separate RAG calls。

- **RAG eval**
  - Text vector hit-rate:
    - generate chunk-grounded questions。
    - measure Hit@1 / Hit@3 / Hit@5 / Hit@10。
  - Cross-chunk hit-rate:
    - test multi-chunk questions and target recall。
  - RAGAS-style eval:
    - faithfulness。
    - answer relevancy。
    - context precision。
    - reference coverage。
  - Table-specific eval:
    - expected table in vector top K。
    - expected table passes threshold。
    - expected table appears in final context。
    - numeric answer matches gold within tolerance。
  - Tool-policy eval:
    - whether agent should call RAG。
    - whether it decomposes comparisons。
    - whether it retries after insufficient RAG。

- **Representative metrics to present**
  - Single-block retrieval:
    - Hit@1: `80%`
    - Hit@3: `94%`
    - Hit@5: `97%`
    - Hit@10: `97%`
  - RAGAS-style mixed eval:
    - target in vector top 10: `97.1%`
    - target in rerank top 3: `94.3%`
    - target in expanded context: `94.3%`
    - target table in context: `80%`
    - faithfulness: `4.89 / 5`
    - answer relevancy: `4.69 / 5`
    - average latency: `6.75s`
  - MSFT parsed table regression:
    - old merged rerank: answer correctness `8/10`, target table in context `6/10`
    - new dual-path: answer correctness `10/10`, target table in context `10/10`

- **Corner cases**
  - Abstract wording:
    - User asks “strategic importance” but filing uses concrete terms like revenue contribution, business description, growth drivers。
    - Fallback query rewrite uses filing-like terms。
  - Table questions:
    - Correct table may have low text rerank score。
    - Need table summary path independent of text rerank。
  - Multi-company comparison:
    - One RAG call should not search multiple filings。
    - Decompose into one call per ticker/year/intention。
  - Table threshold:
    - Too high drops useful tables。
    - Too low injects irrelevant tables。
    - MSFT results suggested `0.65-0.70` may be safer than `0.75` in some mixed cases。
  - Neighbor expansion:
    - Helps when answer spans adjacent chunks。
    - Can introduce noise if nearby section is loosely related。

- **What did not fully work**
  - Single rerank pool for text and tables。
  - Relying only on vector retrieval without BM25 for exact filing terms。
  - Relying only on text chunk table refs; refs can be wrong or incomplete near page breaks。
  - Simple thresholds for “should retry” can be dangerous:
    - supported table questions may have weak text but strong table signal。
    - unsupported but semantically related questions can have high text scores。

- **Future improvements**
  - Better labeled RAG eval sets:
    - supported text。
    - supported table。
    - unsupported-but-plausible filing questions。
    - multi-hop SQL -> RAG questions。
  - Calibrate retrieval fallback using text + table confidence jointly。
  - Add query decomposition before retrieval for true multi-intent questions。
  - Add context budgeter to choose preamble, neighbors, and tables under token budget。
  - Add deterministic table arithmetic / value extraction for high-value financial metrics。

### 3.5 Routing Design

- **Routing before agent**
  - First make `sql` and `rag` work independently。
  - Then define a tool policy:
    - SQL for structured numbers。
    - RAG for filing narrative/table evidence。
    - SQL -> RAG for hybrid numeric + explanation questions。
    - separate RAG calls for each company/year in comparisons。

- **Routing as AI workflow**
  - Early step can be viewed as a workflow:
    - classify question。
    - call SQL or RAG。
    - run fallback if needed。
    - synthesize answer。
  - This helped clarify tool contracts before moving to a general agent。

- **Routing as agent**
  - Later use LangChain `create_tool_calling_agent` + `AgentExecutor`。
  - Model sees tool schemas and system routing rules。
  - Agent can choose multiple steps:
    - SQL。
    - RAG。
    - SQL then RAG。
    - RAG retry。
    - send email after final answer。

- **Tradeoff**
  - Workflow is easier to control and evaluate。
  - Agent is more flexible for multi-step user questions。
  - Agent requires much stronger policy prompt and eval because mistakes are less deterministic。

## 4. Block C: Agent, Memory, Observability, and Evaluation

### 4.1 Agent Design

- **Agent stack**
  - LangChain tool-calling agent。
  - Tools:
    - `sql`
    - `rag`
    - `send_email`
  - Prompt structure:
    - system instructions。
    - DB schema and filing coverage。
    - RAG scope rules。
    - tool catalog。
    - optional session memory。
    - recent chat history。
    - current user question。
    - agent scratchpad for current tool calls。

- **Important agent rules**
  - Use tool observations as only evidence。
  - Do not invent numbers or filing claims。
  - Distinguish:
    - “The filing says...” from RAG。
    - “Calculated from SQL data...” from SQL。
  - For hybrid questions, usually SQL first, then RAG。
  - Carry concrete entities from SQL into RAG。
  - Decompose RAG calls by ticker/year/intention。
  - Retry RAG when previous observation is only “phrase not found” or insufficient。
  - Ask for email if user wants send_email but no address is known。

- **Why agent was needed**
  - Users ask mixed questions:
    - “Which segment grew fastest, and what did management say about it?”
  - A fixed one-shot router is brittle for:
    - dependency chains。
    - follow-up questions。
    - tool failure recovery。
    - email action after answer。
  - Agent can perform multi-step reasoning over tool outputs。

- **Agent output / trace**
  - Tool outputs are compacted before being fed back to the agent。
  - Trace records:
    - step index。
    - action。
    - action input。
    - observation。
  - UI can stream step starts and completed steps。

- **Tradeoff**
  - Agent adds flexibility but increases eval burden。
  - Prompt policy must be very explicit。
  - Tool outputs must be compact to avoid context bloat。
  - Need guardrails to prevent agent from answering with meta-comments instead of actually calling tools。

### 4.2 Memory Design

- **Why memory matters**
  - Users ask follow-ups:
    - “make that shorter”
    - “email it to me”
    - “compare that to Microsoft”
  - Agent needs to remember user preferences and recent answers without rerunning tools unnecessarily。

- **Three-part memory design**
  - **Short-term memory**
    - Recent user/assistant turns stored in `messages`。
    - Injected as LangChain `chat_history`。
    - Sliding window keeps recent verbatim context。
  - **Long-term summary / semantic memory**
    - Older messages folded into session summary。
    - Semantic notes store `Q/A` style snippets。
    - Retrieved by keyword overlap against current query。
    - Used for related follow-ups, not as authoritative filing evidence。
  - **Persistent user facts / preferences**
    - Stored in `memory_facts`。
    - Examples:
      - `user_email`
      - response style。
      - output language。
      - preferred units。
      - explicit “remember that...” preferences。
    - Always injected when relevant。

- **Memory boundaries**
  - Memory is session/user context, not financial evidence。
  - Filing claims still must come from RAG。
  - Financial numbers still must come from SQL。
  - Current-turn tool calls live in `agent_scratchpad`; they are not automatically restored as long-term memory。

- **Compression**
  - Keep a short sliding window of recent turns。
  - Fold older messages into `sessions.summary` without LLM summarization。
  - Cap summary length。
  - Cap semantic notes。
  - Store tool artifacts for debugging, but do not inject them into normal chat history。

- **Tradeoff**
  - Simple keyword retrieval is explainable and cheap。
  - It is less semantically powerful than vector memory。
  - Avoiding LLM summarization reduces cost and hallucination risk, but summaries are less polished。

- **Future improvements**
  - Better semantic memory retrieval using embeddings。
  - More robust preference extraction。
  - Memory eval:
    - follow-up resolution。
    - email reuse。
    - style preference persistence。
    - avoiding irrelevant old memory。
  - Clear UI controls to inspect / clear session memory。

### 4.3 Evaluation Strategy

- **Why evaluation was hard**
  - There is no ready-made golden set for this exact task。
  - Need to evaluate multiple layers separately:
    - PDF parsing。
    - subsection boundaries。
    - table extraction。
    - text retrieval。
    - table retrieval。
    - answer generation。
    - tool routing。
    - multi-step agent behavior。
  - Many failures only appear in corner cases:
    - cross-page tables。
    - header-only continuations。
    - subtle subsection headings。
    - abstract user wording。
    - SQL-to-RAG dependency chains。
  - Some “correct” agent actions are not unique。
    - A reasonable agent might split a question into different valid tool calls。
    - String-based eval can under-credit semantically valid calls。

- **What I evaluated**
  - **Text retrieval**
    - Generate questions from known chunks。
    - Measure whether target chunk appears in top K。
  - **Cross-chunk retrieval**
    - Test groups of adjacent chunks。
    - Measure any-target and all-target recall。
  - **RAG answer quality**
    - RAGAS-style LLM judge:
      - faithfulness。
      - answer relevancy。
      - context precision。
      - reference coverage。
  - **Table QA**
    - Build specific MSFT parsed-table question set。
    - Track table vector hit, threshold pass, final context hit, numeric correctness。
  - **Table layout hard cases**
    - Compare custom baseline with PyMuPDF and PyMuPDF4LLM on selected hard pages。
  - **Agent tool policy**
    - 100-case next-step eval。
    - Tests direct SQL, direct RAG, SQL->RAG dependency, compare decomposition, stop/no-tool cases, fallback behavior。

- **Eval limitations**
  - Generated retrieval questions can be too easy or too close to source wording。
  - LLM-as-judge can be inconsistent。
  - Numeric scoring needs unit normalization。
  - Agent policy expected action can be ambiguous。
  - Need more hand-labeled hard cases。

- **Future eval improvements**
  - Create a curated benchmark from real analyst-style questions。
  - Add human-labeled expected source chunks/tables。
  - Add unsupported-but-plausible questions to test abstention。
  - Add regression tests for every discovered corner case。
  - Add table visual correctness labels, not only text/numeric answer labels。
  - Add end-to-end eval that measures:
    - correct tool route。
    - correct evidence retrieval。
    - correct final answer。
    - latency。
    - confidence / abstention behavior。

## 5. Results and Metrics to Highlight

- **Offline parsing outputs**
  - Apple FY2025 example:
    - main sections: `23`
    - subsections: `179`
    - raw detected tables: `43`
    - merged tables: `42`
    - text chunks: `161`
    - chunks with table refs: `19`

- **Text retrieval**
  - Single-block retrieval:
    - Hit@1: `80%`
    - Hit@3: `94%`
    - Hit@5: `97%`
    - Hit@10: `97%`

- **RAG quality**
  - RAGAS-style mixed eval:
    - target in vector top 10: `97.1%`
    - target in rerank top 3: `94.3%`
    - target in expanded context: `94.3%`
    - target table in context: `80%`
    - faithfulness: `4.89 / 5`
    - answer relevancy: `4.69 / 5`
    - reference coverage: `4.54 / 5`
    - average latency: `6.75s`

- **Table QA improvement**
  - Old merged text+table rerank:
    - target table in final context: `6/10`
    - answer correctness: `8/10`
  - New dual-path text + table retrieval:
    - target table in final context: `10/10`
    - answer correctness: `10/10`

- **Table layout benchmark**
  - Custom baseline on hard MSFT cases:
    - score `0.950`
    - latency `0.007s`
  - PyMuPDF `find_tables`:
    - score `0.662`
    - latency `4.715s`
  - PyMuPDF4LLM layout:
    - score `0.513`
    - latency `3.751s`

- **Latency**
  - RAGAS-style average latency around `6-7s` for some runs。
  - Table-heavy MSFT runs can be higher。
  - Reranking often dominates latency。
  - Anthropic Haiku rerank + Sonnet answer improved practical latency vs slower chat/rerank settings。

## 6. Key Challenges and Lessons Learned

- **PDF parsing is the foundation**
  - If section/subsection/table metadata is wrong, RAG cannot fully recover。
  - Good retrieval starts with good document structure。

- **Tables should not be treated as normal text**
  - Flattened table rows hurt text embedding quality。
  - Table evidence needs separate retrieval and answer handling。

- **A good RAG system needs multiple eval layers**
  - Retrieval hit rate alone is not enough。
  - Need to know:
    - Was the right chunk retrieved?
    - Was the right table retrieved?
    - Did threshold filtering drop it?
    - Did expansion include it?
    - Did answer generation read it correctly?
    - Did agent choose the right tool?

- **Corner cases drive the design**
  - Many improvements came from finding hard examples:
    - header-only cross-page tables。
    - wrong section assignment near page breaks。
    - all-caps centered headings。
    - abstract wording fallback。
    - table summaries retrieved but dropped by rerank。

- **Agent flexibility requires stricter contracts**
  - Tools must return stable statuses and compact JSON。
  - Prompt must explicitly define when to use SQL vs RAG。
  - Agent must not treat insufficient retrieval as final evidence。
  - Tool-policy eval is necessary because agent behavior can regress without code changes。

- **What worked well**
  - TOC-guided sectioning。
  - Font/layout subsection heuristics。
  - Text-only chunks with table refs。
  - Dual-path RAG。
  - SQL/RAG tool separation。
  - Explicit RAG scope parameters。
  - Memory separated into chat history, facts/preferences, and semantic notes。

- **What did not work as well**
  - Regex-only section detection。
  - PyMuPDF-only table extraction。
  - One combined text/table rerank pool。
  - Eval sets generated only from chunks without enough real hard cases。
  - Simple confidence thresholds as a universal fallback trigger。

- **What I would do differently**
  - Build a labeled eval set earlier。
  - Track corner cases from day one as regression tests。
  - Separate visual table detection, table semantic parsing, and table QA eval more explicitly。
  - Add observability earlier to inspect retrieval and agent steps faster。

## 7. Future Work If the System Worked Perfectly

### 7.1 Product Improvements

- Add analyst workflow features:
  - saved research sessions。
  - source citations with page/table preview。
  - compare filings side-by-side。
  - export answer with evidence。
  - ask follow-up questions over selected chunks/tables。

- Improve Chunk Studio:
  - persist visual regions during processing。
  - add debug overlay for table anchors and heading candidates。
  - add manual correction UI for table boundaries and subsection headings。
  - automatically turn manual corrections into regression tests。

### 7.2 Technical Improvements

- **LangGraph**
  - Use LangGraph if workflow becomes more structured and stateful:
    - SQL node。
    - RAG node。
    - decomposition node。
    - evidence sufficiency node。
    - answer synthesis node。
    - email node。
  - Benefits:
    - clearer state machine。
    - easier retries。
    - explicit branching。
    - better testability than a free-form agent loop。

- **LangSmith**
  - Expand tracing and eval logging:
    - tool latency。
    - input/output payloads。
    - retrieval candidates。
    - rerank decisions。
    - final answer quality。
  - Use LangSmith datasets for regression eval:
    - route correctness。
    - RAG answer faithfulness。
    - SQL accuracy。
    - multi-step agent trajectory。

- **Better table reasoning**
  - Normalize VLM markdown into structured rows。
  - Add deterministic value lookup and arithmetic。
  - Add multi-modal answer path for complex tables。
  - Add table-specific confidence and abstention。

- **Better retrieval**
  - Calibrate table thresholds per filing type。
  - Add query decomposition for true multi-hop questions。
  - Add context budget optimization。
  - Add hybrid sparse/dense retrieval tuning。
  - Use learned reranker or cross-encoder if latency budget allows。

- **Better memory**
  - Embed semantic memories。
  - Add memory relevance classifier。
  - Add preference-management UI。
  - Add tests for memory contamination and stale preferences。

- **Better eval**
  - Build a curated suite of real analyst questions。
  - Add negative / unsupported questions。
  - Add visual table boundary labels。
  - Add CI regression tests for discovered corner cases。
  - Evaluate full trajectory, not just final answer。

## 8. Suggested Slide Structure

- Slide 1: Project title and one-sentence goal
  - “An agentic financial research assistant for 10-K filings using SQL + section-aware RAG + table-aware retrieval。”

- Slide 2: Audience and business problem
  - Who uses it。
  - Why 10-K research is hard。
  - What success means。

- Slide 3: End-to-end architecture
  - Offline parsing。
  - SQL/RAG tools。
  - Agent and memory。

- Slide 4: Offline parsing and chunking
  - TOC-guided sections。
  - subsection detection。
  - text chunks and expansion metadata。

- Slide 5: Table handling
  - why tables are hard。
  - layered extraction。
  - VLM markdown + table summary index。
  - key corner cases。

- Slide 6: RAG pipeline
  - text path。
  - table path。
  - dual-path context assembly。
  - fallback。

- Slide 7: SQL and routing
  - SQL input/output。
  - guardrails。
  - SQL -> RAG workflow。
  - routing rules。

- Slide 8: Agent and memory
  - LangChain agent。
  - tool policy。
  - memory three layers。
  - trace / observability。

- Slide 9: Evaluation and results
  - retrieval metrics。
  - RAGAS metrics。
  - table QA improvement。
  - table layout benchmark。

- Slide 10: Lessons learned and future work
  - what worked。
  - what did not。
  - eval difficulty。
  - LangGraph / LangSmith / table reasoning improvements。

## 9. Q&A Topics to Be Ready For

- Why not just use a standard PDF-to-markdown library?
  - Because standard tools do not solve 10-K Item/subsection-aware RAG or cross-page table linking out of the box。

- Why not put tables into the same text chunk?
  - It hurts text embeddings and makes narrative retrieval behave like noisy numeric lookup。

- Why use both SQL and RAG?
  - SQL is reliable for structured numbers; RAG is needed for filing explanations and evidence。

- Why is eval difficult?
  - Need to evaluate parsing, retrieval, table evidence, generation, and agent routing separately。
  - Ground truth is often not obvious without manually inspecting PDF corner cases。

- What was the biggest technical challenge?
  - Table extraction and evaluation:
    - cross-page continuations。
    - header-only bands。
    - visual hierarchy。
    - deciding whether the system retrieved the right evidence。

- What would be the next production step?
  - Build a curated eval set and regression suite。
  - Add stronger observability with LangSmith。
  - Consider LangGraph for explicit workflow control。
  - Improve table normalization and deterministic table QA。

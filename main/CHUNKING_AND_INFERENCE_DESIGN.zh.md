# 分块与推理设计

英文版：`CHUNKING_AND_INFERENCE_DESIGN.md`

本文档描述当前金融 RAG 系统中 PDF 分块、资产提取、向量索引、检索、重排序、上下文扩展与推理的设计。当前实现主要在 Apple FY2025 10-K 上开发并验证，但设计意图是泛化至 Apple、Microsoft 与 Alphabet 的 10-K 申报文件。

核心目标是在合适的粒度上使 10-K 文本可检索，同时保留足够的结构元数据，以便在推理阶段恢复上下文。设计将叙事文本检索与表格数值检索分离，因为金融表格与散文的行为不同，不应被静默混入文本块中。

关于 Chunk Studio 产品层——可视化表格区域检测、裁剪渲染、UI 流程，以及 API 调用时使用的列对齐检测器——请参阅 `chunk_studio/DESIGN.md`。

## 高层工作流

流水线包含六个阶段：

1. 解析可见的 10-K 目录，并将其作为章节发现的菜单。
2. 使用菜单引导的顺序匹配，在正文中定位主要 Item 章节。
3. 利用字体/样式信号检测每个 Item 章节内更小的子章节。
4. 将表格/图片作为独立资产提取，并附加到最近的章节/子章节。
5. 构建纯文本 RAG 块，包含标题路径、表格引用与推理扩展元数据。
6. 在独立的推理层运行推理，采用**并行文本 + 表格摘要检索**、纯文本重排序、表格阈值过滤、上下文扩展、VLM 表格 Markdown 注入与答案生成。

代码按职责拆分：

- `main/chunking/` 包含 PDF 解析、章节划分、资产提取、文本块构建、向量索引、离线 VLM 表格解析、表格摘要向量索引，以及检索命中率评估。
- `main/inference/` 包含运行时 RAG 推理与 RAGAS 风格答案评估。
- `chunk_studio/` 封装同一流水线，用于上传、可视化、VLM 解析检查与问答。

当前 Apple FY2025 输出包括：

- 菜单引导章节：`main/chunking/AAPL_FY2025_menu_guided_sections.json`
- 章节关联资产：`main/chunking/AAPL_FY2025_section_assets.json`
- 纯文本 RAG 块：`main/chunking/AAPL_FY2025_rag_chunks.json`
- 文本向量 DB：`data/index/text_chunks/vectors.db`
- 表格摘要向量 DB：`data/index/table_summaries/vectors.db`（VLM 解析 + `build_table_vector_db.py` 之后）
- 离线 VLM 表格解析：`main/chunking/vlm_table_parse.py`
- 表格摘要索引构建器：`main/chunking/build_table_vector_db.py`
- 推理脚本：`main/inference/text_vector_rag_inference.py`
- 混合推理评估：`main/inference/run_mixed_inference_eval.py`
- RAGAS 风格混合评估：`main/inference/eval_msft_mixed_ragas_style.py`
- MSFT 混合测试集：`main/common/msft_fy2025_mixed_15_inference_test.json`
- MSFT 表格测试集：`main/common/msft_fy2025_parsed_table_test_questions.json`
- MSFT 表格评估：`main/inference/eval_msft_table_test_questions.py`
- 推理评估脚本：`main/inference/eval_single_questions_ragas_style.py`

当前 Apple FY2025 统计：

- 主章节：23
- 子章节：179
- 检测到的原始表格：43
- 合并后表格：42
- 检测到的图片：1
- 文本块：161
- 文本向量 DB 中的表格块：0
- 含表格引用的文本块：19
- 含图片引用的文本块：1

## 模型提供商与环境

推荐的运行时拆分是：**Fireworks 仅用于嵌入**，**Anthropic 用于所有对话步骤**（重排序、答案生成、RAGAS 评判、agent/sql 对话）：

| 阶段 | 提供商 | 默认模型 |
| --- | --- | --- |
| 离线索引构建 + 每查询向量搜索 | Fireworks | `nomic-ai/nomic-embed-text-v1.5` |
| 文本重排序 | Anthropic | `claude-haiku-4-5-20251001` |
| 答案生成 | Anthropic | `claude-sonnet-4-20250514` |
| RAGAS 评判 / agent 循环 | Anthropic | 配置的对话模型 |

当设置了 `ANTHROPIC_API_KEY` 时，`call_chat()` 永远不会路由到 Fireworks，即使调用方传入 Fireworks 模型 ID（如 `accounts/fireworks/models/qwen3-8b`）。`FIREWORKS_API_KEY` 仅嵌入时需要。

通过仓库根目录 `.env` 配置：

```bash
FIREWORKS_API_KEY=...          # 仅用于嵌入
ANTHROPIC_API_KEY=...
ANTHROPIC_RERANK_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-20250514
```

`load_env_file()` 会用 `.env` 中的值覆盖 shell 环境变量，因此编辑文件并重新加载后，本地密钥变更即可生效。

离线文本/表格向量已存储在 SQLite 中。推理时，Fireworks **每个问题调用一次**以嵌入查询；配置 Anthropic 时，重排序与答案生成不使用 Fireworks。

## 为何采用菜单引导设计

早期基于页码范围与通用 Item 标题正则的方案过于脆弱。SEC 10-K PDF 常在页脚、目录、交叉引用、审计报告与页眉中重复出现 Item 引用。PDF 页码与印刷 10-K 页码之间的偏移也可能不一致。

当前设计从可见目录入手，因为它提供了预期章节名称的有序菜单。解析器不依赖页码，而是将目录作为预期 Item 的序列，再按顺序在正文中文搜索。这使章节划分在不同公司与申报风格间更稳健。

重要原则是：目录用作有序语义菜单，而非页码坐标来源。

## 章节检测

章节检测过程使用三层结构：

1. 识别申报文件开头附近可能的目录页。
2. 从这些目录页解析 Item 条目与标题。
3. 在正文中按顺序搜索匹配的 Item 标题与标题文本。

正文搜索基于字符偏移。一旦找到章节标题，章节内容从该标题切至下一个检测到的标题。这避免依赖页码范围确定最终边界。

每个章节获得稳定的章节引用，例如：

- `Part I::Item 1`
- `Part I::Item 1A`
- `Part II::Item 7`
- `Part II::Item 8`

章节对象保留 part、item、章节标题、用于调试的页码跨度、字符数，以及表格/图片引用等元数据。紧凑资产输出中不保留大型原始文本，因为 RAG 就绪表示以块形式存储文本。

## 目录与章节边界情况

章节划分逻辑必须处理若干 PDF 特有问题：

- 可见目录可能包含换行、标点差异或重复的点引导符。
- 部分正文页在普通散文中提及「Item 1A」或「Item 7」，这些不应成为章节起点。
- 印刷页码可能与 PDF 页索引不一致。
- 部分申报在目录外有看起来像目录条目的短行。
- 部分章节没有有意义的序言，直接以子章节开始。
- 部分章节（尤其是 exhibits）包含视觉上类似表格的列表式内容。

菜单引导方法通过强制顺序降低误报。若目录表明下一章节应为 `Item 7`，匹配器不会将段落中稍后出现的无关 `Item 1A` 提及接受为新章节边界。

目录页检测器有意保守。它扫描早期页面，并要求有多条 Item 条目与类页码模式等证据。这避免将正文页错误标记为目录页。

## 子章节检测

切分主要 Item 章节后，流水线使用 PDF 的字体/样式特征检测更小的子章节：

- 字号
- 粗体
- 斜体
- 左缩进
- 行长度
- 该行是否像独立标题
- 文本是否以句子方式结尾
- 行首是否为常见段落词

这产生如下子章节路径：

- `Business > Products > iPhone`
- `Risk Factors > conflict, terrorism, natural disasters...`
- `Management's Discussion and Analysis > Gross Margin > Services`
- `Financial Statements and Supplementary Data > Note 4 – Financial Instruments > Cash, Cash Equivalents and Marketable Securities`

子章节作为物化文本单元存储，而非仅标题标签。每个子章节块具有路径、标题、层级、页码与文本内容。

## 子章节边界情况

子章节检测是启发式的，存在若干已知局限：

- 部分真实标题视觉上较 subtle，可能像普通散文。
- 部分斜体段落看起来像标题，实则为说明性注释。
- 极短标题如「Services」若表格文本仍存在，可能与表格行标签混淆。
- 较长的风险因素标题可能跨多行，难以与普通段落区分。
- Exhibit 章节含许多对检索语义价值不高的类标题行。
- 财务报表附注常有嵌套标题与表格紧密交错。

当前方法优先选择对检索有用的边界，而非完美的视觉重建。若标题缺乏强字体/样式信号，可能被遗漏。部分低价值列表项可能变成块，尤其在 exhibit 章节。

## 表格提取与附加

表格与文本分开提取。当前资产提取器使用 PyMuPDF 表格检测查找边界框、原始行、页位置与粗略复杂度元数据。

每个表格附加到：

- 主章节引用
- 尽可能附加的子章节引用
- 起始/结束页
- 边界框或每页边界框
- 原始行
- 原始文本表示
- 如适用，跨页合并元数据

表格不作为普通文本块嵌入文本向量 DB。文本块保留表格引用，如 `table_004` 或 `table_group_041`。推理时，表格证据可通过两条独立路径进入上下文：

1. **文本块引用** — 若扩展后的文本锚点块有 `table_refs`，加载对应表格 Markdown。
2. **表格摘要向量检索** — 若 VLM 表格摘要通过相似度阈值，直接加载其 Markdown，无需经过文本重排序。

这种分离是因为表格数值应作为结构化或半结构化数据处理。将原始表格行直接混入散文块会使嵌入噪声增大，导致文本检索行为像表格查找。

## 离线 VLM 表格解析（阶段 1）

对于需要可靠数值 QA 的表格，首选证据不再是扁平化的 `raw_rows`。离线路径为：

1. 从 PDF bbox 渲染表格裁剪 PNG（含垂直 padding）。
2. 从 PDF 文本层收集表格前后 ±2 句叙事上下文。
3. 调用 VLM（Anthropic 或 Fireworks 回退）生成：
   - `markdown` — 完整 GitHub 风格表格
   - `summary` — 一句检索摘要
4. 将结果存入 `assets.json` 中各表格的 `vlm_parse` 下。

脚本：`main/chunking/vlm_table_parse.py`

评估/调试有用的 CLI 过滤器：

- `--min-page 3` — 跳过目录/正文前页
- `--exclude-item Item 5` — 跳过股票回购表等
- `--max-tables N` — 批量大小控制

Chunk Studio 在右侧 **Parses** 标签页展示解析结果，并通过 `/api/files/{id}/tables/{table_id}/parse-image.png` 提供裁剪图。

## 表格摘要向量索引（阶段 2）

VLM 解析成功后，将每个表格的 `vlm_parse.summary` 嵌入**独立**向量 DB：

- 全局路径：`data/index/table_summaries/vectors.db`
- 每工作区路径（Chunk Studio）：`{workspace}/index/table_vectors.db`

脚本：`main/chunking/build_table_vector_db.py`

重要：文本块与表格摘要位于不同 DB，推理时独立查询。**不会合并为单一候选池**。

## 跨页表格处理

部分金融表格跨多页。资产提取阶段有跨页合并启发式，核心路径如下：

```text
find_tables()
  -> 同页碎片合并
  -> synthesize_page_bottom_header_tables()   # find_tables 漏掉的页底表头
  -> supplement_header_only_continuations()   # 下一页顶部无表格时从文本层合成续行
  -> merge_connected_tables()                 # 跨页打分合并
```

### 常规合并条件

- 前一表格接近页底（`page_height - bbox.y1 < 95pt`）。
- 下一表格在下一页顶部附近开始（`bbox.y1 < 130pt`）。
- 列数或表格宽度兼容。
- 续表不像完全无关的财务报表。

### header-only 退化表（MSFT Item 5 股息表案例）

这是当前最难、也最有代表性的边界情况：

```text
page 32 bottom:
  Declaration Date | Record Date | Payment Date | Dividend Per Share | Amount
  （find_tables() 常完全漏检）

page 33 top:
  June 10, 2025 | August 21, 2025 | September 11, 2025 | $0.83 | $6,170
  （find_tables() 可检出，但易被归到 Item 7）
```

处理策略：

1. **`pending_merge=True`** — 对「只有 header、无 data row、紧贴页底」的表（含合成的 `table_header_band`）标记待合并。
2. **合成页底表头** — 从 PDF 文本层扫描页底 130pt，把同一 y 带上的列标题拼成 header band；要求命中 ≥2 个表头关键词（declaration / record date / payment date 等），排除散文句。
3. **降低合并阈值** — header-only 前表 merge 阈值从 5 降到 4；列数不一致时仍允许合并（header 5 列 vs data 17 列）。
4. **subsection 归属取 header 半页** — 合并后 `subsection_ref` 和 `section_ref` 继承 header 所在位置，而非续页数据行位置。
5. **页顶续表 section 修正** — `y0 < 150pt` 的表格优先归属上一页 section（避免 Item 5 续表掉进 Item 7）。

合并逻辑有意保守。早期规则过于激进，曾误合并无关报表（如资产负债表 + 现金流量表）。**header-only 跨页**是明确放宽的例外，因为有强几何证据（页底/页顶 + 数据行）。

合并表格获得组 ID，如 `table_group_003`，原始组件仍可能相关。评估需考虑合并表格与其源组件表格之间的语义等价。

### 子章节：全大写孤立行标题（Item 5 案例）

`MARKET AND STOCKHOLDERS`、`SHARE REPURCHASES AND DIVIDENDS` 这类标题常见问题：

- 下划线可能是画线而非 font flag，bold 检测失效。
- 全大写 + 视觉居中 + 字号与正文相同 → 单靠字号无效。

当前 `centered_heading_score()` 使用三项加权：

| 特征 | 分值 |
| --- | --- |
| 行中心 `(x0+x1)/2` 接近页面中心 | +2 |
| 字母 ≥85% 大写 | +2 |
| 前后垂直间距 > 1.25–1.5 × 行高 | 各 +1 |

得分 ≥4 即视为 subsection 候选。这比单独依赖 font flag 更稳。

**隐含假设**：标题层级 ≈ 语义层级。在 Item 5 中，`MARKET AND STOCKHOLDERS` 与 `SHARE REPURCHASES AND DIVIDENDS` 视觉平级，但后者才是真正含表的 subsection；VLM 解析时传入 narrative context（如 "Following are our monthly share repurchases..."）可消解歧义。

## 表格边界情况

表格是当前系统最难的部分。重要边界情况包括：

- PyMuPDF 可能只检测到表格的一部分。
- 表格标题可能在检测边界框之外。
- 续页可能出现重复表头。
- 部分财务报表视觉上像表格，语义上却是宽泛文档。
- 部分行为层级结构，缩进很重要。
- 括号内负值表示可能不一致。
- 表格可能附加到正确章节但错误子章节，若标题靠近页边界。
- Exhibit 章节中部分「表格」实为 exhibit 列表。
- 合并表格 ID 可能与附近文本块引用的表格 ID 不同。
- 文本问题可能从表格引导句生成，但需要文本中不存在的表格数值。

当前设计通过保留表格引用与原始表格文本处理这些。尚未将表格完全规范化为行级或值级记录。对于高质量表格 QA，未来层应解析为规范化记录或使用表格专用答案提示。

## 图片提取

图片作为独立资产提取，通过尺寸过滤后附加到章节/子章节。小的装饰性或噪声图像块被过滤。当前文本向量 DB 不嵌入图片块。文本块保留 `image_refs`，以便未来图片/VLM 路径在需要时检索并处理相关图。

## RAG 文本块构建

最终 RAG 块为纯文本。每个块包含：

- 稳定块 ID
- 源文件
- 章节引用
- 子章节引用
- 标题路径
- 文本内容
- Token 数
- 表格引用
- 图片引用
- 拆分元数据
- 推理扩展元数据

文本内容开头包含标题路径。这改善语义检索，因为像 `iPhone` 或 `Services` 这样的短块否则缺乏足够独立上下文。

块文本有意移除：

- 表格行与扁平化数值表格串
- PDF 页标记
- 重复页脚
- 重复 Item 标题
- 标题路径已表示的冗余章节标签

当前目标约为每块 100–500 token。源子章节本身较短时，部分块更短。系统不在索引时强制合并短块，因为合并会模糊精确章节边界。短块在推理时通过上下文扩展处理。

## 文本块边界情况

文本块构建器需处理若干棘手情况：

- 部分子章节天然较短，如产品描述。
- 部分章节序言是重要背景，但多数问题不直接回答。
- 部分子章节文本含表格引导句，如「The following table shows...」
- 除非显式剥离，部分表格数值仍留在扁平化 PDF 文本中。
- 部分财务报表页含看起来像散文的表格标题与报表名称。
- Exhibit 章节可能产生较长、列表式文本块。
- 移除过多类表格文本会损害询问报表名称或表格描述的问题。
- 保留过多类表格文本会损害文本嵌入。

当前折中是纯文本检索加表格引用。若块引用表格，推理时可添加表格内容，但表格行不作为普通散文嵌入。

## 推理扩展元数据

RAG 块现包含专用于推理时上下文扩展的元数据。这与块归属元数据分开。

归属元数据说明块属于何处：

- `section_ref_id`
- `subsection_ref_id`
- `section_title`
- `header_path`
- `text_unit_id`
- `text_unit_kind`

扩展元数据说明作答时需添加什么：

- 章节序言引用
- 同一拆分文本单元内的前/后块
- 同一章节内的前/后块
- 是否允许跨章节扩展
- 扩展范围

关键设计选择是扩展仅在同一章节内进行。默认禁用跨章节扩展。这防止检索意外将无关章节拉入答案上下文。

最终提示格式化时，选中的锚点块保持全文，章节序言保持全文。相邻前/后块缩短为首两句与末两句。这在保留边界上下文的同时，避免邻近子章节导致提示过大。

## 为何存在章节序言引用

部分 Item 有章节级引言，后接许多子章节。例如，`Risk Factors` 可能在列出具体风险因素前有一般性风险披露。短风险子章节若无序言可能难以解读。

不在每个块中重复序言，每个子章节块存储对章节序言块的引用。推理时若选中子章节，可一次性添加序言。

这避免向量 DB 膨胀，同时保留生成所需的有用上下文。

## 为何存在同一文本单元链接

单个子章节超过目标块大小时，会拆成多块。这些块是同一原始逻辑文本单元的片段。若检索命中一个片段，推理通常应包含前一片段与后一片段。

同一文本单元链接支持这一点。当句子感知拆分器将长子章节切成多块而答案需要周围上下文时尤其重要。

## 为何存在同章节邻居链接

部分问题自然需要同一 Item 章节内邻近子章节。例如：

- MD&A 中产品公告上下文后接关税讨论
- 商业票据后接定期债务
- 股份回购后接普通股股数
- 风险因素序言后接具体风险因素块

同章节邻居链接允许推理添加邻近块，但仅限同一 Item 章节。这与仍禁用的跨章节扩展有意不同。

## 向量索引

### 文本向量 DB

仅文本块嵌入文本向量 DB：

- 全局：`data/index/text_chunks/vectors.db`
- Chunk Studio 工作区：`{workspace}/index/vectors.db`

### 表格摘要向量 DB

仅成功的 VLM 表格摘要嵌入表格摘要向量 DB：

- 全局：`data/index/table_summaries/vectors.db`
- Chunk Studio 工作区：`{workspace}/index/table_vectors.db`

当前嵌入模型：

- `nomic-ai/nomic-embed-text-v1.5`

嵌入维度为 768。嵌入以 float32 blob 存于 SQLite，附带块内容与元数据 JSON。向量搜索在 Python 中对存储嵌入计算余弦相似度。

每行还存储从源文件名解析的申报身份列：

- `ticker` — `AAPL`、`MSFT` 或 `GOOGL`
- `fiscal_year` — `FY2024` 或 `FY2025`
- `source_file` — 如 `MSFT_FY2025_10-K.pdf`

Chunk Studio 上传后磁盘上 PDF 为 `source.pdf`，但建索引时通过 workspace `metadata.json` 的 `original_filename` 解析 canonical 文件名（`main/chunking/filing_metadata.py`）。已有 workspace 可用 `main/chunking/patch_workspace_source_files.py` 批量修补。

多份申报可合并进全局 DB（`data/index/text_chunks/vectors.db`）：向 `build_text_vector_db.py` 传入多个 `*_rag_chunks.json`。

Fireworks 用于嵌入。在本环境中，Python urllib 对 Fireworks 的请求被 Cloudflare 以 403 拦截，而 curl 使用相同密钥与载荷可工作。因此嵌入辅助函数当前通过 curl 调用 Fireworks。

## 检索与推理流程

当前推理路径为**双路径设计**：文本与表格摘要并行检索，分别重排序/过滤，再组装为单一答案上下文。

### 步骤 0 — 可选元数据 scope 过滤（多申报）

向量/BM25 检索前，`load_chunks()` 可通过 SQL `WHERE ticker / fiscal_year` 缩小候选集：

```python
run_pipeline(
    query,
    ticker_filter="MSFT",          # 或 ["MSFT", "GOOGL"]
    fiscal_year_filter="FY2025",   # 或 "2025"
)
```

CLI：`--ticker MSFT --fiscal-year FY2025`。

不传 filter 时搜索全部已索引申报（六份 10-K 约 1500 text chunks）；同时传 ticker + fiscal_year 时收窄到单份申报（约 250 chunks），显著降低跨公司 Item 段落误召回。

Agent 通过 `rag` 工具的结构化参数（`ticker`、`fiscal_year`）传 scope；system prompt（`RAG_SCOPE_RULES`）规定何时传、何时省略、何时传多个 ticker。Hard filter 在代码层执行，不能仅靠 question 字符串里写公司名。

### 步骤 1 — 并行相似度搜索

查询嵌入计算一次，然后用于两个独立 DB：

```text
query embedding
    ↓              ↓
text DB          table summary DB
top 10           top 5
```

两个结果集**不合并、不相互竞争**。

### 步骤 2 — 文本侧重排序

仅将文本 top 10 候选送入重排序器。重排序器选择 **top 3 文本锚点块**。

表格摘要**不参与**重排序。重排序仅针对文本。

实现：`text_vector_rag_inference.py` 中的 `rerank_text_chunks()`。

### 步骤 3 — 表格侧阈值过滤

从表格 top 5 中，仅保留余弦相似度 ≥ 阈值的命中。默认阈值：

- `table_similarity_threshold = 0.75`

低于阈值的命中丢弃。这防止弱表格匹配污染提示。

阈值应在小规模标注查询集上校准。在 MSFT FY2025 表格测试中，多数目标表格得分 0.76–0.86；一张 EPS 表（`table_006_merged`）得分 0.712，在 0.75 被过滤，但仍通过文本块引用正确作答。

在 MSFT FY2025 混合 15 题集（10 表格 + 5 文本）上，阈值 0.75 往往过严：约 1/10 表格问题通过阈值，而约 6/10 在 0.70 通过。对于生产环境 MSFT 风格申报，除非标注评估集确认 0.75 安全，否则起始值宜接近 **0.65–0.70**。

### 步骤 4 — 上下文组装

对每个**文本锚点块**（重排序 top 3）：

- 添加章节序言（全文）
- 添加同章节内前/后块（最终提示中裁剪为首 2 + 末 2 句）
- 检查 `table_refs` 并加载对应表格 Markdown

对每个通过阈值的**表格摘要命中**：

- 直接加载对应 VLM `markdown`

去重规则：同一 `table_id` 在最终表格上下文中最多出现一次。

最终提示顺序：

1. 章节序言
2. 文本锚点/邻居块
3. 表格 Markdown 块，每块前缀为：

```text
[Table: table_005 | Item 7 > Management's Discussion ...]
```

表格正文回退顺序：

1. 当 `vlm_parse.status == success` 时使用 `vlm_parse.markdown`
2. 扁平化 `raw_rows`
3. `raw_text`

### 步骤 5 — 答案生成

将组装的文本 + 表格上下文送入答案模型。设置 `ANTHROPIC_API_KEY` 时，答案生成使用 Anthropic（默认 Sonnet）。重排序使用另一更快 Anthropic 模型（默认 Haiku）。此配置下不使用 Fireworks 对话。

表格上下文是数值/表格值问题的最高优先级证据。

此设计有意分离：

- **文本锚点** — 由重排序选择
- **表格证据** — 由表格摘要向量搜索 + 阈值选择，以及文本锚点可选的 `table_refs`

检索在两侧找到最佳锚点。推理在作答时将其拼接。

## 重排序

重排序**仅应用于文本块**。

重排序器接收文本 top 10 向量候选，选择对回答问题最有用的 top 3 文本块。表格摘要从不进入此步骤。

设置 `ANTHROPIC_API_KEY` 时的默认重排序模型：**Haiku**（`ANTHROPIC_RERANK_MODEL`）。这比早期 Fireworks 重排序运行快得多（MSFT 混合评估中每题约 3s vs 约 15s），在答案生成使用 Sonnet 时仍保持可接受答案质量。

重排序器尤其适用于：

- 在相似风险因素块之间选择
- 在若干相关附注中选择正确财务报表附注
- 选择表格引导子章节而非通用章节序言
- 避免无关但语义宽泛的块

重排序 top 3 文本块是文本侧扩展的锚点。它们不是作答时唯一证据，因为表格 Markdown 可能通过表格阈值路径单独进入。

若重排序模型返回无效 JSON，流水线对文本锚点回退到向量顺序。

## 推理中的表格上下文

表格证据通过两条路径进入答案上下文：

### 路径 A — 表格摘要检索 + 阈值

1. 从表格向量 DB 检索 top 5 表格摘要。
2. 保留相似度 ≥ `table_similarity_threshold`（默认 0.75）的命中。
3. 为这些表格 ID 加载 VLM Markdown。

此路径不依赖文本重排序。它修复了早期失败模式：表格摘要检索正确，但因重排序仅选文本块而被丢弃。

### 路径 B — 文本块 `table_refs`

若任何扩展文本锚点块引用 `table_004`，加载该表格 Markdown 作为次要证据。在以下情况仍有用：

- 目标表格摘要得分略低于阈值
- 问题自然锚定在引用附近表格的叙事文本上

两条路径按 `table_id` 去重。

最终提示中每个表格块格式为：

```text
[Table: {table_id} | {section/header path}]
markdown:
...
```

答案提示将表格上下文视为数值/表格值问题的最高优先级证据。

当前局限：若未对表格运行 VLM 解析，流水线回退到扁平化 `raw_rows` / `raw_text`，对复杂版式较弱。生产表格 QA 应在构建表格摘要索引前运行 `vlm_table_parse.py`。

## 查询分解

曾考虑对多块问题做查询分解。基于评估，建议为：

- 普通单主题问题，使用直接检索。
- 明确多意图问题，分解可能有帮助。
- 无论是否分解，检索后使用上下文扩展。

主要原因是许多多块失败并非纯嵌入失败。部分生成的评估组包含并非真正需要的相邻块。此类情况下分解帮助不大。对于真正的多跳问题，如一部分关于商业票据、另一部分关于股份回购，分解可能提高召回。

## 评估摘要

已运行若干健全性评估。

单块检索，100 道生成问题：

- Hit@1（首位命中率）：80%
- Hit@3：94%
- Hit@5：97%
- Hit@10：97%

跨块检索，2–5 块混合组：

- 任一目标 Hit@10：98%
- 平均目标召回@10：65%
- 全部目标 Hit@10：30%

组大小为 2 时：

- 平均目标召回@10：87.5%
- 全部目标 Hit@10：75%
- 任一目标 Hit@10：100%

组大小为 3 时：

- 平均目标召回@10：63.6%
- 全部目标 Hit@10：27.3%
- 任一目标 Hit@10：100%

RAGAS 风格混合推理评估，30 道文本题 + 5 道表格题：

- 目标在向量 top 10 中：97.1%
- 目标在重排序 top 3 中：94.3%
- 目标在扩展上下文中：94.3%
- 目标表格在上下文中：80%
- 平均忠实度：4.89 / 5
- 平均答案相关性：4.69 / 5
- 平均上下文精确度：3.91 / 5
- 平均参考覆盖率：4.54 / 5
- 平均端到端延迟：6.75 秒
- 最大延迟：12.97 秒

文本题表现优于表格题。表格题答案相关性与参考覆盖率较低，因为模型有时未能正确读取表格原始文本，即使目标表格已存在。

添加相邻上下文裁剪后，非表格 30 题 RAGAS 风格运行结果：

- 目标在向量 top 10 中：100%
- 目标在重排序 top 3 中：96.7%
- 目标在扩展上下文中：96.7%
- 平均忠实度：4.90 / 5
- 平均答案相关性：4.83 / 5
- 平均上下文精确度：3.97 / 5
- 平均参考覆盖率：4.67 / 5
- 平均端到端延迟：7.15 秒
- 最大延迟：15.28 秒
- 通过率：93.3%

另一次 10 题表格专项运行：

- 目标表格在上下文中：80%
- 平均忠实度：4.40 / 5
- 平均答案相关性：4.90 / 5
- 平均上下文精确度：3.70 / 5
- 平均参考覆盖率：4.60 / 5
- 平均端到端延迟：11.49 秒
- 通过率：70%
- 边界通过率：10%
- 失败率：20%

### MSFT FY2025 已解析表格回归（10 题）

测试集：`main/common/msft_fy2025_parsed_table_test_questions.json`

测试表格：MSFT Chunk Studio 工作区中前 10 张 VLM 解析的 Item 7 MD&A 表格（第 33–40 页）。

#### 旧合并重排序流水线

设置：文本 top 10 + 表格 top 5 合并为单一重排序池，重排序 top 3。

| 指标 | 结果 |
| --- | --- |
| 表格向量 Hit@5 | 10/10 |
| 目标表格在重排序 top 3 中 | 1/10 |
| 目标表格在最终上下文中 | 6/10 |
| 答案正确性（人工检查） | 8/10 |

主要失败模式：表格摘要被检索到，但重排序仅选文本块。许多答案仍看起来正确，因为 MD&A 叙事文本或 `table_refs` 碰巧含相同数字。

股息问题（`table_005`）失败，因为重排序选了含错误 `table_refs` 的股息相关文本，从未注入小型股息表 Markdown。

#### 新双路径流水线

设置：文本 top 10 重排序 top 3 + 表格 top 5 阈值 0.75，独立路径。

| 指标 | 结果 |
| --- | --- |
| 表格向量 Hit@5 | 10/10 |
| 目标表格通过阈值 | 9/10 |
| 目标表格在最终上下文中 | 10/10 |
| 答案正确性（人工检查） | 10/10 |

说明：

- 稀释 EPS 的 `table_006_merged` 得分 0.703，在阈值 0.75 被过滤，但仍通过文本块引用正确作答。
- 此运行平均端到端延迟约 22 秒（Fireworks 对话）；重排序占主导延迟。
- 评估脚本中自动数值匹配器因未一致规范化「million」单位而低估正确性。

评估输出：

- `main/inference/msft_fy2025_table_test_eval.json` — 旧流水线
- `main/inference/msft_fy2025_table_test_eval_v2.json` — 双路径流水线
- 对应 `.jsonl` 日志，含每题流水线路径与延迟

### MSFT FY2025 混合推理（15 题：10 表格 + 5 文本）

测试集：`main/common/msft_fy2025_mixed_15_inference_test.json`

工作区：`data/chunk_studio/1779921176-msft-fy2025-10-k-8d505c867d/`

Haiku 重排序 + Sonnet 答案的代表性 5 题运行：

| 设置 | 数值正确性 | 平均延迟 |
| --- | --- | --- |
| Haiku 重排序 + Haiku 答案 | 4/5 | ~26s |
| Haiku 重排序 + Sonnet 答案 | 3/5 | ~9.4s（RAGAS 回放） |

说明：

- 较低延迟主要来自更快的 Haiku 重排序与更短 Sonnet 答案，而非跳过检索。
- Fireworks 嵌入不可用时，RAGAS 风格评估可回放缓存向量命中；完整端到端延迟与检索测量请使用 `--no-replay`。
- 阈值 0.75 下的主要表格失败是阈值/过滤问题，而非重排序遗漏。

评估产物：

- `main/inference/msft_fy2025_mixed_5_anthropic_eval.json`
- `main/inference/msft_fy2025_mixed_5_ragas_style_eval.json`
- `main/inference/msft_fy2025_mixed_15_inference_results.json`

## 推理延迟

当前推理路径记录以下延迟：

- 加载块与表格资产
- 向量搜索（共享查询嵌入 + 文本/表格 DB 查询）
- 文本重排序
- 上下文扩展
- 最终答案生成
- 总时间

MSFT 10 题双路径评估（Fireworks 对话）中，平均端到端延迟约 22 秒，重排序通常是最慢阶段。

MSFT 混合问题上使用 Anthropic Haiku 重排序 + Sonnet 答案时，重排序降至约 2–4 秒，短评估批次总延迟约 8–12 秒，取决于上下文大小与回放模式。

可通过以下方式降低延迟：

- 对极高置信度文本 top1 检索跳过重排序。
- 使用更小/更快的重排序模型。
- 当选中块已较长时减少上下文扩展。
- 裁剪相邻块上下文（前/后邻居现已实现）。
- 缓存重复问题的嵌入。
- 在服务器进程中内存缓存表格查询与块元数据。
- 离线运行 VLM 解析，使推理从不等待图像解析。

## 已知弱点

当前设计在文本检索上表现良好，在已解析表格 QA 上远优于旧合并重排序路径，但仍存在弱点：

- 表格 QA 质量依赖离线 VLM 解析覆盖率；未解析表格仍回退到弱扁平化文本。
- 未解析或弱链接表格仍可通过文本块 `table_refs` 进入上下文，即使表格摘要检索失败或低于阈值。
- `table_similarity_threshold` 需要校准；过高会丢弃有用表格，过低会引入噪声。
- 部分表格引导文本问题需要表格内容，但可能被评估为文本问题。
- 若组为随机相邻块，跨块生成评估问题可能有噪声。
- Exhibit 章节产生低价值列表块。
- 部分章节/子章节边界因字体检测局限而不完美。
- 文本重排序 top 3 可能排除对宽泛问题有用的次要叙事证据。
- 若章节含许多 loosely 相关子章节，同章节邻居扩展可能引入噪声。
- 相邻块裁剪减小提示规模，但若答案依赖邻居块中部细节可能丢失信息。
- `table_refs` 附加可能指向错误解析表格（如小型/孤立表格，股息表链接到附近无关表）。
- 衍生表格问题（如百分比）在分母表未通过检索或扩展链接时失败。
- 章节序言有帮助，但并非每个章节都有有意义序言。

## 推荐生产推理策略

推荐推理策略为：

1. 查询嵌入一次。
2. **并行**检索文本 top 10 与表格摘要 top 5。
3. **仅对文本**重排序以选择 top 3 锚点块。
4. 用 `table_similarity_threshold` 过滤表格摘要命中（MSFT 风格申报起始 0.70；在标注查询上校准）。
5. 保守扩展文本锚点块：
   - 始终包含章节序言引用。
   - 包含同一文本单元前/后块，但在最终提示中裁剪相邻块。
   - 启用时包含同章节前/后块，但在最终提示中裁剪相邻块。
   - 永不自动跨章节边界。
6. 加载表格 Markdown：
   - 所有通过阈值的表格摘要命中，以及
   - 扩展文本块附加的任何 `table_refs`。
7. 对表格 ID 去重，格式化为 `[Table: id | section]`。
8. 若查询含多个独立子句，考虑检索前查询分解。

此策略保持文本与表格检索独立，避免表格摘要被文本重排序丢弃，同时仍允许叙事上下文通过文本锚点进入。

## 未来改进

最有价值的后续改进包括：

- 在标注表格问题上按申报文件/嵌入模型校准 `table_similarity_threshold`。
- 改进小型/孤立表格的 `table_refs` 链接（股息表、页断附近附注表）。
- 将表格行规范化为结构化记录以进行确定性算术检查。
- 改进评估生成，使跨块问题仅使用连贯的同章节组。
- 为明确多跳问题添加查询分解。
- 改进 subtle 标题与财务报表附注的子章节检测。
- 添加上下文预算器，按 token 预算选择序言、同一文本单元邻居、同章节邻居与表格。
- 将 VLM 解析 + 表格摘要索引默认接入 Chunk Studio 处理流水线。
- 添加服务端缓存以实现低延迟重复推理。
- 对视觉复杂表格，可选在 VLM Markdown 之外将表格裁剪图传入多模态答案步骤。

## Agent 层（编排，不替代上文 RAG 推理）

### 端到端（用户一问）

```text
[离线] PDF → chunking → embedding → vectors.db（+ 可选 table summary DB）
[每轮] session → build_agent_memory → AgentExecutor 循环（约 6 步）
         sql: Text-to-SQL → 校验执行 → 失败时最多 3 次 correct_sql
         rag: 可选 ticker/fiscal_year filter → 向量+BM25 检索 → 文本 rerank → 扩上下文 → answer LLM
         observation JSON（含 scope_filters）→ scratchpad；append_turn；窗口溢出 fold → summary
```

| 入口 | 代码 |
|------|------|
| HTTP | `chunk_studio/agent_bridge.py` |
| CLI | `main/agent/agent.py` |

**路由：** 无独立分类器，每步 LLM tool calling。

**RAG scope：** `rag` 工具支持可选 `ticker`、`fiscal_year` 参数。`system_prompt.py` 的 `RAG_SCOPE_RULES` 指导模型何时传递（sql 之后、用户指定 scope、跨公司对比）。检索在 `load_chunks()` 应用 filter；`question` 只承载主题/章节表述。

**记忆：** `agent_memory.py` · `MEMORY_DESIGN.zh.md`

**错误：** SQL 在 `text_to_sql.py` 内重试；Agent 改问/换工具靠 prompt · `agent/README.md`

**未实现：** Agent/RAG 检索前 query decomposition（评测脚本另有 generate question）。

## 结论

系统使用精确、元数据丰富的文本块进行检索，并将上下文拼接延迟到推理阶段。这对 10-K 申报是正确权衡：小块检索更好，而元数据驱动扩展恢复答案生成所需的周围上下文。

当前系统最强部分是文本检索与文本 grounding 作答。流水线停止强制表格摘要经过文本重排序后，表格 QA 显著改善。推荐生产形态为：

```text
text path:   可选 ticker/year filter -> vector top10 + BM25 top10 -> text rerank top3 -> expand neighbors/refs
table path:  可选 ticker/year filter -> summary vector top5 -> threshold filter -> inject VLM markdown
answer:      preamble + text + [Table: id | section] markdown blocks
agent rag:   rag(question, ticker?, fiscal_year?) -> run_pipeline filter -> 双路径如上
```

稳健表格 QA 仍依赖离线 VLM 解析质量、阈值校准与正确 `table_refs`，但双路径设计符合金融 10-K 证据的实际行为：叙事与表格相关，但不应在同一重排序池中竞争。

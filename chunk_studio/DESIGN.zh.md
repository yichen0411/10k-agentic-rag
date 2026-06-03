# Chunk Studio 设计

英文版：`DESIGN.md`

本文档描述 Chunk Studio 当前的产品设计：上传 10-K PDF，生成 sections/chunks/assets，可视化表格与图表，并（可选）对分块结果进行问答。

核心产品目标是 **正确的视觉表格区域**，而非完美的单元格解析。财务 10-K 表格往往通过版式传递语义（缩进、粗体标题、小计、灰色条带、跨页延续）。对本产品而言，截图必须首先正确；解析出的行/单元格属于次要元数据。

实现主要位于：

- `chunk_studio/server.py` — FastAPI 后端、视觉表格检测器、裁剪渲染
- `chunk_studio/static/index.html` — 单页 UI
- `main/chunking/` — 原有分节、资产提取、分块构建流水线，由 Chunk Studio 复用

更广泛的 RAG 分块与推理设计，请参阅 `main/CHUNKING_AND_INFERENCE_DESIGN.md`。

---

## 高层架构

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

Chunk Studio 是在现有 `main` 流水线之上的轻量产品层：

1. **分节（Sectioning）** — 目录引导的 TOC + 正文标题检测（`toc_guided_section_probe.py`）
2. **资产提取（Asset extraction）** — PyMuPDF `find_tables()` + 图像提取（`section_asset_extractor.py`）
3. **分块构建（Chunk building）** — 仅文本的 RAG 分块，含表格/图像引用（`rag_chunk_builder.py`）
4. **视觉表格服务（Visual table serving）** — API 调用时的列对齐检测器（`chunk_studio/server.py`）
5. **离线 VLM 表格解析（Offline VLM table parse）** — 表格裁剪 → markdown + 摘要（`main/chunking/vlm_table_parse.py`）
6. **问答推理（Q&A inference）** — 文本 + 表格摘要双路径检索（`main/inference/text_vector_rag_inference.py`）

重要分工：

| 阶段 | 职责 |
| --- | --- |
| 离线处理 | sections、原始 assets、chunks |
| API 时视觉层 | 表格区域检测、合并、为 UI 渲染裁剪 |
| 离线 VLM 解析 | 表格裁剪 PNG → markdown + 摘要，存入 `assets.json` |
| 问答 | 双路径检索：文本 rerank + 表格阈值 |

视觉检测器在调用 `/api/files/{id}/assets` 或表格裁剪端点时运行。检测器变更后 **无需** 重新处理 PDF。

VLM 解析是离线/批处理步骤（CLI 或未来的 process 钩子）。解析结果可在右侧 **Parses** 标签页中查看。

---

## 处理流程

用户点击 **Process** 时，后端运行三个步骤，并将进度写入 `metadata.json`：

| 步骤 | 状态键 | 执行内容 |
| --- | --- | --- |
| Reading structure | `sectioning` | 解析可见 TOC，定位 Item 章节，检测子章节 |
| Finding assets | `extracting_assets` | 对表格/图像运行 `build_asset_payload()` |
| Writing chunks | `building_chunks` | 对文本分块运行 `build_rag_payload()` |

Chunk Studio 默认处理 **不会** 构建 embedding。问答为可选功能，需要构建向量索引。

工作区布局：

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

## 设计原则：视觉优先的表格

### 为何单独使用 `find_tables()` 不够

PyMuPDF 的 `page.find_tables()` 有用，但在 10-K 财务表格上很脆弱，因为：

- PDF 没有真正的表格语义 — 只有文字、坐标、线条、矩形
- 许多财务表格没有完整网格线
- 列通过空白对齐，而非边框
- 一个逻辑表格常被检测为许多单行碎片
- 跨页延续在 PDF 中没有显式链接
- 章节/子章节归属可能将同一视觉表格拆到多个元数据桶中
- 换行标签、缩进、粗体子标题和小计会破坏朴素的行/单元格分组

因此旧流程是：

```text
find_tables() -> fragmented bboxes -> merge heuristics -> crop
```

这会产生过多小裁剪，并丢失跨页上下文。

当前流程是：

```text
words + geometry -> visual table region -> crop
find_tables() -> metadata / fallback only
```

### 为何视觉推断是长期正确方向

财务表格并非总是扁平矩阵。版式常编码依赖关系：

- 缩进 = 父子行层级
- 粗体 = 章节标题或小计
- 灰色背景 = 分组块
- 水平线 = 小计边界
- 空白间距 = 逻辑断点
- 跨页版式 = 同一报表的延续

对于复杂表格问答，可靠证据是 **表格图像**，而非扁平化的单元格文本。解析出的单元格有助于检索和粗略 grounding；最终推理应使用表格裁剪及附近叙述上下文，理想情况下在回答时使用多模态模型。

Chunk Studio 当前在检测阶段 **仅使用几何启发式**。处理过程中 **不会** 对每一页运行 VLM。

---

## 表格检测：三层结构

Chunk Studio 当前结合三层检测。

### 第 1 层 — 原始提取（`section_asset_extractor.py`）

处理过程中：

- PyMuPDF `find_tables()` 提取表格碎片
- 同页碎片合并
- 跨页相连表格合并
- 表格/图像归属到最近的 section/subsection

这会生成 `assets.json`，条目如 `table_001`、`table_group_137` 等。

### 第 2 层 — 解析表格的视觉合并（API 回退）

若列对齐检测失败，API 回退到按几何合并解析碎片：

**同页合并（`visual_table_group`）**

- 同一页
- 左边缘与宽度相近
- 列数相近
- 碎片间垂直间距小
- **不** 要求同一 subsection（subsection 归属对财务行不可靠）

**解析组的跨页合并**

- 前一裁剪接近页底
- 下一碎片接近页顶
- 宽度/列兼容
- 吸收下一页的延续行

### 第 3 层 — 列对齐视觉区域（主路径，API 时）

这是 UI 当前使用的主要检测器。

它 **不** 调用任何模型。仅使用 PyMuPDF 词坐标。

---

## 列对齐检测器

### 输入

对 PDF 每一页：

```python
page.get_text("words")
```

每个词具有 `(x0, y0, x1, y1, text, ...)`。

简化行示例：

```text
row 1: Assets                     $ 128,335   95,466
       left_anchor=42             anchors=[456, 510, 570]

row 2: Accounts receivable                 43,052   56,924
       left_anchor=42             anchors=[510, 570]

row 3: Inventory                           1,234    1,100
       left_anchor=42             anchors=[510, 570]
```

### 步骤 1 — 将词聚类为视觉行

- 按垂直中心再按 x 排序
- 将 y 中心相差 <= 3.5pt 的词归为同一行

### 步骤 2 — 计算行特征

对每一行：

- `bbox`
- `text`
- `numeric_anchors`：类数字 token 的右边缘（`x1`），四舍五入到 6pt
- `left_anchor`：首个 token 的 x0 四舍五入
- `numeric_count`
- `is_tableish`

一行被视为类表格行（table-ish），若：

- 具有 >= 2 个数字 anchor，或
- 具有 >= 4 个词、>= 1 个数字 anchor，且水平跨度 > 220pt

类数字 token 匹配如下模式：

- `128,335`
- `(2,625)`
- `$`
- `%`
- `-`, `—`

### 步骤 3 — 查找连续对齐的运行段

扫描连续的类表格行。在 `_same_alignment_run()` 为真时扩展运行段：

- 行间垂直间距：`-3 .. 42` pt
- 数字 anchor 重叠 >= 2，或
- anchor 重叠 >= 1 且左 anchor 相差在 10pt 内

接受该运行段，若：

- 长度 >= 3 行，或
- 长度 >= 2 行且最大数字计数 >= 3

### 步骤 4 — 垂直扩展区域

找到运行段后，向上/下扩展：

**向上扩展** 用于标题/单位行，若：

- 间距 <= 34pt
- 行较宽（> 180pt）
- 文本像标题：`(in millions)`、`year`、`june`、`september`、`ended`，或包含数字

**向下扩展** 当下一行：

- 间距 <= 34pt
- 仍为类表格行或含数字

最终 bbox 使用整页宽度：

```text
x0 = 0
x1 = page width
y0 = first used row top - 18pt
y1 = last used row bottom + 18pt
```

### 步骤 5 — 合并同页区域

合并同页相邻区域，若：

- 垂直间距 <= 24pt
- 数字 anchor 重叠 >= 2

### 步骤 6 — 合并跨页视觉区域

将第 N 页的区域 A 与第 N+1 页的区域 B 链接，若：

- A 接近页底（`bbox.y1 > 620`）
- B 接近页顶（`bbox.y0 < 180`）
- 数字 anchor 重叠 >= 2
- 最多吸收一个延续页

这可避免将多个无关页吞并成一个巨大区域。

### 步骤 7 — 与解析表格回退组合

对每个检测到的视觉区域：

- 查找重叠的解析视觉组（`overlap > 0.25`）
- 从最佳重叠解析表格继承 section 元数据
- 输出为 `visual_region_XXX`

然后追加尚未被覆盖的旧解析视觉组（`overlap > 0.72`）。

若检测器抛出异常，API 仅回退到解析表格视觉组。

---

## 检测信号优先级

| 信号 | 作用 |
| --- | --- |
| 连续行上重复的数字列右边缘 | **主要** |
| 稳定的标签左边缘 | 强次要 |
| 稳定的行节奏 / 小垂直间距 | 强次要 |
| 多列数字密度 | 次要 |
| 表格上方的标题/单位行 | 扩展辅助 |
| 跨页底/顶邻近 + 相同 anchor | 延续辅助 |
| 水平/垂直线、灰色填充 | 仅可能的未来加分项 |
| Subsection 元数据 | **不** 用于视觉合并 |
| 仅数字密度 | 本身太弱 |
| 单独 `find_tables()` bbox | 对 UI 裁剪过于碎片化 |

关键洞察：**连续行之间的列对齐一致性是 10-K PDF 中最可靠的表格信号**。

---

## 输出 Schema

### 单页视觉区域

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

### 跨页解析组

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

### 返回 UI 的 API 摘要结构

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

Assets API 还报告：

```json
{
  "counts": {
    "visual_tables": 101,
    "visual_detector": "column_alignment"
  }
}
```

---

## 表格裁剪渲染

端点：

```text
GET /api/files/{file_id}/tables/{table_id}/crops/{crop_idx}.png
```

渲染规则：

- 使用 PyMuPDF pixmap 从源 PDF 裁剪
- 优先可读页切片，而非紧贴检测框
- 缩放：`Matrix(2.5, 2.5)`
- PNG 响应为 no-cache

### 单页 / 单裁剪表格

```text
clip = full page width
y0 = bbox.top - 76/96pt padding
y1 = bbox.bottom + 76/96pt padding
```

### 多页表格

对具有多个裁剪的 `bbox_by_page`：

- 首页裁剪：从 `bbox.top - 220pt` 到页底
- 末页裁剪：从页顶到 `bbox.bottom + 180pt`
- 中间页：整页

因此像第 63 页底部 + 第 64 页顶部这类跨页表格，现在会显示有用上下文，而非单行细条。

---

## UI 设计说明

当前 UI 目标：

- 简洁、不花哨的布局
- 层级化的 chunk 与 asset 树
- 仅显示当前树节点标签，而非完整重复路径字符串
- chunk 文本在树叶子节点内联展开
- 右栏显示表格裁剪预览，而非 HTML 表格
- 预览面板可拖拽列宽调整
- 处理过程中带步骤器/进度的活动日志
- 选择表格时更新预览，但不折叠 asset 树

表格预览行为：

- 每个裁剪为大型可滚动/拖拽平移的图像区域
- 重新加载后图像 URL 含 cache-bust 查询参数
- 无自定义滚动滑条；仅原生/触控板滚动

问答使用 `main/CHUNKING_AND_INFERENCE_DESIGN.md` 中记录的双路径推理流水线：

```text
text path:   top10 -> text rerank top3 -> expand preamble/neighbors/refs
table path:  summary top5 -> similarity threshold (default 0.75) -> VLM markdown
answer:      preamble + text + [Table: id | section] blocks
```

当 `.env` 按推荐分工配置时，各阶段模型提供商：

| 阶段 | 提供商 |
| --- | --- |
| Query embedding | Fireworks |
| Text rerank | Anthropic（默认 Haiku） |
| Answer | Anthropic（默认 Sonnet） |

设置 `ANTHROPIC_API_KEY` 时，聊天不使用 Fireworks。

Chunk Studio `/api/files/{id}/ask` 参数包括：

- `vector_top_k`（默认 10）
- `rerank_top_n`（默认 3，仅文本）
- `table_vector_top_k`（默认 5）
- `table_similarity_threshold`（默认 0.75）

需要工作区索引：

- `{workspace}/index/vectors.db` — 文本 chunks
- `{workspace}/index/table_vectors.db` — VLM 表格摘要

存在 `FIREWORKS_API_KEY` 时，Process 过程中构建向量索引。表格摘要索引目前在 VLM 解析后为独立构建步骤。

---

## 当前 VLM 的使用范围

**表格区域检测不使用 VLM。** 检测 100% 基于 PyMuPDF 词的本地几何。

**离线表格解析使用 VLM。** 对 `assets.json` 中选定的表格，Chunk Studio 可渲染裁剪 PNG 并调用 VLM 生成：

- `vlm_parse.markdown`
- `vlm_parse.summary`

这些字段用于：

- **Parses** 检查器标签页
- 问答时使用的表格摘要向量 DB

脚本：

- `main/chunking/vlm_table_parse.py`
- `main/chunking/build_table_vector_db.py`

各阶段多模态模型的推荐使用：

| 阶段 | 是否使用 VLM？ |
| --- | --- |
| 检测表格区域 | 否 — 每页太慢/太贵 |
| 渲染表格裁剪 | 否 |
| 离线将表格裁剪解析为 markdown | 是 — 批处理/离线 |
| 检索候选表格 | 否 — 嵌入 `vlm_parse.summary`，而非图像 |
| 回答复杂表格问题 | 是 — 注入 VLM markdown；可选未来步骤：将裁剪图像传给多模态回答模型 |
| 恢复层级/缩进语义 | 是 — VLM markdown 比扁平化行更好地保留版式 |

推荐推理模式：

```text
retrieval -> text top10 + table-summary top5 (parallel)
text       -> rerank top3 anchors + neighbor expansion
table      -> threshold filter -> VLM markdown
answer     -> preamble + text + [Table: id | section]
citation   -> chunk_id / table_id
```

---

## 已知限制与边界情况

当前检测器比解析优先的裁剪好得多，但仍是启发式的。产品标准仍然是：**截图必须正确**。以下各节记录现有合并逻辑及仍失败的边界情况。

### 当前已有的合并

共有三层，而非零：

| 层 | 位置 | 合并内容 |
| --- | --- | --- |
| 解析同页合并 | `section_asset_extractor.py` | 同页单行 `find_tables()` 碎片 |
| 解析跨页合并 | `section_asset_extractor.py` | 表格触及底/顶且列/宽度匹配时的相邻页 |
| 视觉区域合并 | `chunk_studio/server.py` | 列对齐行运行段；同页相邻区域；数字 anchor 匹配的跨页区域 |

这些层 **保守**。它们改善许多 10-K 表格，但不保证一个逻辑表格对应一个 UI 条目。

### 边界如何确定（回顾）

边界 **不** 基于网格线或 VLM。它们来自：

```text
words -> visual rows -> numeric column anchors -> consecutive aligned runs -> expanded bbox
```

区域被接受，当运行段具有：

- >= 3 行对齐的类表格行，或
- >= 2 行且数字 token 总数 >= 3

同页区域合并要求：

- 垂直间距 <= 24pt
- 数字 anchor 重叠 >= 2

跨页视觉区域合并要求：

- 前一区域接近页底（`y1 > 620`）
- 下一区域接近页顶（`y0 < 180`）
- 数字 anchor 重叠 >= 2
- 最多吸收一个延续页

这意味着边界实质上是 **数字对齐行的块**，而非财务报表表格的完整语义范围。

### 仍失败的边界情况

#### 1. 一个逻辑表格拆成两个 UI 条目

仍会在以下情况发生：

- 两个表格块之间有空行、散文句子或非数字子标题
- 行运行段间垂直间距超过约 42pt
- 第二块具有不同数字 anchor（小计行、缩进偏移、列数变化）
- 同页两个检测区域相距 > 24pt
- 因覆盖不完整，同时保留新的 `visual_region_*` 与旧的 `visual_table_*` / `table_group_*` 回退

UI 症状：两个相邻表格条目，视觉上属于同一张报表。

#### 2. 跨页仅标题尾部 + 下一页完整表格主体

这是当前最弱的情况。

示例：

```text
page N bottom:
  "Designated as Hedging Instruments"
  "Foreign exchange contracts purchased"   (no values)

page N+1 top:
  full numeric rows for the same table
```

失败原因：

- 标题/分类行常 **零或一个** 数字 anchor
- 可能不符合 `is_tableish`
- 即使被检测到，跨页合并目前要求 **anchor 重叠 >= 2**
- 仅标题尾部无法满足该规则
- 解析的 `find_tables()` 可能完全漏掉标题行，仅检测到第 N+1 页主体行

UI 症状：第 N 页标题/上下文缺失，或标题与主体显示为两个无关表格。

目前 **尚无** 专门的仅标题延续规则。

#### 3. 页底分类标题，下一页数据行

与情况 2 相关，常见于衍生工具/披露表格：

```text
page N bottom:
  "Not Designated as Hedging Instruments"

page N+1 top:
  "Foreign exchange contracts purchased" 15,214 | 7,167
  ...
```

若分类标题是第 N 页最后一项，且首批数据行从第 N+1 页开始，合并取决于碎片检测的运气。视觉裁剪可能显示页底上下文，但表格仍可能表示为分离条目或不完整组。

#### 4. 数字列极少的表格

多为文本列、比率表，或数字稀疏的标签密集表：

- 可能永远达不到 `is_tableish`
- 可能不产生视觉区域
- 可能仅以微小解析碎片存在

#### 5. 数字密集的散文被误判为表格

页面上真实表格外有许多对齐数字（脚注、行内统计、列表式散文）时，偶尔会产生误报区域。

#### 6. 超过一个延续页的多页表格

跨页合并对视觉区域 **故意限制为一个延续页**。 spanning 3+ 页的表格可能显示为多个链接或未链接区域。

#### 7. 由缩进编码的层级

父子行关系、粗体小计、灰色条带和换行标签保留在 **裁剪图像** 中，但未恢复为结构化层级。扁平的行/列元数据仍丢失语义依赖。

#### 8. Section/subsection 归属噪声

底层提取器可能将每个财务行归属到不同 subsection。视觉合并故意忽略 subsection 以进行裁剪，但元数据 `header_path` 在截图正确时仍可能看起来错误。

#### 9. 重复或重叠的表格条目

因为 API 组合了：

- 新的 `visual_region_*` 检测器输出，以及
- 未覆盖的回退解析组

表格列表可能包含邻近或相同内容的重叠条目。

### 当前的跨页处理

跨页通过两种方式处理：

**检测 / 分组**

- 通过底/顶几何 + 列数的解析 `table_group_*`
- 通过底/顶几何 + anchor 重叠的视觉区域跨页合并
- 吸收下一页延续行的解析组尾部合并

**渲染**

即使 bbox 很紧，裁剪也会添加上下文：

- 首页：从 `bbox.top - 220pt` 到页底
- 下一页：从页顶到 `bbox.bottom + 180pt`

因此跨页 **截图** 可能看起来合理，而 **分组** 仍不完整。分组与裁剪不是同一问题。

### 示例：有改进但未解决

激励近期工作的衍生工具表格案例：

- 第 63 页底部：标题 + 首批行
- 第 64 页顶部：延续行

当前状态：

- 优于单行细条裁剪
- `table_group_137` 可合并部分延续行
- 仍不保证将仅标题尾部与下一页完整主体统一
- 仍可能因 anchor 和检测运气看到分离条目

### 针对上述问题的计划修复

最高价值下一批规则：

1. **仅标题的跨页延续** — 部分实现：当页尾/页首不含散文时，跨页合并可桥接；延续裁剪会从首页 prepend `header_crop`。
2. **同页间距桥接** — 已实现：当间距仅含表格内部行（小计、分类子标题、列标题）或完全无文本时，合并相邻区域。
3. **UI/API 去重** — 当视觉区域已覆盖 >= 72% 重叠时，抑制回退解析组。
4. **处理时持久化视觉区域** — 避免每次 API 调用重新计算略有不同的结果。

### 现已实现的合并规则

**同页**

合并区域 A + 区域 B，当它们之间的垂直间距：

- **完全没有** 词，或
- 仅含 **表格内部行**，如小计、`Changes in Fair Value...`、`Total debt investments`、列标题碎片，或
- 仍满足小间距的旧 anchor 重叠启发式。

这修复了 **MSFT 第 60 页** 等情况，此前一张公允价值表在内部小计/子标题块处被拆分。

**跨页**

合并当：

- 前一区域接近页底且下一区域接近页顶，且
- 页尾/页首不含散文，或 anchor 重叠，或前一页以仅标题行结束且下一页以数字行开始。

**解析的标题 carry-over**

对 `crop_idx > 0`，PNG 渲染为：

```text
header band from first page/table top
+
continuation body crop
```

因此下游表格图像解析总能看到列标题，即使表格主体在后续页或后续区域延续。

---

## 验证快照（MSFT FY2025）

开发期间使用的 MSFT 工作区：

| 指标 | 数值 |
| --- | --- |
| Raw parsed tables | 97 |
| VLM-parsed tables (first MD&A batch) | 10 |
| Text chunks | 204 |
| Text vectors | 204 |
| Table summary vectors | 10 |
| Column-alignment visual tables served | 101 |

10 道解析表格问题的表格 QA 回归：

| 流水线 | 答案正确率 | 目标表格在上下文中 |
| --- | --- | --- |
| Old merged rerank | 8/10 | 6/10 |
| Dual-path + threshold 0.75 | 10/10 | 10/10 |

评估产物：

- `main/inference/msft_fy2025_table_test_eval_v2.json`
- `main/inference/msft_fy2025_table_test_eval_v2.jsonl`

视觉表格由 `/api/files/{file_id}/assets` 实时提供，无需重新处理。VLM 解析结果位于 `assets.json`。

---

## Agent Q&A（`/agent`）

独立页面：**http://127.0.0.1:8010/agent**（Chunk Studio 顶栏与文件卡片也有入口）。

基于 LangChain（`main/agent/langchain_agent.py`），工具：

| 工具 | 作用 |
|------|------|
| `sql` | 只读 SQLite 财务库（`data/financials.db`，内部 Text-to-SQL） |
| `rag` | 当前 workspace 的 `index/vectors.db` + `assets.json` 做 10-K 检索问答 |
| `send_email` | 可选：把最终答案发到用户邮箱（SMTP） |

### API

- 流式：`POST /api/files/{file_id}/agent/trace/stream`（NDJSON），经 `agent_bridge.py`。
- Agent / Memory 流水线：`main/agent/README.md`、`MEMORY_DESIGN.zh.md`。

### 前置条件

- 文件 `status=ready`，且磁盘上有 **`index/vectors.db`**（处理时勾选 **Build embeddings**）。
- `.env`：`ANTHROPIC_API_KEY`、`FIREWORKS_API_KEY`（建索引时用 embedding）。

### 可选：发邮件（写在 `.env`，不是 `.env.example`）

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=你的Gmail@gmail.com
SMTP_USER=你的Gmail@gmail.com
SMTP_PASSWORD=Gmail应用专用密码
SMTP_USE_TLS=true
```

- **App password**（应用专用密码）≠ 登录密码 ≠ Passkey。
- `SMTP_FROM` / `SMTP_USER` 填同一个 Gmail。

详见 `main/agent/README.md`（路由逻辑、SQL/RAG 无结果时的行为）。

### 路由说明

每一步由模型根据工具说明和上一步 observation 决定（见 `main/agent/README.md`）。

---

## 如何运行

从仓库根目录：

```bash
cd agentic-rag-takehome-fw
uvicorn chunk_studio.server:app --host 127.0.0.1 --port 8010
```

打开：

```text
http://127.0.0.1:8010/          # Chunk Studio
http://127.0.0.1:8010/agent     # Agent 问答
```

上传 PDF，点击 **Process**（需要 Agent 时请勾选 embeddings），检查 chunks/assets/tables，或打开 **Agent Q&A**。

环境：

- 仓库根目录 `.env`：`FIREWORKS_API_KEY`、`ANTHROPIC_API_KEY`，发邮件再加 SMTP 变量
- 可选：`ANTHROPIC_RERANK_MODEL`、`ANTHROPIC_CHAT_MODEL`
- 默认端口：`8010`

---

## 推荐的后续步骤

若继续本产品，最有价值的跟进项：

1. 将列对齐检测器从 `server.py` 移到 `main/chunking/visual_table_detector.py`
2. 处理时持久化 `visual_regions.json`，而非在 API 时计算
3. 通过重叠/页/section 将视觉区域关联到 chunks，而非解析碎片引用
4. 在 UI 增加调试叠加模式，显示检测到的行 anchor 与区域 bbox
5. 默认将 VLM 解析 + 表格摘要索引接入 Process 流水线
6. 按每份申报文件的标注表格问题校准 `table_similarity_threshold`（MSFT 混合评估表明 0.65–0.70 可能比 0.75 更安全）
7. 将解析单元格文本保留为辅助元数据，而非表格的主要表示

产品成功标准仍应为：

> **表格的截图必须正确。**

其他一切均为次要。

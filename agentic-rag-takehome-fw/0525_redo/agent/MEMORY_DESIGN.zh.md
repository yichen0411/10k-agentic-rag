# Agent Memory（Pipeline 逻辑）

实现：`0525_redo/agent/agent_memory.py` · 库：`data/agent_memory.db`

## 1. 存储

| 层 | 表 / 字段 | 进入下一轮 prompt 的方式 |
|----|-----------|-------------------------|
| **短期** | `messages`（user/assistant） | `get_history()` → LangChain `chat_history` |
| **工具痕迹** | `tool_artifacts` | **不**进 `chat_history`；仅落库，供 snapshot / 调试 |
| **情景** | `memory_facts` KV | `build_memory_context()` 每轮**全量**注入 episodic 块 |
| **语义** | `memory_semantic` + `sessions.summary` | 按当前 `query` **关键词打分**检索后注入 `memory_context` |

本轮进行中的 tool call / observation 只在 LangChain **`agent_scratchpad`**（不写入 DB，下轮不恢复）。

## 2. 每轮时序（与 agent 衔接）

```text
1. build_agent_memory(session_id, query)
     chat_history  = 本轮提问前的 messages
     memory_context = episodic + 检索后的 semantic/summary

2. langchain_agent.run_langchain_agent(..., chat_history, memory_context)
     system 末尾可追加 ## Session memory（memory_context）

3. append_turn(user, assistant, tool_steps)
     messages += 本轮 Q/A
     memory_facts / memory_semantic / tool_artifacts 写入
     _prune_old_messages() 若超出窗口则 fold → summary

4. save_last_prompt_snapshot(memory_context, chat_history, scratchpad)
     冻结「该轮实际注入」供审计（非 live 重算 retrieval）
```

## 3. 压缩

- **窗口**：`SHORT_TERM_WINDOW_TURNS`（当前测试 **2** 轮）→ `PRUNE_KEEP_MESSAGES = 2 × turns`
- **溢出 fold**：旧 message 格式化为 `- role: text`（每条 content 最多 400 字符），**append** 到 `sessions.summary`；合并后 **>6000 字符只保留尾部 6000** — **无 LLM 摘要**
- **语义笔记**：每轮 `Q: …\nA: …` 模板（截断），最多 50 条，删最旧
- **检索衰减**：`score = keyword_overlap × max(0.5, 1 - 0.02×age_days)`；`RETRIEVAL_MIN_SCORE=0.06`，最多 8 条

## 4. 清理

`clear_session(session_id)`：删 messages / facts / semantic / artifacts / summary 等。

## 5. 与 RAG/SQL 的边界

- `memory_context`：**会话**记忆，不是 filing 向量检索。
- Agent 调 `rag` / `sql` 时传入的 `question` 由模型根据 `chat_history` + `memory_context` 自行改写；**无**独立的 query-rewrite 服务。
- `rag` 另传可选 `ticker` / `fiscal_year` 做申报 metadata hard filter（见 `system_prompt.py` 的 `RAG_SCOPE_RULES`），与 `question` 主题表述分开。

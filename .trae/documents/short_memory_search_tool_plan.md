# 让 short 记忆生效：新增 LLM 可调用的记忆检索工具

## Context（为什么做这个改动）

长期记忆分 core / short 两层，但 **short 层从不进入 LLM 的上下文**：

- 注入端 `load_memory_node`（`app/graph/nodes.py:393-415`）只 `get_all(uid, "core")`，short 被明确排除；
- 没有任何自动升级路径（唯一出口是前端记忆页手动"置顶核心"）；
- short 现有的作用只剩"14 天未更新自动过期 + 超限时优先淘汰"。

等于抽取出来就闲置了。本次改动给 LLM 增加一个**按需调用的记忆检索工具**，让 short（以及按主题定位 core）真正参与对话。

已确定的两个约束（用户拍板）：
1. 走"记忆检索工具"路线（不是自动注入、不是自动晋升）；
2. 工具返回范围：**最近 7 天内更新、最多 10 条**。

保留不动的不变量：`_build_system_prompt` 的 system 前缀结构、`tests/test_prompt_prefix.py` 守卫的前缀缓存稳定性、以及"记忆是用户级数据"的既有隔离语义（项目规则只要求知识库按 team 隔离，记忆不涉及）。

## 实施步骤

### 1. MemoryService 新增通用检索方法
文件：`app/services/memory_service.py`

```python
def search(self, user_id: str, query: str = "", tier: str | None = None,
           since: datetime | None = None, limit: int = 10) -> list[dict]:
    """按关键词模糊检索记忆（key/topic/值），按 updated_at 倒序。"""
```

要点：
- `select(Memory).where(user_id[, tier][, updated_at >= since])`，`order_by(updated_at.desc()).limit(limit)`
- `query` 非空时对 `key / topic / cast(Memory.value, Text)` 做 `ilike`（先转义 `%`、`_`，配 `escape="\\"`）
- **只加参数、不写策略**：7 天 / 10 条的取舍放在工具层，服务层保持通用
- 返回值复用 `list_memory` 的字段形状：`[{key, value, tier, topic, updated_at(ISO)}]`
- 读操作**不调 `touch()`**：避免检索把 short 续命、干扰 TTL/LRU 语义

### 2. 适配器注入 memory_service
文件：`app/mcp/adapter.py`

- `__init__(..., kb_service=None, memory_service=None)`，存 `self.memory_service`
- 原生分支调用改为 `native["func"](args, user_id, self.kb_service, self.memory_service)`
- 同步扩展 6 个既有原生工具签名（`_list_kb_files` / `_get_local_document` / `_get_utc_time` / `_search_knowledge_base` / `_save_document` / `_export_bibtex`）为 `(args, user_id, kb_service, memory_service)`，并更新 L53-58 的签名说明注释
- 不做 `inspect` 动态派发、不引入 services 容器对象——保持"一个统一调用约定"的既有风格

### 3. 注册并实现 `search_memories` 工具
文件：`app/mcp/adapter.py`

- `_native_tool_specs()` 追加一项：
  - name：`search_memories`
  - description 要点：查"当前用户的长期记忆"；当你需要用户偏好 / 研究背景 / 正在进行的项目，或用户提到"我之前说过"时调用；**core 记忆已常驻系统提示，本工具主要用于取回短期（临时）记忆与按主题定位条目**
  - parameters：`query`(string, 可选，关键词)、`tier`(enum: core/short, 可选)；`required: []`
- 实现 `async def _search_memories(args, user_id, kb_service=None, memory_service=None) -> dict`：
  - `memory_service is None` → 返回结构化降级 `{"count": 0, "memories": [], "note": "记忆服务不可用"}`（不抛错，LLM 不会反复重试）
  - `cutoff = now - 7d`；**short 查 `since=cutoff`，core 不传 since**（core 是稳定偏好，被 7 天窗口砍掉只会无谓丢信息）
  - 两次 `asyncio.to_thread(memory_service.search, ...)`，合并去重 → 按 updated_at 倒序 → 截断到 **10 条**
  - 返回：`{"count": N, "note": "tier=short 为临时信息，可能已过期", "memories": [...]}`
  - `value` 取 `value.get("v")`（`Memory.value` 是 JSON dict；旧数据兜底 `str(value)`），单条超 500 字符截断

### 4. main.py 接线（顺序调整）
文件：`app/main.py`（约 L156-163）

把 `memory_service = MemoryService()` 上移到 `MCPToolAdapter(...)` 构造之前，并注入：

```python
memory_service = MemoryService()
mcp_adapter = MCPToolAdapter(mcp_host, tracer,
                             kb_service=kb_service, memory_service=memory_service)
```

`WorkflowContext(..., mcp_adapter, tracer, memory_service)` 传同一个实例（保证 MemoryService 内部的 user 级锁仍共享）。
其它两处 `MCPToolAdapter` 构造点（`tests/test_native_tools.py:34`、`tests/test_tool_executor.py:60`）因新参数有默认值而不受影响。

### 5. 提示词：不改
`_IDENTITY`（`nodes.py:516-518`）是所有请求共享的固定前缀段，改它 = 一次性全局前缀缓存失效。工具 description 已足以引导模型，`tests/test_prompt_prefix.py` 的不变量保持不动。若后续 A/B 观察发现模型不调用，再单独评估。

### 6. 测试
新增 `tests/test_memory_tool.py`（参照 `tests/test_memory.py` 的 `_make_ctx` / `tests/test_memory_bloat.py` 的"拨回 updated_at"手法）：

- `search_memories` 出现在 `schemas_for_llm()` 目录中
- **7 天过滤**：造一条 8 天前的 short + 一条新 short，仅新条返回
- **10 条上限**：造 12 条命中，返回恰 10 条
- **core 不受 7 天限制**：60 天前更新的 core 仍能查到
- `query` 命中 key / topic / value 三种路径；用户隔离（u2 查不到 u1）
- tier 标注与 value 超长截断；`memory_service=None` 时结构化降级

回归：`tests/test_native_tools.py`（目录断言补 `search_memories`）、`test_tool_executor.py`、`test_memory.py`、`test_memory_bloat.py`、`test_prompt_prefix.py`。

## 风险与边界

- 多 agent 图（brainstorm / seminar / deep_research）会自动获得该工具（都经 `ctx.mcp_adapter.call`）：记忆按 `user_id` 隔离，无 team 约束冲突，可接受；若日后要收敛，可按图做工具白名单
- 工具输出会进 checkpoint（仅调用时发生），不触碰 system 前缀缓存
- JSON 列 `ilike` 需 `cast(..., Text)`，PG 下可行
- 空 query + 无记忆 → 返回空结构而非报错

## 验证清单

1. `pytest tests/test_memory_tool.py tests/test_native_tools.py tests/test_tool_executor.py tests/test_memory.py tests/test_memory_bloat.py tests/test_prompt_prefix.py -q`（Postgres 容器 `ra-postgres` 需在运行）
2. 端到端：管理页手动建一条 short 记忆（或先走一轮抽取落一条 short），在普通对话里问"我之前说过我在做哪个项目"，观察是否触发 `search_memories`
3. `tool_call_log` 表 / 追踪页确认 args 与 output（含 tier 标注）
4. 跨会话再问一次，确认能召回；把 short 的 updated_at 拨回 8 天前，确认工具不再返回它（验证 7 天窗口）
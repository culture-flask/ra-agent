from dataclasses import dataclass
import json
import re
import uuid

from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage, ToolMessage

from app.core.events import emit
from app.core.logging import get_logger
from app.services.memory_service import MAX_MEMORIES_PER_USER
from app.graph.state import AgentState

logger = get_logger("graph_nodes")


@dataclass
class WorkflowContext:
    """编排层依赖注入：节点所需的 llm / 知识库等服务，编译时注入。"""
    settings: object
    llm_service: object
    kb_service: object
    mcp_adapter: object = None     
    tracer: object = None
    memory_service: object = None  

# ---------- 长记忆----------
EXTRACT_PROMPT = """从这段对话中提取值得长期记住的用户信息（研究主题、偏好、
项目背景等），输出 JSON：{"memories":[{"key":"snake_case键名","value":"简短值"}]}
如果没有值得记住的信息，输出 {"memories":[]}。只输出 JSON，不要其他文字。"""

KEY_RE = re.compile(r"^[a-z0-9_]{2,32}$")     # 键名白名单：小写/数字/下划线


def _parse_memories(text: str) -> list[dict]:
    """解析 LLM 返回的记忆 JSON（容忍围栏/夹带文字）。解析失败返回空列表。"""
    text = text.strip()
    if "```" in text:                            # 去掉 markdown 围栏
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        data = json.loads(text)                  # 直接解析
    except json.JSONDecodeError:
        # 兜底：从文本里抠出第一个 { ... } 子串再解析（LLM 常夹带说明文字）
        m = re.search(r"\{[^{}]*\}", text, re.S) if False else re.search(r"\{.*\}", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return data.get("memories", []) if isinstance(data, dict) else []


def _review(memory: dict) -> bool:
    """写入前审核（规则版）：键名合法 + 值非空且够长 + 单条长度上限。"""
    key, value = memory.get("key", ""), memory.get("value", "")
    if not KEY_RE.match(key):
        return False
    if not isinstance(value, str) or len(value.strip()) < 4:
        return False
    return len(value) <= 200


# ---------- 自动上下文压缩 ----------
COMPACT_KEEP_ROUNDS = 4          # 压缩后保留的最近轮数
COMPACT_MIN_ROUNDS = 20          # 轮数超过该值触发压缩（>20 轮）

COMPACT_PROMPT = """你是对话总结助手。下面是用户与科研助手的多轮对话历史。
请把它整理成一份简洁的中文总结，保留以下信息：
- 用户的研究主题、关键问题与已获得的结论
- 对话中确认的事实、参数、偏好（后续对话可能继续引用）
- 尚未解决或待跟进的问题
要求：第三人称叙述，按主题组织，200~400 字，不要遗漏重要细节。只输出总结正文。"""


def _round_count(messages: list) -> int:
    """轮数 = 用户提问条数（一对 user+assistant 算一轮）。"""
    return sum(1 for m in messages if getattr(m, "type", "") == "human")


def _estimate_tokens(messages: list) -> int:
    """粗估 token 数：中英文混合约 2 字符/token（宁可早压缩也不爆窗）。"""
    chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    return chars // 2


def _split_keep_and_old(messages: list, keep_rounds: int):
    """按轮数切分：保留最近 keep_rounds 轮，其余归为待总结历史。

    用对象身份切分（消息 id 可能是 None——手工构造/输入转换的消息没有 id）。
    """
    keep, user_seen = [], 0
    for m in reversed(messages):
        keep.append(m)
        if getattr(m, "type", "") == "human":
            user_seen += 1
            if user_seen >= keep_rounds:
                break
    keep.reverse()
    keep_refs = {id(m) for m in keep}
    old = [m for m in messages if id(m) not in keep_refs]
    return keep, old


async def compact_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """自动上下文压缩（图的第一站）：轮数 >20 或 token 估测达上下文上限 80% 时，
    把最近 4 轮之外的历史交给 LLM 总结，后续生成使用「总结 + 最近 4 轮」。

    - 总结存入 conversation_summary（拼进系统提示词），旧消息用 RemoveMessage
      删除（messages 是 add_messages reducer，直接传列表只会追加）
    - 压缩失败（LLM 异常）静默跳过，绝不阻断主对话
    """
    messages = state.get("messages") or []
    if not messages:
        return {}
    # 上下文窗口优先从模型 /models 响应探测（LLMService.context_window_for），
    # 假服务/异常时回退配置默认值
    svc = ctx.llm_service
    if hasattr(svc, "context_window_for"):
        window = int(svc.context_window_for(state["user_id"]))
    else:
        window = int(getattr(ctx.settings, "llm_context_window", 32768))
    rounds = _round_count(messages)
    tokens = _estimate_tokens(messages)
    if rounds <= COMPACT_MIN_ROUNDS and tokens <= window * 0.8:
        return {}

    keep, old = _split_keep_and_old(messages, COMPACT_KEEP_ROUNDS)
    if not old:
        return {}
    try:
        model = ctx.llm_service.get_chat_model(
            state["user_id"], temperature=state.get("temperature"))
        resp = await model.ainvoke([SystemMessage(content=COMPACT_PROMPT)] + old)
        summary = str(resp.content or "").strip()
    except Exception as e:
        logger.warning("context compact failed, skip: %s", e)
        return {}
    if not summary:
        return {}
    # RemoveMessage 按 id 匹配：输入转换/手工构造的消息可能没有 id，先补齐
    for m in old:
        if not getattr(m, "id", None):
            m.id = uuid.uuid4().hex
    compacted = _round_count(old)
    emit("compact", {"compacted_rounds": compacted,
                     "keep_rounds": len(keep) // 2,
                     "total_rounds": rounds,
                     "tokens_estimated": tokens,
                     "window": window})
    return {
        "conversation_summary": summary,
        "messages": [RemoveMessage(id=m.id) for m in old],
    }


# ---------- 知识库路由（LLM 意图判断） ----------
ROUTE_PROMPT = """你是问答路由，负责判断用户提问是否需要查询知识库，以及查哪些库。

可用知识库（JSON 数组，只列用户可见的库）：
{catalog}

判断规则：
- 闲聊、寒暄、数学计算、写代码、通用常识（无需特定资料就能回答）→ 不需要检索
- 问题涉及知识库里的具体内容（术语、资料、论文、实验记录、项目背景等），
  或用户明确要求"查/搜/总结知识库" → 需要检索，并选出最相关的库
- 拿不准时倾向于需要检索，宁可多选一个相关的库也不漏掉

只输出 JSON，不要任何其他文字：
{{"needs_retrieval": true或false, "kbs": [{{"name": "库名", "scope": "public或private"}}]}}
不需要检索时 kbs 为 []。"""


def _parse_route(text: str) -> dict:
    """解析路由 LLM 返回的 JSON（容忍围栏/夹带文字）。解析失败返回空 dict。"""
    text = text.strip()
    if "```" in text:                            # 去掉 markdown 围栏
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _resolve_selected_kbs(kbs: list, picks: list) -> list:
    """把 LLM 选的 (name/scope) 解析回可见知识库对象。

    只允许命中「用户可见」的库：名称不存在、或属于他人私人库的名称一律忽略，
    防止用户（或 LLM 被诱导）越权检索。scope 写错/漏写时按名称兜底匹配。
    """
    selected, seen = [], set()
    for pick in picks or []:
        if not isinstance(pick, dict):
            continue
        name = str(pick.get("name", "")).strip()
        scope = str(pick.get("scope", "")).strip().lower()
        if not name:
            continue
        matches = [kb for kb in kbs
                   if kb.name == name and (not scope or kb.scope == scope)]
        if not matches:                          # scope 漏写/写错 → 名称兜底
            matches = [kb for kb in kbs if kb.name == name]
        for kb in matches:
            if kb.kb_id not in seen:
                seen.add(kb.kb_id)
                selected.append(kb)
    return selected


async def load_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """跨会话读取：把该用户的长记忆载入状态（图的第一站）。"""
    if ctx.memory_service is None:
        return {"memory": {}}
    memory = ctx.memory_service.get_all(state["user_id"])
    emit("memory_load", {"count": len(memory)})
    return {"memory": memory}


async def extract_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """对话结束后抽取值得记住的信息：交给 LLM 从对话中提炼。

    重试耗尽仍失败时静默跳过——记忆抽取是锦上添花，绝不能打挂主对话。
    """
    if ctx.memory_service is None:
        return {"new_memories": []}
    try:
        model = ctx.llm_service.get_chat_model(
            state["user_id"], temperature=state.get("temperature"))
        system = SystemMessage(content=EXTRACT_PROMPT)
        resp = await model.ainvoke([system] + state["messages"][-4:])   # 只看最近几轮
        memories = _parse_memories(str(resp.content or ""))
    except Exception as e:
        logger.warning("memory extract failed, skip: %s", e)
        memories = []
    emit("memory_extract", {"candidates": len(memories)})
    return {"new_memories": memories}


async def save_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """写入前审核 → 落库：审核不过的丢弃，超上限的丢弃。"""
    if ctx.memory_service is None:
        return {}
    saved = 0
    for m in state.get("new_memories", []):
        if not _review(m):
            continue                                   # 审核不通过，丢弃
        if ctx.memory_service.count(state["user_id"]) >= MAX_MEMORIES_PER_USER:
            break                                      # 达到上限，停止
        try:
            ctx.memory_service.set(state["user_id"], m["key"], {"v": m["value"]})
            saved += 1
        except Exception:
            pass          # 记忆写入失败绝不影响主对话（锦上添花原则）
    emit("memory_save", {"saved": saved})
    return {}

def _build_system_prompt(state: AgentState) -> str:
    """组装系统提示词：有检索结果 → 基于知识库作答；没有 → 直接用自己的知识作答。

    自动压缩产生的历史总结放在最前面（替代被压缩掉的旧轮次）。
    """
    parts = []
    if state.get("conversation_summary"):
        parts.append(f"[历史对话总结] {state['conversation_summary']}")
    if state.get("retrievals"):
        parts.append("你是科研助手，基于知识库检索结果回答用户问题，引用时标明来源（public/private）。")
        lines = []
        for r in state["retrievals"]:
            if r.get("type") == "parent":
                # 聚合父块：完整段落，标注来源文件/页码与命中片段数
                loc = r.get("source") or "未知来源"
                if r.get("pages"):
                    loc += f" 第{'-'.join(str(p) for p in r['pages'])}页"
                lines.append(
                    f"[知识库检索结果·上下文段落 ({r.get('scope')} / {r.get('kb_name')} / {loc}，"
                    f"含{r.get('hit_chunks')}个命中片段)] {r['text']}")
            else:
                lines.append(f"[知识库检索结果 ({r.get('scope')} / {r.get('kb_name')})] {r['text']}")
        parts.append("\n".join(lines))
    else:
        parts.append("你是科研助手，根据你自己的知识回答用户问题。")
    if state.get("memory"):
        parts.append(f"[用户记忆] {json.dumps(state['memory'], ensure_ascii=False)}")
    return "\n\n".join(parts)


async def supervisor_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """路由决策（LLM 意图判断）：先看用户有没有可见知识库。

    - 没有可见库 → 无需检索，直接生成
    - 有可见库 → 把「库名 + scope」目录交给 LLM，让它判断本次提问
      是否需要检索、以及选哪几个库（按名称），再做可见性校验后落 state
    - LLM 判断异常/解析失败 → 降级为全部可见库检索（保持 RAG 兜底）
    """
    # 只用"可检索"的库（用户可自行禁用某库参与对话检索）
    kbs = ctx.kb_service.list_queryable_kbs(state["user_id"])
    if not kbs:
        emit("supervisor", {"needs_retrieval": False, "kb_count": 0, "selected": []})
        return {"needs_retrieval": False, "selected_kb_ids": []}

    catalog = [{"name": kb.name, "scope": kb.scope} for kb in kbs]
    selected: list = []
    try:
        model = ctx.llm_service.get_chat_model(
            state["user_id"], temperature=state.get("temperature"))
        prompt = ROUTE_PROMPT.format(catalog=json.dumps(catalog, ensure_ascii=False))
        resp = await model.ainvoke([SystemMessage(content=prompt)] + state["messages"][-4:])
        route = _parse_route(str(resp.content or ""))
        if not route:                         # LLM 没按 JSON 输出 → 无法判断意图
            raise ValueError("unparseable route output")
        needs = bool(route.get("needs_retrieval"))
        if needs:
            selected = _resolve_selected_kbs(kbs, route.get("kbs"))
            if not selected:                  # 说要查但一个库都没选中 → 全查，避免漏检索
                selected = list(kbs)
    except Exception as e:
        logger.warning("kb routing failed, fallback to all visible kbs: %s", e)
        needs, selected = True, list(kbs)
    emit("supervisor", {
        "needs_retrieval": needs,
        "kb_count": len(kbs),
        "selected": [{"name": kb.name, "scope": kb.scope} for kb in selected],
    })
    return {"needs_retrieval": needs,
            "selected_kb_ids": [kb.kb_id for kb in selected]}


def route_supervisor(state: AgentState) -> str:
    return "retrieve" if state.get("needs_retrieval") else "generate"

def route_after_generate(state: AgentState) -> str:
    """LLM 要调工具 → 走 tool_executor；否则结束。"""
    last = state["messages"][-1]
    return "tool_executor" if getattr(last, "tool_calls", None) else "done"


async def tool_executor_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """执行 LLM 请求的工具：经 MCPToolAdapter → MCP tools/call，结果回灌（§7.6）。"""
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        emit("tool_start", {"name": call["name"], "args": call["args"]})
        out = await ctx.mcp_adapter.call(call["name"], call["args"],
                                         state["session_id"], state["user_id"])
        results.append(ToolMessage(content=json.dumps(out, ensure_ascii=False),
                                   tool_call_id=call["id"]))
    return {"messages": results}

def _hit_rank_key(h: dict) -> float:
    """跨库合并排序键：hybrid 结果按 RRF 融合分（越大越好），vector 按距离（越小越好）。"""
    if h.get("score") is not None:
        return -float(h.get("score") or 0)
    return float(h.get("distance") or float("inf"))


def _chunk_group_of(hit: dict, group_size: int) -> int:
    """命中 chunk 的父块组号：chunk_id 形如 {kb_id}_{doc_id}_{i}，组 = i // group_size。"""
    try:
        i = int(str(hit.get("id", "")).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        i = 0
    return i // group_size


def _expand_parent_blocks(ctx, kb_hits: list, parent_budget: int,
                          group_size: int, max_chars: int,
                          total: int) -> list[dict]:
    """聚合返回：前 parent_budget 个不同父块展开为完整段落，其余按分数返回小 chunk。

    - 父块 key = (kb_id, doc_id, 组号)；已展开组内的命中 chunk 不再返回
      （其上下文已包含在父块里，只累计 hit_chunks）
    - 父块分数 = 组内最高命中分（按分数降序扫描，第一个命中的分即组内最高）
    - 父块与小 chunk 按分数混排（复用 _hit_rank_key）
    - 展开失败（磁盘缺失）自动降级为小 chunk；父块不足 budget 时自然退化
    """
    parents: dict = {}                       # key -> 父块条目
    expanded: set = set()
    for h in kb_hits:
        if len(expanded) >= parent_budget:
            break
        doc_id = (h.get("metadata") or {}).get("doc_id")
        if not doc_id:
            continue
        key = (h.get("kb_id"), doc_id, _chunk_group_of(h, group_size))
        if key in expanded:
            continue
        block = ctx.kb_service.get_parent_block(
            key[0], doc_id, key[2], group_size=group_size, max_chars=max_chars)
        if block is None:
            continue
        entry = {
            "type": "parent",
            "text": block["text"],
            "kb_id": h.get("kb_id"), "kb_name": h.get("kb_name"),
            "scope": h.get("scope"),
            "doc_id": doc_id, "group": key[2],
            "source": block.get("source"), "pages": block.get("pages"),
            "hit_chunks": 1,
            "score": h.get("score"), "distance": h.get("distance"),
            "bm25_score": h.get("bm25_score"), "method": h.get("method"),
            "metadata": {"scope": h.get("scope"), "doc_id": doc_id},
        }
        parents[key] = entry
        expanded.add(key)

    chunk_budget = max(0, total - len(expanded))   # 剩余名额给小 chunk
    chunks: list[dict] = []
    for h in kb_hits:
        doc_id = (h.get("metadata") or {}).get("doc_id")
        key = ((h.get("kb_id"), doc_id, _chunk_group_of(h, group_size))
               if doc_id else None)
        if key in expanded:
            parents[key]["hit_chunks"] += 1    # 上下文已在父块里，只累计不重复返回
            continue
        if len(chunks) >= chunk_budget:
            break
        chunks.append({**h, "type": "chunk"})
    all_entries = list(parents.values()) + chunks
    all_entries.sort(key=_hit_rank_key)
    return all_entries


async def retrieve_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """按 LLM 选定的知识库检索（仅限可见库），结果带 scope 标签（引用溯源）。

    检索模式：state.retrieval_mode（前端每轮传）→ 全局配置 retrieval_mode 兜底。
    """
    kbs = ctx.kb_service.list_kbs(state["user_id"])
    selected_ids = set(state.get("selected_kb_ids") or [])
    # 双保险：即使 selected_ids 携带被禁用的库（路由与检索之间用户改了开关），也跳过
    targets = [kb for kb in kbs
               if kb.kb_id in selected_ids and kb.retrieval_enabled]
    mode = (state.get("retrieval_mode")
            or getattr(ctx.settings, "retrieval_mode", "hybrid"))
    # 检索数量：每库 k 条 / 合并后总共 top 条（state 传入 → 全局配置兜底，带范围约束）
    per_kb = int(state.get("per_kb_k") or 0) or int(
        getattr(ctx.settings, "retrieval_per_kb_k", 3))
    total = int(state.get("total_k") or 0) or int(
        getattr(ctx.settings, "retrieval_total_k", 5))
    per_kb = max(1, min(per_kb, 20))
    total = max(1, min(total, 50))
    # 聚合返回：父块名额（0=关闭，全返回小 chunk；None 或 <0 = 全局配置默认）
    parent_groups = state.get("parent_groups")
    if parent_groups is None or parent_groups < 0:
        parent_groups = int(getattr(ctx.settings, "retrieval_parent_groups", 3))
    group_size = int(getattr(ctx.settings, "retrieval_parent_group_size", 3))
    max_chars = int(getattr(ctx.settings, "retrieval_parent_max_chars", 4000))
    hits = []
    for kb in targets:
        hits.extend(ctx.kb_service.search(kb.kb_id, state["query"], k=per_kb,
                                          user_id=state["user_id"], mode=mode))
    hits.sort(key=_hit_rank_key)
    if parent_groups > 0:
        # 候选池要留足余量：已展开父块组内的命中会被跳过，多取一些候选
        top = _expand_parent_blocks(ctx, hits[:max(total + parent_groups * 2,
                                                   len(hits))],
                                    parent_budget=min(parent_groups, total),
                                    group_size=max(1, group_size),
                                    max_chars=max(200, max_chars),
                                    total=total)
    else:
        top = [{**h, "type": "chunk"} for h in hits[:total]]
    # 引用溯源推流：前端据此展示检索来源面板（text 截断，避免事件过大）
    emit("retrievals", {"mode": mode, "results": [
        {"type": h.get("type", "chunk"), "kb_name": h.get("kb_name"),
         "scope": h.get("scope"),
         "distance": h.get("distance"), "bm25_score": h.get("bm25_score"),
         "score": h.get("score"), "method": h.get("method"),
         "source": h.get("source") or (h.get("metadata") or {}).get("source"),
         "page": (h.get("metadata") or {}).get("page"),
         "pages": h.get("pages"), "hit_chunks": h.get("hit_chunks"),
         "text": str(h.get("text", ""))[:300]}
        for h in top]})
    return {"retrievals": top}


async def generate_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """用（用户级）LLM 生成答复：系统提示词带记忆/检索结果 + 用户消息。

    用 astream 逐 token 生成：LangGraph 以 stream_mode="messages" 把每个 token
    实时推给 SSE 端点（前端打字机效果）；聚合后的完整 message 仍写入 state，
    tool_calls 也随之聚合，不影响 generate ⇄ tool_executor 循环。
    """
    model = ctx.llm_service.get_chat_model(
        state["user_id"], temperature=state.get("temperature"))
    if ctx.mcp_adapter is not None:
        schemas = await ctx.mcp_adapter.schemas_for_llm()
        if schemas:
            model = model.bind_tools(schemas)      # 告诉 LLM"你有这些工具可用"
    system = SystemMessage(content=_build_system_prompt(state))
    if ctx.tracer is not None:
        log_id = ctx.tracer.start("llm", getattr(model, "model_name", "chat"),
                                  state["session_id"], state["user_id"])
    resp = None
    async for chunk in model.astream([system] + state["messages"]):
        resp = chunk if resp is None else resp + chunk     # 逐 token 聚合为完整消息
        # 每个 token 实时推给事件总线（SSE 端点持续 drain → 前端打字机效果）
        text = chunk.content
        if isinstance(text, str):
            if text:
                emit("token", {"content": text})
        elif isinstance(text, list):
            for p in text:
                if isinstance(p, dict) and p.get("text"):
                    emit("token", {"content": p["text"]})
    if resp is None:
        resp = AIMessage(content="")
    answer = str(resp.content) if resp.content else ""
    if ctx.tracer is not None:
        ctx.tracer.success(log_id, answer[:2000])
    return {"messages": [resp], "answer": answer}
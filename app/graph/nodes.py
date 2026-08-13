from dataclasses import dataclass
import json
import re

from langchain_core.messages import AIMessage, SystemMessage,ToolMessage

from app.core.events import emit
from app.services.memory_service import MAX_MEMORIES_PER_USER
from app.graph.state import AgentState


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


async def load_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """跨会话读取：把该用户的长记忆载入状态（图的第一站）。"""
    if ctx.memory_service is None:
        return {"memory": {}}
    memory = ctx.memory_service.get_all(state["user_id"])
    emit("memory_load", {"count": len(memory)})
    return {"memory": memory}


async def extract_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """对话结束后抽取值得记住的信息：交给 LLM 从对话中提炼。"""
    if ctx.memory_service is None:
        return {"new_memories": []}
    model = ctx.llm_service.get_chat_model(state["user_id"])
    system = SystemMessage(content=EXTRACT_PROMPT)
    resp = await model.ainvoke([system] + state["messages"][-4:])   # 只看最近几轮
    memories = _parse_memories(str(resp.content or ""))
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
    """把检索结果拼进系统提示词——RAG 的核心：知识先进 prompt，LLM 才能引用。"""
    parts = ["你是科研助手，基于知识库检索结果回答用户问题，引用时标明来源（public/private）。"]
    if state.get("memory"):
        parts.append(f"[用户记忆] {json.dumps(state['memory'], ensure_ascii=False)}")
    if state.get("retrievals"):
        lines = [f"[知识库检索结果 ({r.get('scope')} / {r.get('kb_name')})] {r['text']}"
                 for r in state["retrievals"]]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def supervisor_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """路由决策：用户有可用知识库 → 先检索；没有 → 直接生成。"""
    kbs = ctx.kb_service.list_kbs(state["user_id"])
    needs = len(kbs) > 0
    emit("supervisor", {"needs_retrieval": needs, "kb_count": len(kbs)})
    return {"needs_retrieval": needs}


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

async def retrieve_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """两级 KB 合并检索：public + 本人 private，结果带 scope 标签（引用溯源）。"""
    kbs = ctx.kb_service.list_kbs(state["user_id"])
    hits = []
    for kb in kbs:
        hits.extend(ctx.kb_service.search(kb.kb_id, state["query"], k=3,
                                          user_id=state["user_id"]))
    hits.sort(key=lambda h: h.get("distance", 0))
    top = hits[:5]
    # 引用溯源推流：前端据此展示检索来源面板（text 截断，避免事件过大）
    emit("retrievals", {"results": [
        {"kb_name": h.get("kb_name"), "scope": h.get("scope"),
         "distance": h.get("distance"), "text": str(h.get("text", ""))[:300]}
        for h in top]})
    return {"retrievals": top}


async def generate_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """用（用户级）LLM 生成答复：系统提示词带记忆/检索结果 + 用户消息。

    用 astream 逐 token 生成：LangGraph 以 stream_mode="messages" 把每个 token
    实时推给 SSE 端点（前端打字机效果）；聚合后的完整 message 仍写入 state，
    tool_calls 也随之聚合，不影响 generate ⇄ tool_executor 循环。
    """
    model = ctx.llm_service.get_chat_model(state["user_id"])
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
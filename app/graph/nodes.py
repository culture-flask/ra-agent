from dataclasses import dataclass

from langchain_core.messages import SystemMessage

from app.core.events import emit
from app.graph.state import AgentState


@dataclass
class WorkflowContext:
    """编排层依赖注入：节点所需的 llm / 知识库等服务，编译时注入。"""
    settings: object
    llm_service: object
    kb_service: object


def _build_system_prompt(state: AgentState) -> str:
    """把检索结果拼进系统提示词——RAG 的核心：知识先进 prompt，LLM 才能引用。"""
    parts = ["你是科研助手，基于知识库检索结果回答用户问题，引用时标明来源（public/private）。"]
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


async def retrieve_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """两级 KB 合并检索：public + 本人 private，结果带 scope 标签（引用溯源）。"""
    kbs = ctx.kb_service.list_kbs(state["user_id"])
    hits = []
    for kb in kbs:
        hits.extend(ctx.kb_service.search(kb.kb_id, state["query"], k=3,
                                          user_id=state["user_id"]))
    hits.sort(key=lambda h: h.get("distance", 0))
    return {"retrievals": hits[:5]}


async def generate_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """用（用户级）LLM 生成答复：系统提示词带记忆/检索结果 + 用户消息。"""
    model = ctx.llm_service.get_chat_model(state["user_id"])
    system = SystemMessage(content=_build_system_prompt(state))
    resp = await model.ainvoke([system] + state["messages"])
    return {"messages": [resp], "answer": str(resp.content) if resp.content else ""}
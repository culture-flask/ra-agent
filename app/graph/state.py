from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """LangGraph 图状态"""
    user_id: str
    session_id: str
    query: str                              # 用户本次提问
    messages: Annotated[list[BaseMessage], add_messages]   # 对话历史（自动追加）
    retrievals: list[dict]                  # 检索结果（带 scope 标签）
    needs_retrieval: bool                   # supervisor 的路由决策（LLM 意图判断）
    selected_kb_ids: list[str]              # LLM 按名称选中的知识库 id（仅可见库）
    answer: str                             # 最终答复
    memory: dict                            # 长期记忆（用户级，跨会话）
    new_memories: list[dict]                # 本次抽取待审核的记忆
    conversation_summary: str               # 自动压缩产生的历史对话总结（拼进系统提示词）

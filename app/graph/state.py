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
    retrieval_mode: str                     # 检索模式：vector（纯向量）| hybrid（向量+BM25）
    per_kb_k: int                           # 每个知识库检索几条（0=全局默认）
    total_k: int                            # 所有库合并后总共取几条（0=全局默认）
    temperature: float | None               # 生成温度（None=默认 0.3；0 也合法）
    parent_groups: int | None               # 聚合返回的父块名额（None=全局默认；0=关闭聚合）
    last_usage: dict | None                 # 上一次 generate 的真实 token 用量（来自 LLM 响应）
    stopped: bool                           # 本轮生成被用户手动中断（部分答复仍入 checkpoint）
    # 跨轮检索状态（第一优先，P3-34）：记录上一轮是否检索过、搜了哪些库、命中几条。
    # 由 checkpointer 持久保留（_initial_state 不重置它），供 supervisor 判断
    # "延续上轮话题且上轮已基于检索回答"时跳过重复检索。只作路由决策输入，
    # 绝不注入 system/历史——那会打穿前缀缓存。
    last_retrieval_state: dict | None

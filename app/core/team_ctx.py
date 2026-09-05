"""多 agent 团队上下文：请求内声明"当前调用方属于哪支团队"。

知识库检索的隔离依据（见 kb_service.list_queryable_kbs）：
- team=None（普通对话/未声明）→ 只能检索普通库，多 agent 沉淀库全部不可见
- team=debate/seminar/deep_research → 普通库 + 本团队的沉淀库（团队互相隔离）

API 层在流式/追问端点入口 set，MCP 原生工具（search_knowledge_base 等）
在任意深度读取——ContextVar 随 asyncio 任务树传播，无需层层传参。
"""

import contextvars

# debate=争鸣社 | seminar=格致会讲 | deep_research=溯源社 | None=普通对话
_team: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_team", default=None)


def set_agent_team(team: str | None) -> None:
    """声明当前请求的团队身份（端点入口调用；None=普通对话，显式覆盖防串场）。"""
    _team.set(team)


def get_agent_team() -> str | None:
    """读取当前团队身份；原生工具据此过滤沉淀库可见性。"""
    return _team.get()

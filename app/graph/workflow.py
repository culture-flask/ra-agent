from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio

from psycopg import OperationalError


async def aget_state_retry(graph, config: dict, attempts: int = 3):
    """aget_state 的连接自愈重试。

    Postgres/项目重启后，连接池异步丢弃旧连接存在竞态：第一次取出的
    连接可能仍报 OperationalError（"the connection is closed"）。
    指数退避重试，池换上新连接后即恢复。"""
    for i in range(attempts):
        try:
            return await graph.aget_state(config)
        except OperationalError:
            # 整类连接级错误都重试：连接关闭 / AdminShutdown（Postgres 重启
            # "terminating connection due to administrator command"）/ 断连。
            # 池会在失败后异步丢弃坏连接并重建，退避一次即拿到新连接。
            if i == attempts - 1:
                raise
            await asyncio.sleep(0.5 * (i + 1))
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    WorkflowContext,
    compact_node,
    generate_node,
    load_memory_node,
    retrieve_node,
    route_after_generate,
    route_supervisor,
    supervisor_node,
    tool_executor_node,
)
from app.graph.state import AgentState

# P1-8：extract_memory / save_memory 不再进图——记忆抽取/落库由 API 层在
# 答案生成完（SSE 已推 done）之后以后台任务补跑，不阻塞流式结束。


async def _build_checkpointer(database_url: str) -> AsyncPostgresSaver:
    """创建 Postgres 检查点保存器（异步连接池版）。

    用连接池而非单条长连接：Postgres/项目重启后旧连接会死亡
    （psycopg.OperationalError: the connection is closed），详情/追问等
    aget_state 调用会集体 500；池在取出时校验并自动重建坏连接。
    prepare_threshold=0 是 langgraph 官方对 checkpoint 表的建议配置。
    """
    url = database_url.replace("postgresql+psycopg://", "postgresql://")
    pool = AsyncConnectionPool(
        url, min_size=1, max_size=10, open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver


async def build_graph(ctx: WorkflowContext) -> StateGraph:
    builder = StateGraph(AgentState)

    async def _load_memory(s): return await load_memory_node(ctx, s)
    async def _compact(s): return await compact_node(ctx, s)
    async def _supervisor(s): return await supervisor_node(ctx, s)
    async def _retrieve(s): return await retrieve_node(ctx, s)
    async def _generate(s): return await generate_node(ctx, s)
    async def _tool_executor(s): return await tool_executor_node(ctx, s)

    builder.add_node("load_memory", _load_memory)
    builder.add_node("compact", _compact)
    builder.add_node("supervisor", _supervisor)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)
    builder.add_node("tool_executor", _tool_executor)

    builder.add_edge(START, "load_memory")                 # 先读记忆
    builder.add_edge("load_memory", "compact")             # 再查是否需压缩
    builder.add_edge("compact", "supervisor")
    builder.add_conditional_edges("supervisor", route_supervisor,
                                  {"retrieve": "retrieve", "generate": "generate"})
    builder.add_edge("retrieve", "generate")
    # done 直达 END：记忆抽/存在图外后台补跑，SSE 提前结束
    builder.add_conditional_edges("generate", route_after_generate,
                                  {"tool_executor": "tool_executor", "done": END})
    builder.add_edge("tool_executor", "generate") 

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)
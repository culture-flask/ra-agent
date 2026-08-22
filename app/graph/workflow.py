import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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
    """创建 Postgres 检查点保存器（异步版）。
    """
    url = database_url.replace("postgresql+psycopg://", "postgresql://")
    conn = await psycopg.AsyncConnection.connect(url, autocommit=True)
    saver = AsyncPostgresSaver(conn)
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
    # done 直达 END（P1-8）：记忆抽/存在图外后台补跑，SSE 提前结束
    builder.add_conditional_edges("generate", route_after_generate,
                                  {"tool_executor": "tool_executor", "done": END})
    builder.add_edge("tool_executor", "generate") 

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)
import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    WorkflowContext,
    generate_node,
    retrieve_node,
    route_supervisor,
    supervisor_node,
)
from app.graph.state import AgentState


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

    async def _supervisor(s): return await supervisor_node(ctx, s)
    async def _retrieve(s): return await retrieve_node(ctx, s)
    async def _generate(s): return await generate_node(ctx, s)

    builder.add_node("supervisor", _supervisor)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_supervisor,
                                  {"retrieve": "retrieve", "generate": "generate"})
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)
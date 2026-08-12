import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    WorkflowContext,
    extract_memory_node,
    generate_node,
    load_memory_node,
    retrieve_node,
    route_after_generate,
    route_supervisor,
    save_memory_node,
    supervisor_node,
    tool_executor_node,
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

    async def _load_memory(s): return await load_memory_node(ctx, s)
    async def _supervisor(s): return await supervisor_node(ctx, s)
    async def _retrieve(s): return await retrieve_node(ctx, s)
    async def _generate(s): return await generate_node(ctx, s)
    async def _tool_executor(s): return await tool_executor_node(ctx, s)
    async def _extract_memory(s): return await extract_memory_node(ctx, s)
    async def _save_memory(s): return await save_memory_node(ctx, s)

    builder.add_node("load_memory", _load_memory)
    builder.add_node("supervisor", _supervisor)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)
    builder.add_node("tool_executor", _tool_executor)
    builder.add_node("extract_memory", _extract_memory)
    builder.add_node("save_memory", _save_memory)

    builder.add_edge(START, "load_memory")                 # 先读记忆
    builder.add_edge("load_memory", "supervisor")
    builder.add_conditional_edges("supervisor", route_supervisor,
                                  {"retrieve": "retrieve", "generate": "generate"})
    builder.add_edge("retrieve", "generate")
    builder.add_conditional_edges("generate", route_after_generate,
                                  {"tool_executor": "tool_executor", "done": "extract_memory"})
    builder.add_edge("tool_executor", "generate") 
    builder.add_edge("extract_memory", "save_memory")      # Day 7：抽→审→存
    builder.add_edge("save_memory", END)

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)
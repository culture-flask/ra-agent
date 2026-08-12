import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.abstractions.llm import LLMService
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.kbs import router as kbs_router
from app.api.traces import router as traces_router
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.net import apply_proxy
from app.core.tracing import Tracer
from app.graph.nodes import WorkflowContext
from app.graph.workflow import build_graph
from app.mcp.adapter import MCPToolAdapter
from app.mcp.host import MCPHost
from app.services.kb_service import KBService
from app.services.memory_service import MemoryService
from app.api.memories import router as memories_router
from app.settings import BASE_DIR, Settings

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时/关闭时执行。"""
    settings = Settings.load()
    apply_proxy(settings)          # 外部网络代理（无代理环境也能出网）
    app.state.settings = settings

    # --- 编排层装配（图 + 知识库 + LLM）---
    kb_service = KBService(settings)
    llm_service = LLMService(system_default=settings.llm_system_default,
                             system_api_key=settings.llm_api_key)
    # --- MCP 工具框架 + 调用追踪 ---
    tracer = Tracer()
    mcp_host = MCPHost(settings.mcp_servers, base_dir=BASE_DIR)
    mcp_adapter = MCPToolAdapter(mcp_host, tracer)
    await mcp_adapter.ensure_catalog()         # 启动时 tools/list 动态发现
    logger.info("发现 %d 个 MCP 工具", len(mcp_host.tools or []))

    memory_service = MemoryService()
    ctx = WorkflowContext(settings, llm_service, kb_service, mcp_adapter, tracer, memory_service)
    app.state.graph = await build_graph(ctx)      # async：内部建 AsyncPostgresSaver
    app.state.kb_service = kb_service
    app.state.tracer = tracer
    app.state.memory_service = memory_service

    logger.info("starting %s", settings.app_name)
    yield
    logger.info("shutting down")


app = FastAPI(title="ra-agent", version="0.1.0", lifespan=lifespan)
register_exception_handlers(app)           # 第 10 节实现
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(kbs_router)
app.include_router(traces_router)
app.include_router(memories_router)

@app.get("/health")
async def health():
    """健康检查：K8s/Compose 用它判断服务是否存活。"""
    return {"status": "ok", "app": "ra-agent"}


if __name__ == "__main__":
    setup_logging()
    s = Settings.load()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=True)
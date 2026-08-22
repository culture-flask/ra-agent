import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.abstractions.llm import LLMService
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.feedbacks import router as feedbacks_router
from app.api.kbs import router as kbs_router
from app.api.traces import router as traces_router
from app.api.llm_config import router as llm_config_router
from app.api.settings import router as settings_router
from app.core.crypto import SecretCrypto
from app.core.db import engine
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.net import apply_proxy
from app.core.tracing import Tracer
from app.graph.nodes import WorkflowContext
from app.graph.workflow import build_graph
from app.mcp.adapter import MCPToolAdapter
from app.mcp.host import MCPHost
from app.models import Conversation, Feedback, Memory
from app.services.kb_service import KBService
from app.services.memory_service import MemoryService
from app.api.memories import router as memories_router
from app.settings import BASE_DIR, Settings

logger = get_logger("main")

# 模块导入即配置日志：uvicorn app.main:app 启动时 __name__ != "__main__"，
# 若只在 __main__ 块里 setup_logging，日志永远不落 logs/app.log（只走
# stderr 兜底裸输出）。幂等保护在 setup_logging 内，重复导入不重复挂 handler。
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时/关闭时执行。"""
    settings = Settings.load()
    apply_proxy(settings)          # 外部网络代理（无代理环境也能出网）
    app.state.settings = settings
    # 会话登记表（跨设备同步）：幂等建表，已有表不动
    Conversation.__table__.create(engine, checkfirst=True)
    # 记忆分层列（膨胀控制）：新建表含新列；旧表幂等补列 + 存量回填
    Memory.__table__.create(engine, checkfirst=True)
    # 用户反馈（P3-19 反馈闭环）：评测集种子数据
    Feedback.__table__.create(engine, checkfirst=True)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS "
                          "tier VARCHAR(8) NOT NULL DEFAULT 'core'"))
        conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS "
                          "topic VARCHAR(64) NOT NULL DEFAULT ''"))
        conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS "
                          "last_used_at TIMESTAMPTZ"))
        conn.execute(text("UPDATE memories SET last_used_at = updated_at "
                          "WHERE last_used_at IS NULL"))
        # 用户自定义上下文窗口：旧表幂等补列（空 = 自动探测/兜底默认）
        conn.execute(text("ALTER TABLE user_llm_config ADD COLUMN IF NOT EXISTS "
                          "context_window INTEGER"))
        # 检索开关细化到用户：旧表幂等补列（禁用列表，用户间互不影响）
        conn.execute(text("ALTER TABLE kbs ADD COLUMN IF NOT EXISTS "
                          "retrieval_disabled_users JSON NOT NULL DEFAULT '[]'"))

        # ---- P1-6 孤儿状态自愈 ----
        # 入库/重建/复制是 BackgroundTasks，随进程消亡且进度只存内存字典；
        # 进程中途被杀会把 kbs.status 永久卡在非终态——此后该库上传永远
        # 409 "already indexing"，只能手改数据库解救。启动时统一复位：
        # 标 failed（诚实——部分文件可能没嵌完）并立刻解除 409 死锁，
        # 用户重新上传即可（_finish_ingest 允许失败库继续入库）。
        res = conn.execute(text("UPDATE kbs SET status = 'failed' "
                                "WHERE status IN ('indexing', 'reembedding', 'copying')"))
        if res.rowcount:
            logger.warning("启动复位 %d 个非终态知识库（上次进程中断残留）",
                           res.rowcount)

    # --- 编排层装配（图 + 知识库 + LLM）---
    kb_service = KBService(settings)
    llm_service = LLMService(system_default=settings.llm_system_default,
                             system_api_key=settings.llm_api_key,
                             crypto=SecretCrypto(settings.jwt_secret),
                             retry_max_retries=settings.llm_retry_max_retries,
                             retry_base_delay=settings.llm_retry_base_delay,
                             context_window_default=settings.llm_context_window)
    # --- MCP 工具框架 + 调用追踪 ---
    tracer = Tracer()
    mcp_host = MCPHost(settings.mcp_servers, base_dir=BASE_DIR)
    mcp_adapter = MCPToolAdapter(mcp_host, tracer, kb_service=kb_service)
    await mcp_adapter.ensure_catalog()         # 启动时 tools/list 动态发现
    logger.info("发现 %d 个 MCP 工具", len(mcp_host.tools or []))

    memory_service = MemoryService()
    ctx = WorkflowContext(settings, llm_service, kb_service, mcp_adapter, tracer, memory_service)
    app.state.workflow_ctx = ctx                  # P1-8：API 层后台记忆管线复用同一编排上下文
    app.state.graph = await build_graph(ctx)      # async：内部建 AsyncPostgresSaver
    app.state.kb_service = kb_service
    app.state.tracer = tracer
    app.state.memory_service = memory_service
    app.state.llm_service = llm_service

    logger.info("starting %s", settings.app_name)
    yield
    logger.info("shutting down")


app = FastAPI(title="ra-agent", version="0.1.0", lifespan=lifespan)
# 前端（浏览器跨域访问）需要 CORS；本地/私有化场景放开即可
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
register_exception_handlers(app)           # 第 10 节实现
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(kbs_router)
app.include_router(traces_router)
app.include_router(memories_router)
app.include_router(feedbacks_router)
app.include_router(llm_config_router)
app.include_router(settings_router)

@app.get("/health")
async def health():
    """健康检查：K8s/Compose 用它判断服务是否存活。"""
    return {"status": "ok", "app": "ra-agent"}


# 托管前端（ra-web）：前后端同源，浏览器不再跨域，规避本地网络拦截。
# 挂载在 API 路由之后——Starlette 按注册顺序匹配，/api/* 与 /health 优先命中。
# 前端目录在项目内（ra-agent/ra-web）
WEB_DIR = BASE_DIR / "ra-web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    s = Settings.load()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=True)
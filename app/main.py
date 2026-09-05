import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.abstractions.llm import LLMService
from app.api.auth import router as auth_router
from app.api.brainstorm import router as brainstorm_router
from app.api.seminar import router as seminar_router
from app.api.deep_research import router as deep_research_router   # 溯源社：深度调研式多 agent
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.feedbacks import router as feedbacks_router
from app.api.usage import router as usage_router
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
from app.graph.brainstorm import build_brainstorm_graph
from app.graph.seminar import build_seminar_graph
from app.graph.deep_research import build_deep_research_graph      # 溯源社：深度调研式多 agent
from app.graph.nodes import WorkflowContext
from app.graph.workflow import build_graph
from app.mcp.adapter import MCPToolAdapter
from app.mcp.host import MCPHost
from app.models import BrainstormSession, Conversation, Feedback, LLMUsage, Memory
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
    # 用户反馈：评测集种子数据
    Feedback.__table__.create(engine, checkfirst=True)
    # Token 用量计量（P3-20）：成本报表与配额演进数据源
    LLMUsage.__table__.create(engine, checkfirst=True)
    # 头脑风暴会话登记表
    BrainstormSession.__table__.create(engine, checkfirst=True)
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
        # 用量计量表补列（P3-20）：存量开发库补 cached_tokens
        conn.execute(text("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS "
                          "cached_tokens INTEGER NOT NULL DEFAULT 0"))
        # 会话 id 加宽（幂等）：头脑风暴会话 id = "bs-" + uuid（39 字符），
        # 36 列会让每条工具追踪/用量写入报 StringDataRightTruncation
        conn.execute(text("ALTER TABLE tool_call_log "
                          "ALTER COLUMN session_id TYPE VARCHAR(64)"))
        conn.execute(text("ALTER TABLE llm_usage "
                          "ALTER COLUMN session_id TYPE VARCHAR(64)"))

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
        # 头脑风暴：进程中断残留 running → 复位 failed（同 kbs 自愈模式）
        res_bs = conn.execute(text("UPDATE brainstorm_sessions "
                                   "SET status = 'failed' "
                                   "WHERE status = 'running'"))
        if res_bs.rowcount:
            logger.warning("启动复位 %d 个中断的头脑风暴会话", res_bs.rowcount)
        # 两支多 agent 团队分家：旧库补 team 列（幂等，新库由 create 带出）
        has_team = conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'brainstorm_sessions' "
            "AND column_name = 'team'")).scalar()
        if not has_team:
            conn.execute(text("ALTER TABLE brainstorm_sessions "
                              "ADD COLUMN team VARCHAR(16) "
                              "NOT NULL DEFAULT 'debate'"))
            logger.warning("已为 brainstorm_sessions 补 team 列（存量行=debate）")
        # 知识库 kind 列：区分普通库与多 agent 自动沉淀库（幂等，新库由 create 带出）
        has_kind = conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'kbs' AND column_name = 'kind'")).scalar()
        if not has_kind:
            conn.execute(text("ALTER TABLE kbs ADD COLUMN kind VARCHAR(16) "
                              "NOT NULL DEFAULT 'user'"))
            logger.warning("已为 kbs 补 kind 列（存量行=user）")
        # 存量沉淀库改名（头脑风暴成果→辩论式agent纪要 / 会讲纪要→研讨式agent纪要）
        # 并标注 archive（幂等：旧名不存在时均为 no-op）
        conn.execute(text("UPDATE kbs SET name = '辩论式agent纪要' "
                          "WHERE name = '头脑风暴成果'"))
        conn.execute(text("UPDATE kbs SET name = '研讨式agent纪要' "
                          "WHERE name = '会讲纪要'"))
        conn.execute(text("UPDATE kbs SET kind = 'archive' "
                          "WHERE name IN ('辩论式agent纪要', '研讨式agent纪要') "
                          "AND kind = 'user'"))
        # 用户级嵌入模型默认配置（每用户一条）：建库缺省时优先于系统默认
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS user_embedding_config ("
            "user_id VARCHAR(36) PRIMARY KEY, "
            "provider VARCHAR(32) NOT NULL, "
            "model_id VARCHAR(128) NOT NULL, "
            "dim INTEGER NOT NULL, "
            "base_url VARCHAR(256), "
            "api_key VARCHAR(512), "
            "updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now())"))

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
    app.state.workflow_ctx = ctx                  # API 层后台记忆管线复用同一编排上下文
    app.state.graph = await build_graph(ctx)      # async：内部建 AsyncPostgresSaver
    app.state.brainstorm_graph = await build_brainstorm_graph(ctx)   # 争鸣社子图（独立 checkpointer）
    app.state.seminar_graph = await build_seminar_graph(ctx)   # 会讲子图
    # 溯源社子图（独立 checkpointer：AsyncPostgresSaver 绑定事件循环，不可跨图共享实例）
    app.state.deep_research_graph = await build_deep_research_graph(ctx)
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
register_exception_handlers(app)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(brainstorm_router)
app.include_router(seminar_router)
app.include_router(deep_research_router)   # 溯源社：深度调研式多 agent
app.include_router(conversations_router)
app.include_router(kbs_router)
app.include_router(traces_router)
app.include_router(memories_router)
app.include_router(feedbacks_router)
app.include_router(usage_router)
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
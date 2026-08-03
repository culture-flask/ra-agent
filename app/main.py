import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.settings import Settings

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时/关闭时执行。"""
    settings = Settings.load()
    app.state.settings = settings          # 挂到 app 上，路由里用 request.app.state.settings 取
    logger.info("starting %s", settings.app_name)
    yield
    logger.info("shutting down")


app = FastAPI(title="ra-agent", version="0.1.0", lifespan=lifespan)
register_exception_handlers(app)           # 第 10 节实现


@app.get("/health")
async def health():
    """健康检查：K8s/Compose 用它判断服务是否存活。"""
    return {"status": "ok", "app": "ra-agent"}


if __name__ == "__main__":
    setup_logging()
    s = Settings.load()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=True)
"""调用追踪 API：查询 ToolCallLog（用户/会话过滤）。"""

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

router = APIRouter(prefix="/api/v1", tags=["traces"])


@router.get("/traces")
async def list_traces(request: Request, user_id: str | None = None,
                      session_id: str | None = None):
    """查询调用日志：默认全部，可按 user_id / session_id 过滤。"""
    return await run_in_threadpool(request.app.state.tracer.list,
                                   user_id=user_id, session_id=session_id)

"""调用追踪 API：查询 ToolCallLog。

P0-1 鉴权改造：身份一律取自 Bearer token 并强制按当前用户过滤——
调用日志包含工具参数与输出明文，绝不允许跨用户读取
（改造前接受任意 user_id 参数，任何登录用户可翻全库，属越权读取）。
"""

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from app.core.deps import get_current_user
from app.models import User

# 路由级闸门：本组全部端点要求登录态
router = APIRouter(prefix="/api/v1", tags=["traces"],
                   dependencies=[Depends(get_current_user)])


@router.get("/traces")
async def list_traces(request: Request, session_id: str | None = None,
                      user: User = Depends(get_current_user)):
    """查询【当前用户自己】的调用日志，可按 session_id 过滤。"""
    return await run_in_threadpool(request.app.state.tracer.list,
                                   user_id=user.id, session_id=session_id)

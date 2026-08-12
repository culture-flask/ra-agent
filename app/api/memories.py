from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1", tags=["memories"])


@router.get("/memories")
async def list_memories(request: Request, user_id: str):
    """查看用户长期记忆（只能查自己的；生产接入鉴权后强制用 token 的用户）。"""
    return request.app.state.memory_service.list(user_id)
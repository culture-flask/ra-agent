"""长期记忆：查看 / 分层调级（置顶核心|降为短期）/ 选择性删除（单个或批量）。

P0-1 鉴权改造：身份一律取自 Bearer token；记忆是用户级隐私数据，
改造前凭 query 参数即可读删任意用户的记忆，现已收口为强制本人。
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.deps import get_current_user
from app.models import User

# 路由级闸门：本组全部端点要求登录态
router = APIRouter(prefix="/api/v1", tags=["memories"],
                   dependencies=[Depends(get_current_user)])


class MemoryTierRequest(BaseModel):
    tier: str          # core | short


class MemoryDeleteRequest(BaseModel):
    keys: list[str]


@router.get("/memories")
async def list_memories(request: Request,
                        user: User = Depends(get_current_user)):
    """查看当前用户长期记忆（含层级/主题/最近使用时间）。"""
    return await run_in_threadpool(request.app.state.memory_service.list_memory, user.id)


@router.patch("/memories/{key}/tier")
async def update_memory_tier(key: str, req: MemoryTierRequest, request: Request,
                             user: User = Depends(get_current_user)):
    """手动调级：short → core（置顶常驻，注入每轮对话）；core → short（降级，TTL 过期清除）。"""
    if req.tier not in ("core", "short"):
        raise HTTPException(status_code=400, detail="tier 必须是 core 或 short")
    ok = await run_in_threadpool(request.app.state.memory_service.set_tier,
                                 user.id, key, req.tier)
    if not ok:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"key": key, "tier": req.tier}


@router.delete("/memories/{key}")
async def delete_memory(key: str, request: Request,
                        user: User = Depends(get_current_user)):
    """删除单条记忆。"""
    n = await run_in_threadpool(request.app.state.memory_service.delete_many,
                                user.id, [key])
    if n == 0:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": [key]}


@router.post("/memories/delete")
async def delete_memories(req: MemoryDeleteRequest, request: Request,
                          user: User = Depends(get_current_user)):
    """批量选择性删除（前端勾选后一次删多条）。"""
    n = await run_in_threadpool(request.app.state.memory_service.delete_many,
                                user.id, req.keys)
    return {"deleted_count": n}

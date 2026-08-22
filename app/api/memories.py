"""长期记忆：查看 / 分层调级（置顶核心|降为短期）/ 选择性删除（单个或批量）。"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["memories"])


class MemoryTierRequest(BaseModel):
    tier: str          # core | short


class MemoryDeleteRequest(BaseModel):
    keys: list[str]


@router.get("/memories")
async def list_memories(request: Request, user_id: str):
    """查看用户长期记忆（含层级/主题/最近使用时间）。"""
    return await run_in_threadpool(request.app.state.memory_service.list_memory, user_id)


@router.patch("/memories/{key}/tier")
async def update_memory_tier(key: str, req: MemoryTierRequest, request: Request,
                             user_id: str):
    """手动调级：short → core（置顶常驻，注入每轮对话）；core → short（降级，TTL 过期清除）。"""
    if req.tier not in ("core", "short"):
        raise HTTPException(status_code=400, detail="tier 必须是 core 或 short")
    ok = await run_in_threadpool(request.app.state.memory_service.set_tier,
                                 user_id, key, req.tier)
    if not ok:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"key": key, "tier": req.tier}


@router.delete("/memories/{key}")
async def delete_memory(key: str, request: Request, user_id: str):
    """删除单条记忆。"""
    n = await run_in_threadpool(request.app.state.memory_service.delete_many,
                                user_id, [key])
    if n == 0:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": [key]}


@router.post("/memories/delete")
async def delete_memories(req: MemoryDeleteRequest, request: Request,
                          user_id: str):
    """批量选择性删除（前端勾选后一次删多条）。"""
    n = await run_in_threadpool(request.app.state.memory_service.delete_many,
                                user_id, req.keys)
    return {"deleted_count": n}

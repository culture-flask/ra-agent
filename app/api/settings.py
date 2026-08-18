"""运行配置端点：向前端暴露 yaml 中的检索等运行参数。

前端 UI 默认值此前硬编码在 JS 里（3/5/3），改 yaml 无感知；
现在由这里下发，前端未手动设置时跟随 yaml。
"""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.get("/retrieval")
async def retrieval_settings(request: Request):
    """检索运行参数（config/settings.yaml → retrieval 段）：前端默认值来源。"""
    s = request.app.state.settings
    return {
        "mode": s.retrieval_mode,
        "per_kb_k": s.retrieval_per_kb_k,
        "total_k": s.retrieval_total_k,
        "parent_groups": s.retrieval_parent_groups,
        "parent_group_size": s.retrieval_parent_group_size,
        "parent_max_chars": s.retrieval_parent_max_chars,
    }

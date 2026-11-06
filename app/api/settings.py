"""运行配置端点：向前端暴露 yaml 中的检索等运行参数。

前端 UI 默认值此前硬编码在 JS 里（3/5/3），改 yaml 无感知；
现在由这里下发，前端未手动设置时跟随 yaml。
纳入全局鉴权（内容虽非敏感，业务面统一收口）。
"""

from fastapi import APIRouter, Depends, Request

from app.core.deps import get_current_user

# 路由级闸门：本组全部端点要求登录态
router = APIRouter(prefix="/api/v1/settings", tags=["settings"],
                   dependencies=[Depends(get_current_user)])


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


@router.get("/multimodal")
async def multimodal_settings(request: Request):
    """服务器的媒体处理能力；不宣称用户选择的模型也支持这些模态。"""
    service = request.app.state.multimodal_service
    limits = service.limits
    return {
        "available": service.availability(),
        "default": {"image": False, "audio": False, "video": False},
        "max_files": 8, "max_file_bytes": limits.max_bytes,
        "max_audio_seconds": limits.max_audio_seconds,
        "max_video_seconds": limits.max_video_seconds,
        "max_video_frames": limits.max_video_frames,
        "video_visual_only": True,
        "model_support_required": True,
    }

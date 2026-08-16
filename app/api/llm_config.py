"""用户 LLM 配置 API：provider 目录、配置 CRUD（掩码）、
一键获取模型列表。"""

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])

# 上下文窗口允许范围（token）：下限防误填过小立刻触发压缩，上限覆盖长窗口模型
_CTX_WINDOW_MIN, _CTX_WINDOW_MAX = 1024, 10_000_000


class LLMConfigRequest(BaseModel):
    user_id: str
    provider: str
    base_url: str
    model_id: str
    api_key: str
    is_default: bool = False
    context_window: int | None = Field(
        None, ge=_CTX_WINDOW_MIN, le=_CTX_WINDOW_MAX)   # None = 自动（探测/兜底）


class LLMModelsRequest(BaseModel):
    provider: str
    base_url: str
    api_key: str


class LLMConfigUpdateRequest(BaseModel):
    """更新已保存配置：只传要改的字段，None 表示保持原值。

    context_window 例外：0 = 清除（恢复自动），正整数 = 显式设置。
    """
    provider: str | None = None
    base_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    context_window: int | None = Field(None, ge=0, le=_CTX_WINDOW_MAX)

    @field_validator("context_window")
    @classmethod
    def _ctx_window_valid(cls, v: int | None) -> int | None:
        """0（清除）以外的正数必须落在允许区间——过小会轮轮触发压缩。"""
        if v is not None and v != 0 and v < _CTX_WINDOW_MIN:
            raise ValueError(f"context_window 需为 0（清除为自动）"
                             f"或 {_CTX_WINDOW_MIN} ~ {_CTX_WINDOW_MAX}")
        return v


async def _fetch_models(settings, provider: str, base_url: str,
                        api_key: str) -> list[dict]:
    """拉取某 provider 可选模型（附带上下文窗口探测）：不可列出
    （listable=false）用内置 catalog 兜底。

    返回 [{"id": str, "context_window": int | None}, ...]--
    模型元数据里带 context_length 等字段的平台（如 OpenRouter）能拿到真实窗口。
    """
    from app.abstractions.llm import extract_context_window
    catalog = settings.llm_providers.get(provider)
    if catalog is not None and not catalog.get("listable", True):
        return [{"id": m, "context_window": None}
                for m in catalog.get("catalog", [])]
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        return [{"id": m.get("id"), "context_window": extract_context_window(m)}
                for m in r.json().get("data", [])]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"provider error: {e.response.text[:200]}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"provider unreachable: {e}")


@router.get("/providers")
async def list_providers(request: Request):
    """provider 分类目录（配置内置）：前端做"选厂商"下拉框。"""
    return request.app.state.settings.llm_providers


@router.get("/configs")
async def list_configs(request: Request, user_id: str):
    """该用户的配置列表：api_key 只回显掩码（sk-...1234）。"""
    return await run_in_threadpool(request.app.state.llm_service.list_configs,
                                   user_id)


@router.post("/configs")
async def save_config(req: LLMConfigRequest, request: Request):
    """保存用户配置：api_key 加密落库；is_default 时互斥。"""
    try:
        config_id = await run_in_threadpool(
            request.app.state.llm_service.set_user_config,
            req.user_id, req.provider, req.base_url, req.model_id,
            req.api_key, req.is_default, req.context_window)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    return {"id": config_id, "api_key_masked": _mask(req.api_key)}


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


@router.delete("/configs/{config_id}")
async def delete_config(config_id: str, request: Request, user_id: str):
    """删除一条配置：仅本人可删。"""
    ok = await run_in_threadpool(request.app.state.llm_service.delete_config,
                                 user_id, config_id)
    if not ok:
        raise HTTPException(status_code=404, detail="config not found")
    return {"deleted": config_id}


@router.patch("/configs/{config_id}/default")
async def set_default_config(config_id: str, request: Request, user_id: str):
    """把已保存的配置切换为默认模型（互斥：清除该用户其他默认）。"""
    ok = await run_in_threadpool(request.app.state.llm_service.set_default_config,
                                 user_id, config_id)
    if not ok:
        raise HTTPException(status_code=404, detail="config not found")
    return {"default": config_id}


@router.post("/models")
async def list_models(req: LLMModelsRequest, request: Request):
    """一键获取模型列表：OpenAI 兼容 provider 调 {base_url}/models；
    不可列出（listable=false）则用内置 catalog 兜底。"""
    return await _fetch_models(request.app.state.settings,
                               req.provider, req.base_url, req.api_key)


@router.get("/configs/{config_id}/models")
async def list_config_models(config_id: str, request: Request, user_id: str):
    """某条已保存配置的可选模型列表：api_key 由后端解密调用，不离开服务端。

    前端切换模型时不需要重新输入 base_url / api_key。
    """
    svc = request.app.state.llm_service
    cfg = await run_in_threadpool(svc.get_config, user_id, config_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="config not found")
    return await _fetch_models(request.app.state.settings,
                               cfg.provider, cfg.base_url, cfg.api_key or "")


@router.patch("/configs/{config_id}")
async def update_config(config_id: str, req: LLMConfigUpdateRequest,
                        request: Request, user_id: str):
    """更新已保存配置（切换模型）：只传要改的字段，None 保持原值。"""
    ok = await run_in_threadpool(
        request.app.state.llm_service.update_config,
        user_id, config_id,
        provider=req.provider, base_url=req.base_url,
        model_id=req.model_id, api_key=req.api_key,
        context_window=req.context_window)
    if not ok:
        raise HTTPException(status_code=404, detail="config not found")
    return {"updated": config_id}
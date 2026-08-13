"""用户 LLM 配置 API：provider 目录、配置 CRUD（掩码）、
一键获取模型列表。"""

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


class LLMConfigRequest(BaseModel):
    user_id: str
    provider: str
    base_url: str
    model_id: str
    api_key: str
    is_default: bool = False


class LLMModelsRequest(BaseModel):
    provider: str
    base_url: str
    api_key: str


@router.get("/providers")
async def list_providers(request: Request):
    """provider 分类目录（配置内置）：前端做"选厂商"下拉框。"""
    return request.app.state.settings.llm_providers


@router.get("/configs")
async def list_configs(request: Request, user_id: str):
    """该用户的配置列表：api_key 只回显掩码（sk-...1234）。"""
    return request.app.state.llm_service.list_configs(user_id)


@router.post("/configs")
async def save_config(req: LLMConfigRequest, request: Request):
    """保存用户配置：api_key 加密落库；is_default 时互斥。"""
    try:
        config_id = request.app.state.llm_service.set_user_config(
            req.user_id, req.provider, req.base_url, req.model_id,
            req.api_key, req.is_default)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    return {"id": config_id, "api_key_masked": _mask(req.api_key)}


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


@router.post("/models")
async def list_models(req: LLMModelsRequest, request: Request):
    """一键获取模型列表：OpenAI 兼容 provider 调 {base_url}/models；
    不可列出（listable=false）则用内置 catalog 兜底。"""
    catalog = request.app.state.settings.llm_providers.get(req.provider)
    if catalog is not None and not catalog.get("listable", True):
        return catalog.get("catalog", [])
    url = f"{req.base_url.rstrip('/')}/models"
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {req.api_key}"},
                      timeout=10)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"provider error: {e.response.text[:200]}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"provider unreachable: {e}")
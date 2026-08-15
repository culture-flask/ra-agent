"""知识库 ：建库/列表/检索 + 多格式文档上传（后台异步入库）。"""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["kbs"])


class KBCreateRequest(BaseModel):
    name: str
    scope: str = "public"          # public | private
    user_id: str = "u1"
    texts: list[str] = []
    embedding_provider: str | None = None      # 每库可选嵌入模型
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    embedding_base_url: str | None = None      # 该库自定义嵌入端点（可选）
    embedding_api_key: str | None = None       # 该库专用嵌入密钥（可选，加密存储）

class KBRebuildRequest(BaseModel):
    embedding_provider: str | None = None      
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    embedding_base_url: str | None = None      # 不传则按 provider 继承/默认
    embedding_api_key: str | None = None       # 不传则沿用原库密钥

class KBEmbeddingUpdateRequest(BaseModel):
    """修改嵌入配置（创建后随时可改）：只更新传了的字段。"""
    embedding_provider: str | None = None
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    embedding_base_url: str | None = None      # 空串=清空回退 provider 默认端点
    embedding_api_key: str | None = None       # 空串=清除专用密钥

class KBRetrievalUpdateRequest(BaseModel):
    """允许/禁止该库被对话检索。"""
    enabled: bool

def _kb_dict(kb) -> dict:
    return {
        "kb_id": kb.kb_id, "name": kb.name, "scope": kb.scope,
        "owner_user_id": kb.owner_user_id, "category_id": kb.category_id,
        "embedding_provider": kb.embedding_provider,
        "embedding_model_id": kb.embedding_model_id,
        "embedding_dim": kb.embedding_dim, "status": kb.status,
        "embedding_base_url": kb.embedding_base_url,       # 端点非敏感，可回显便于编辑
        "has_embedding_key": bool(kb.embedding_api_key),   # 不下发明文，只告知是否设了专用密钥
        "embedded_model": kb.embedded_model,               # 向量实际由谁嵌入
        "embedding_mismatch": _kb_mismatch(kb),            # 查询模型 ≠ 嵌入模型时的提醒
        "retrieval_enabled": kb.retrieval_enabled,         # 是否允许被对话检索
        "source_doc_ids": kb.source_doc_ids or [],
        "created_at": kb.created_at.isoformat() if hasattr(kb.created_at, "isoformat") else str(kb.created_at),
    }

def _kb_mismatch(kb) -> str | None:
    """与 KBService.embedding_mismatch 相同逻辑；API 层免依赖 service 实例。"""
    em = kb.embedded_model or {}
    if not em:
        return None
    if (em.get("provider") == kb.embedding_provider
            and em.get("model_id") == kb.embedding_model_id
            and em.get("dim") == kb.embedding_dim):
        return None
    return (f"向量库由 {em.get('provider')}/{em.get('model_id')}（dim {em.get('dim')}）嵌入，"
            f"当前配置为 {kb.embedding_provider}/{kb.embedding_model_id}"
            f"（dim {kb.embedding_dim}），检索结果可能不准确；"
            f"如需向量匹配新模型，请重新上传文档或使用「重建」")


def _get_kb(request: Request, kb_id: str):
    try:
        return request.app.state.kb_service.get_kb(kb_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="kb not found")


@router.post("/kbs")
async def create_kb(req: KBCreateRequest, request: Request):
    try:
        kb = request.app.state.kb_service.create_kb(
            name=req.name, scope=req.scope, user_id=req.user_id, texts=req.texts,
            provider=req.embedding_provider, model_id=req.embedding_model_id,
            dim=req.embedding_dim, api_key=req.embedding_api_key,
            base_url=req.embedding_base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))   # 未知 provider 等
    return _kb_dict(kb)


@router.get("/kbs")
async def list_kbs(request: Request, user_id: str = "u1"):
    return [_kb_dict(kb) for kb in request.app.state.kb_service.list_kbs(user_id)]


@router.get("/kbs/{kb_id}")
async def get_kb(kb_id: str, request: Request):
    """查单个 KB（含入库状态，轮询入库进度用）。"""
    return _kb_dict(_get_kb(request, kb_id))


@router.get("/kbs/{kb_id}/search")
async def search_kb(kb_id: str, query: str, request: Request, k: int = 5,
                    user_id: str = "u1", mode: str | None = None):
    """检索测试：mode=vector（纯向量）| hybrid（向量+BM25），默认全局配置。"""
    kb = _get_kb(request, kb_id)
    mode = mode or request.app.state.settings.retrieval_mode
    return request.app.state.kb_service.search(kb.kb_id, query, k=k,
                                               user_id=user_id, mode=mode)


@router.post("/kbs/{kb_id}/documents")
async def upload_documents(kb_id: str, request: Request,
                           background: BackgroundTasks,
                           files: list[UploadFile] = File(...)):
    """批量文档上传：一次请求可带多个文件（multipart 字段名均为 files）。

    流程：status=indexing → 后台 parse/chunk/embed → status=ready|failed。
    轮询 GET /kbs/{kb_id} 观察进度（整批一次状态流转，不逐文件分状态）。
    用 FastAPI BackgroundTasks（响应返回后由框架可靠执行，测试可预期）；
    生产环境可换成 Celery/ARQ Worker（第 10 天）。
    """
    kb = _get_kb(request, kb_id)
    if kb.status == "indexing":
        # 上一批还在后台入库（远端嵌入模型可能很慢）：拒绝并发，避免两个任务抢状态/抢写向量库
        raise HTTPException(status_code=409,
                            detail="already indexing: 上一批文件还在入库中，请等待完成后再上传")
    payload: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        if not content:
            continue                     # 跳过空文件，其余照常入库
        payload.append((f.filename or "upload.bin", content))
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")
    kb_service = request.app.state.kb_service
    background.add_task(kb_service.ingest_files, kb.kb_id, payload)
    return {"kb_id": kb.kb_id, "status": "indexing",
            "count": len(payload),
            "filenames": [name for name, _ in payload],
            "filename": payload[0][0]}   # 兼容旧调用方

@router.patch("/kbs/{kb_id}/embedding")
async def update_kb_embedding(kb_id: str, req: KBEmbeddingUpdateRequest,
                              request: Request):
    """修改知识库的嵌入配置（provider/model/端点/密钥），创建后随时可改。

    已入库的向量不会重新嵌入：若换了模型，接口返回的 embedding_mismatch
    会提醒检索结果可能不准确（重新上传或重建可让向量匹配新模型）。
    """
    _get_kb(request, kb_id)   # 先确认库存在，404 优先
    try:
        kb = request.app.state.kb_service.update_embedding(
            kb_id, provider=req.embedding_provider,
            model_id=req.embedding_model_id, dim=req.embedding_dim,
            base_url=req.embedding_base_url, api_key=req.embedding_api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _kb_dict(kb)


@router.patch("/kbs/{kb_id}/retrieval")
async def update_kb_retrieval(kb_id: str, req: KBRetrievalUpdateRequest,
                              request: Request):
    """允许/禁止该库被对话检索（库本身保留，可随时恢复）。"""
    _get_kb(request, kb_id)
    kb = request.app.state.kb_service.set_retrieval(kb_id, req.enabled)
    return _kb_dict(kb)


@router.post("/kbs/{kb_id}/rebuild")
async def rebuild_kb(kb_id: str, req: KBRebuildRequest, request: Request):
    """复制原 chunk + 新嵌入模型重新向量化 → 新 KB，旧 KB 保留。"""
    kb = _get_kb(request, kb_id)
    try:
        new_kb = request.app.state.kb_service.rebuild(
            kb.kb_id, provider=req.embedding_provider,
            model_id=req.embedding_model_id, dim=req.embedding_dim,
            api_key=req.embedding_api_key, base_url=req.embedding_base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _kb_dict(new_kb)


@router.delete("/kbs/{kb_id}")
async def delete_kb(kb_id: str, request: Request, user_id: str = "u1"):
    """删除知识库：私人库仅属主可删；清理向量库 / 磁盘 chunk / 元数据。"""
    kb = _get_kb(request, kb_id)
    if kb.scope == "private" and kb.owner_user_id and kb.owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="forbidden: not kb owner")
    request.app.state.kb_service.delete_kb(kb_id)
    return {"deleted": kb_id}

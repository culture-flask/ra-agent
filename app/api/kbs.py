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

class KBRebuildRequest(BaseModel):
    embedding_provider: str | None = None      
    embedding_model_id: str | None = None
    embedding_dim: int | None = None

def _kb_dict(kb) -> dict:
    return {
        "kb_id": kb.kb_id, "name": kb.name, "scope": kb.scope,
        "owner_user_id": kb.owner_user_id, "category_id": kb.category_id,
        "embedding_provider": kb.embedding_provider,
        "embedding_model_id": kb.embedding_model_id,
        "embedding_dim": kb.embedding_dim, "status": kb.status,
        "source_doc_ids": kb.source_doc_ids or [],
        "created_at": kb.created_at.isoformat() if hasattr(kb.created_at, "isoformat") else str(kb.created_at),
    }


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
            dim=req.embedding_dim)
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
                    user_id: str = "u1"):
    kb = _get_kb(request, kb_id)
    return request.app.state.kb_service.search(kb.kb_id, query, k=k, user_id=user_id)


@router.post("/kbs/{kb_id}/documents")
async def upload_document(kb_id: str, request: Request,
                          background: BackgroundTasks,
                          file: UploadFile = File(...)):
    """多格式文档上传：解析入库放后台执行，立即返回 status=indexing。

    流程：status=indexing → 后台 parse/chunk/embed → status=ready|failed。
    轮询 GET /kbs/{kb_id} 观察进度。
    用 FastAPI BackgroundTasks（响应返回后由框架可靠执行，测试可预期）；
    生产环境可换成 Celery/ARQ Worker（第 10 天）。
    """
    kb = _get_kb(request, kb_id)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    kb_service = request.app.state.kb_service
    background.add_task(kb_service.ingest_file, kb.kb_id,
                        file.filename or "upload.bin", content)
    return {"kb_id": kb.kb_id, "status": "indexing", "filename": file.filename}

@router.post("/kbs/{kb_id}/rebuild")
async def rebuild_kb(kb_id: str, req: KBRebuildRequest, request: Request):
    """复制原 chunk + 新嵌入模型重新向量化 → 新 KB，旧 KB 保留。"""
    kb = _get_kb(request, kb_id)
    try:
        new_kb = request.app.state.kb_service.rebuild(
            kb.kb_id, provider=req.embedding_provider,
            model_id=req.embedding_model_id, dim=req.embedding_dim)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _kb_dict(new_kb)

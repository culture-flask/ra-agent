"""知识库 ：建库/列表/检索 + 多格式文档上传（后台异步入库）。"""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.cancel import request_cancel

router = APIRouter(prefix="/api/v1", tags=["kbs"])


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""          # 知识库介绍：必填，LLM 选库参考
    scope: str = "public"          # public | private
    user_id: str = "u1"
    texts: list[str] = []
    embedding_provider: str | None = None      # 每库可选嵌入模型
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    embedding_base_url: str | None = None      # 该库自定义嵌入端点（可选）
    embedding_api_key: str | None = None       # 该库专用嵌入密钥（可选，加密存储）

class KBRebuildRequest(BaseModel):
    mode: str = "reembed"               # reembed=重新向量化（可换模型）| copy=完全复制（不过嵌入）
    embedding_provider: str | None = None
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    embedding_base_url: str | None = None      # 不传则按 provider 继承/默认
    embedding_api_key: str | None = None       # 不传则沿用原库密钥
    user_id: str = "u1"                        # 重建/复制发起者：新库强制归其私人所有

class KBEmbeddingUpdateRequest(BaseModel):
    """修改嵌入配置（创建后随时可改）：只更新传了的字段。"""
    embedding_provider: str | None = None
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    embedding_base_url: str | None = None      # 空串=清空回退 provider 默认端点
    embedding_api_key: str | None = None       # 空串=清除专用密钥

class KBUpdateRequest(BaseModel):
    """修改知识库名称/介绍（创建后随时可改）：传了才改。"""
    name: str | None = None
    description: str | None = None


class KBRetrievalUpdateRequest(BaseModel):
    """允许/禁止该库被对话检索。"""
    enabled: bool

def _kb_dict(kb, user_id: str | None = None) -> dict:
    return {
        "kb_id": kb.kb_id, "name": kb.name,
        "description": kb.description or "",   # 知识库介绍
        "scope": kb.scope,
        "owner_user_id": kb.owner_user_id, "category_id": kb.category_id,
        "embedding_provider": kb.embedding_provider,
        "embedding_model_id": kb.embedding_model_id,
        "embedding_dim": kb.embedding_dim, "status": kb.status,
        "embedding_base_url": kb.embedding_base_url,       # 端点非敏感，可回显便于编辑
        "has_embedding_key": bool(kb.embedding_api_key),   # 不下发明文，只告知是否设了专用密钥
        "embedded_model": kb.embedded_model,               # 向量实际由谁嵌入
        "embedding_mismatch": _kb_mismatch(kb),            # 查询模型 ≠ 嵌入模型时的提醒
        "retrieval_enabled": kb.retrieval_enabled,         # 库主全局允许检索
        # 当前用户视角的检索开关（per-user，其他用户的禁用不影响此值）
        "retrieval_enabled_for_user": (
            kb.retrieval_enabled
            and user_id not in (kb.retrieval_disabled_users or [])),
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


async def _get_kb(request: Request, kb_id: str):
    """取 KB：Postgres 读是阻塞 I/O，放线程池避免卡住事件循环。"""
    try:
        return await run_in_threadpool(request.app.state.kb_service.get_kb, kb_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="kb not found")


@router.post("/kbs")
async def create_kb(req: KBCreateRequest, request: Request):
    """建库：知识库介绍必填（不能为空），供 LLM 选库时判断相关性。"""
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="知识库介绍不能为空")
    try:
        kb = await run_in_threadpool(
            request.app.state.kb_service.create_kb,
            name=req.name, scope=req.scope, user_id=req.user_id, texts=req.texts,
            provider=req.embedding_provider, model_id=req.embedding_model_id,
            dim=req.embedding_dim, api_key=req.embedding_api_key,
            base_url=req.embedding_base_url, description=req.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))   # 未知 provider 等
    return _kb_dict(kb)


@router.get("/kbs")
async def list_kbs(request: Request, user_id: str = "u1"):
    kbs = await run_in_threadpool(request.app.state.kb_service.list_kbs, user_id)
    return [_kb_dict(kb, user_id) for kb in kbs]


@router.get("/kbs/{kb_id}")
async def get_kb(kb_id: str, request: Request, user_id: str = "u1"):
    """查单个 KB（含入库状态，轮询入库进度用）。"""
    return _kb_dict(await _get_kb(request, kb_id), user_id)


@router.patch("/kbs/{kb_id}")
async def update_kb(kb_id: str, req: KBUpdateRequest, request: Request):
    """修改知识库名称/介绍（创建后随时可改）。"""
    await _get_kb(request, kb_id)   # 404 优先
    try:
        kb = await run_in_threadpool(
            request.app.state.kb_service.update_info,
            kb_id, name=req.name, description=req.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _kb_dict(kb)


@router.get("/kbs/{kb_id}/search")
async def search_kb(kb_id: str, query: str, request: Request, k: int = 5,
                    user_id: str = "u1", mode: str | None = None):
    """检索测试：mode=vector（纯向量）| hybrid（向量+BM25），默认全局配置。"""
    kb = await _get_kb(request, kb_id)
    mode = mode or request.app.state.settings.retrieval_mode
    return await run_in_threadpool(request.app.state.kb_service.search,
                                   kb.kb_id, query, k=k,
                                   user_id=user_id, mode=mode)


class KBDeleteRequest(BaseModel):
    doc_ids: list[str] = []


@router.get("/kbs/{kb_id}/files")
async def list_kb_files(kb_id: str, request: Request):
    """该库的源文件列表：原始文件名、片段数、页码范围（删除管理用）。"""
    kb = await _get_kb(request, kb_id)
    return await run_in_threadpool(request.app.state.kb_service.list_documents, kb.kb_id)


@router.post("/kbs/{kb_id}/documents/delete")
async def delete_kb_documents(kb_id: str, req: KBDeleteRequest, request: Request):
    """按 doc_id 批量删除源文件：Chroma 向量 + 磁盘 chunk + 元数据同步清理。"""
    await _get_kb(request, kb_id)
    result = await run_in_threadpool(request.app.state.kb_service.delete_documents,
                                     kb_id, req.doc_ids)
    return {"kb_id": kb_id, **result}


@router.post("/kbs/{kb_id}/documents")
async def upload_documents(kb_id: str, request: Request,
                           background: BackgroundTasks,
                           files: list[UploadFile] = File(...)):
    """批量文档上传：一次请求可带多个文件（multipart 字段名均为 files）。

    流程：status=indexing -> 后台逐文件 解析/分块/嵌入（单文件失败不影响
    其他）-> status=ready|failed（>=1 个成功即 ready）。
    轮询 GET /kbs/{kb_id}/ingest 看逐文件进度与错误明细。
    用 FastAPI BackgroundTasks（响应返回后由框架可靠执行，测试可预期）；
    生产环境可换成 Celery/ARQ Worker（第 10 天）。
    """
    kb = await _get_kb(request, kb_id)
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


@router.get("/kbs/{kb_id}/ingest")
async def get_ingest_progress(kb_id: str, request: Request):
    """入库进度：{total, done, succeeded, failed, status, files:[{filename,
    status, chunks, error}]}，前端轮询画进度条 + 展示逐文件错误明细。

    没有入库记录（新建库未上传过 / 服务重启后）返回 null。
    """
    await _get_kb(request, kb_id)
    return await run_in_threadpool(
        request.app.state.kb_service.ingest_progress, kb_id)


@router.post("/kbs/{kb_id}/ingest/cancel")
async def cancel_ingest(kb_id: str, request: Request):
    """终止正在进行的入库：设置取消标记，后台任务在文件边界/向量化前
    检查命中，清残留 chunk 并回滚状态（已成功文件保留）。"""
    await _get_kb(request, kb_id)
    request_cancel(kb_id)
    return {"cancelled": True, "kb_id": kb_id}


@router.post("/kbs/{kb_id}/rebuild/cancel")
async def cancel_rebuild(kb_id: str, request: Request):
    """终止正在进行的重建/复制（按 new_kb_id）：任务在批次边界检查，
    命中后删除半成品重建库（目录/向量/元数据全清）。"""
    await _get_kb(request, kb_id)
    svc = request.app.state.kb_service
    prog = svc.rebuild_progress(kb_id)
    if prog and prog.get("status") in ("reembedding", "copying"):
        request_cancel(kb_id)
        return {"cancelled": True, "kb_id": kb_id}
    return {"cancelled": False, "kb_id": kb_id,
            "detail": "该库没有正在进行的重建/复制"}

@router.patch("/kbs/{kb_id}/embedding")
async def update_kb_embedding(kb_id: str, req: KBEmbeddingUpdateRequest,
                              request: Request):
    """修改知识库的嵌入配置（provider/model/端点/密钥），创建后随时可改。

    已入库的向量不会重新嵌入：若换了模型，接口返回的 embedding_mismatch
    会提醒检索结果可能不准确（重新上传或重建可让向量匹配新模型）。
    """
    await _get_kb(request, kb_id)   # 先确认库存在，404 优先
    try:
        kb = await run_in_threadpool(
            request.app.state.kb_service.update_embedding,
            kb_id, provider=req.embedding_provider,
            model_id=req.embedding_model_id, dim=req.embedding_dim,
            base_url=req.embedding_base_url, api_key=req.embedding_api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _kb_dict(kb)


@router.patch("/kbs/{kb_id}/retrieval")
async def update_kb_retrieval(kb_id: str, req: KBRetrievalUpdateRequest,
                              request: Request, user_id: str = "u1"):
    """允许/禁止【当前用户】的对话检索此库（per-user，不影响其他用户）。"""
    await _get_kb(request, kb_id)
    kb = await run_in_threadpool(request.app.state.kb_service.set_retrieval,
                                 kb_id, user_id, req.enabled)
    return _kb_dict(kb, user_id)


@router.post("/kbs/{kb_id}/rebuild")
async def rebuild_kb(kb_id: str, req: KBRebuildRequest, request: Request,
                     background: BackgroundTasks):
    """复制原 chunk 生成新 KB，旧 KB 保留。两种模式：

    - reembed（默认）：新嵌入模型重新向量化（可换模型）
    - copy：完全复制——向量数据原样搬运，不调嵌入 API（秒级，配置原样继承）

    后台异步执行，立即返回 {status, new_kb_id}；前端轮询
    GET /kbs/{new_kb_id}/rebuild-progress 看进度。
    新库强制为发起者的「私人」库（不继承原库 scope）——不污染公共库。
    """
    await _get_kb(request, kb_id)
    svc = request.app.state.kb_service
    new_kb_id = uuid.uuid4().hex[:12]
    if req.mode == "copy":
        background.add_task(svc.copy_kb, kb_id, req.user_id, new_kb_id)
        return {"status": "copying", "new_kb_id": new_kb_id}
    # reembed：先同步校验嵌入配置（非法 provider 立刻 400，不排队进后台）
    try:
        svc.resolve_embedding_meta(req.embedding_provider,
                                   req.embedding_model_id,
                                   req.embedding_dim,
                                   req.embedding_base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    background.add_task(svc.rebuild, kb_id,
                        provider=req.embedding_provider,
                        model_id=req.embedding_model_id,
                        dim=req.embedding_dim,
                        api_key=req.embedding_api_key,
                        base_url=req.embedding_base_url,
                        user_id=req.user_id, new_kb_id=new_kb_id)
    return {"status": "reembedding", "new_kb_id": new_kb_id}


@router.get("/kbs/{kb_id}/rebuild-progress")
async def get_rebuild_progress(kb_id: str, request: Request):
    """重建/复制进度：{status, phase, total, done, pct, error}；无记录返回 null。"""
    await _get_kb(request, kb_id)
    return await run_in_threadpool(
        request.app.state.kb_service.rebuild_progress, kb_id)


@router.delete("/kbs/{kb_id}")
async def delete_kb(kb_id: str, request: Request, user_id: str = "u1"):
    """删除知识库：私人库仅属主可删；清理向量库 / 磁盘 chunk / 元数据。"""
    kb = await _get_kb(request, kb_id)
    if kb.scope == "private" and kb.owner_user_id and kb.owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="forbidden: not kb owner")
    await run_in_threadpool(request.app.state.kb_service.delete_kb, kb_id)
    return {"deleted": kb_id}

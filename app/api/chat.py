import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import RemoveMessage
from pydantic import BaseModel, Field

from app.core.cancel import clear_stop, request_stop
from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.core.events import clear_event_sink, set_event_sink
from app.core.logging import get_logger
from app.core.tokens import estimate_tokens
from app.graph.nodes import (
    extract_memory_node,
    save_memory_node,
)
from app.models import Conversation, User, utcnow
from app.services.parsing import parse_file

logger = get_logger("chat")

# 路由级闸门：本组全部端点要求登录态（含 /chat/files 附件上传）。
# 端点内再次声明 get_current_user 时 FastAPI 按请求缓存依赖结果，不会重复查库。
router = APIRouter(prefix="/api/v1", tags=["chat"],
                   dependencies=[Depends(get_current_user)])


class ChatRequest(BaseModel):
    # 归属用户取自 Bearer token（P0-1 鉴权改造），请求体不再携带 user_id
    session_id: str = "s1"
    message: str = Field(min_length=1)
    # 分支会话：新 thread 的初始历史（选中消息及之前），由前端随首次请求传入
    history: list[dict] = Field(default_factory=list)
    # 重新生成：先回退 checkpoint 里最后一条回复，再基于最后一条用户消息重生成
    rewind: bool = False
    # 检索模式：vector（纯向量）| hybrid（向量+BM25）；None = 全局默认
    retrieval_mode: str | None = None
    # 检索数量：每库几条 / 合并后总共几条；None 或 0 = 全局默认
    per_kb_k: int | None = None
    total_k: int | None = None
    # 生成温度：0~2；None = 默认 0.3
    temperature: float | None = None
    # 聚合返回：展开的父块数（0=关闭，全返回小 chunk）；None = 全局默认
    parent_groups: int | None = None
    # 对话附件：/chat/files 上传解析后返回的 file_id 列表，内容拼进本轮用户消息
    attachments: list[str] = []
    # 会话标题：本地已改名的会话首次登记时用（None=自动取首条消息截断）
    title: str | None = None


def _initial_state(req: ChatRequest, user_id: str,
                   append_message: bool = True) -> dict:
    """组装图初始状态。

    - user_id 来自鉴权依赖（token 注入），非客户端可指定
    - history：分支会话首次请求携带的历史（dict → LangChain 兼容消息格式）
    - attachments：附件全文拼进本轮用户消息（进 checkpoint，后续追问 LLM 仍可引用）
    - append_message=False（rewind 场景）：不追加用户消息——checkpoint 里
      已回退到最后一条用户消息，直接以其为起点重新生成
    """
    messages = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in req.history]
    message = req.message
    if append_message and req.attachments:
        blocks = []
        for fid in req.attachments:
            att = _CHAT_FILES.get(fid)
            if att:
                blocks.append(f"[附件：{att['filename']}]\n{att['text']}")
        if blocks:
            # 问题在前便于提取会话标题；标记分隔附件块，历史读取时可拆出附件名
            message = message + _ATT_MARK + "\n\n".join(blocks)
    if append_message:
        messages.append({"role": "user", "content": message})
    return {
        "user_id": user_id,
        "session_id": req.session_id,
        "query": req.message,
        "messages": messages,
        # 每轮对话重置检索状态：checkpointer 会保留上一轮的 retrievals，
        # 不显式清空会让上一轮的知识库结果泄漏进本轮的系统提示词
        "retrievals": [],
        "needs_retrieval": False,
        "selected_kb_ids": [],
        "retrieval_mode": req.retrieval_mode or "",
        "per_kb_k": req.per_kb_k or 0,
        "total_k": req.total_k or 0,
        "temperature": req.temperature,
        "parent_groups": req.parent_groups,
    }


# ---------- 对话附件：上传解析后暂存，随下一轮消息拼给 LLM ----------
_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024     # 单文件上传上限 20MB
_ATTACHMENT_MAX_CHARS = 100_000              # 单附件解析文本上限（防小窗口模型爆窗）
_CHAT_FILES_MAX = 100                        # 进程内暂存条数上限（FIFO 淘汰最旧）
_ATT_MARK = "\n<<<附件>>>\n"                 # 用户消息里问题与附件块的边界标记

_CHAT_FILES: dict[str, dict] = {}            # file_id -> {filename, text}


def _register_conversation(user_id: str, session_id: str, title: str) -> None:
    """登记/刷新会话（conversations 表，跨设备同步列表用）。

    title 只在首次登记时落（首条用户消息截断），之后仅刷新 updated_at。
    """
    with SessionLocal() as db:
        row = db.get(Conversation, session_id)
        if row is None:
            db.add(Conversation(session_id=session_id, user_id=user_id,
                                title=(title or "新会话")[:128]))
        else:
            row.updated_at = utcnow()
        db.commit()


def _conv_title(messages: list) -> str:
    """从图状态消息里取首条用户消息生成会话标题（与前端截断规则一致）。"""
    for m in messages:
        if getattr(m, "type", "") == "human":
            text = str(getattr(m, "content", "")).strip().replace("\n", " ")
            return text[:24] + "…" if len(text) > 24 else text
    return "新会话"


# ---------- P1-8：记忆管线后台化 ----------
# 强引用集合防后台任务被 GC（asyncio 只持弱引用，任务完成后自动移除）
_BG_TASKS: set = set()


def _spawn_memory_pipeline(app, uid: str, result: dict,
                           temperature=None) -> None:
    """答案生成完（SSE 已推 done）之后，在后台任务补跑记忆抽取/落库。

    此前 extract/save 是图的两个节点，graph.ainvoke 要等它们做完才返回——
    用户看到最后一个 token 后还要白等一次 LLM 抽取往返（2~10 秒）才收到
    done 事件。现在移出关键路径：感知延迟归零；失败仅记日志（记忆是
    锦上添花，绝不影响主对话——与节点内既有的降级原则一致）。
    """
    ctx = getattr(app.state, "workflow_ctx", None)
    if ctx is None or ctx.memory_service is None:
        return

    async def _pipeline():
        state = {"user_id": uid,
                 "messages": result.get("messages", []),
                 "temperature": temperature}
        out = await extract_memory_node(ctx, state)
        state.update(out)
        await save_memory_node(ctx, state)

    try:
        task = asyncio.create_task(_pipeline())
    except RuntimeError:
        return                                  # 无运行中事件循环（极端时序），放弃本次抽取
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


@router.post("/chat/files")
async def upload_chat_files(files: list[UploadFile] = File(...)):
    """批量上传对话附件：解析出纯文本暂存服务端，返回 file_id。

    附件与知识库入库无关（不向量化、不持久化）——只把文本拼进发送消息的
    用户消息里，供 LLM 阅读；进程内暂存，重启即失。逐文件返回成功/失败，
    单个文件解析失败不影响其余。
    """
    out = []
    for f in files:
        name = f.filename or "upload.bin"
        try:
            content = await f.read()
            if not content:
                raise ValueError("空文件")
            if len(content) > _ATTACHMENT_MAX_BYTES:
                raise ValueError("文件超过 20MB 上限")
            text = await run_in_threadpool(parse_file, name, content)
            text = text[:_ATTACHMENT_MAX_CHARS]           # 超长截断
            file_id = uuid.uuid4().hex[:12]
            _CHAT_FILES[file_id] = {"filename": name, "text": text}
            while len(_CHAT_FILES) > _CHAT_FILES_MAX:     # FIFO 淘汰
                _CHAT_FILES.pop(next(iter(_CHAT_FILES)))
            out.append({"file_id": file_id, "filename": name, "chars": len(text)})
        except ValueError as e:
            out.append({"filename": name, "error": str(e)})
    return {"files": out}


async def _context_usage(graph, config: dict, window: int) -> dict:
    """当前会话上下文占用：窗口上限 + 实际 token 用量（LLM 响应带回）+ 占用比例。

    优先用上一次 generate 的真实用量（usage_metadata：input+output，含系统
    提示词与检索结果，比字符估算准）；没有时（新会话/假模型/平台不回传）
    回退 core.tokens.estimate_tokens 粗估，source 字段标明数据来源。
    """
    snap = await graph.aget_state(config)
    values = (snap.values or {}) if snap else {}
    msgs = list(values.get("messages", []))
    usage = values.get("last_usage") or {}
    if usage.get("total_tokens"):
        used, source = int(usage["total_tokens"]), "llm"
    else:
        used, source = (estimate_tokens(msgs) if msgs else 0), "estimated"
    return {"window": window, "used_tokens": used,
            "ratio": round(used / window * 100, 1) if window else 0,
            "source": source}


async def _rewind_to_last_user(graph, config: dict) -> bool:
    """重新生成前的回退：把 checkpoint 消息截断到最后一条用户消息。

    丢弃其后的 assistant/tool 消息（旧回复或失败残留），返回是否执行了回退。
    注意：messages 键走 add_messages reducer，直接传列表是"追加"而非替换，
    必须用 RemoveMessage 按 id 删除要丢弃的消息。
    """
    snap = await graph.aget_state(config)
    stored = list((snap.values or {}).get("messages", [])) if snap else []
    if not stored:
        return False
    keep = list(stored)
    while keep and getattr(keep[-1], "type", "") != "human":
        keep.pop()
    if not keep:
        return False
    keep_ids = {m.id for m in keep if getattr(m, "id", None)}
    removes = [RemoveMessage(id=m.id) for m in stored
               if getattr(m, "id", None) and m.id not in keep_ids]
    if removes:
        await graph.aupdate_state(config, {"messages": removes})
    return True


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/chat/context")
async def chat_context(session_id: str, request: Request,
                       user: User = Depends(get_current_user)):
    """当前会话上下文占用（打开/切换会话时前端拉取；对话中由 SSE context 事件刷新）。

    会话已登记且不属于当前用户 → 403（未登记视为首轮进行中，放行返回空占用）。
    """
    with SessionLocal() as db:
        row = db.get(Conversation, session_id)
    if row is not None and row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not conversation owner")
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}
    svc = request.app.state.llm_service
    if hasattr(svc, "context_window_for"):
        # 可能触发同步 httpx /models 探测，放线程池避免阻塞事件循环
        window = await run_in_threadpool(svc.context_window_for, user.id)
    else:
        window = int(request.app.state.settings.llm_context_window)
    return await _context_usage(graph, config, window)


class StopRequest(BaseModel):
    session_id: str


@router.post("/chat/stop")
async def chat_stop(req: StopRequest, user: User = Depends(get_current_user)):
    """用户请求终止【自己的】会话当前生成：只设标记，generate 节点在下一个
    token 边界优雅退出（已生成的部分答复保留并写入 checkpoint）。

    属主校验：登记行存在且属于他人 → 403。行不存在（首轮对话尚未登记，
    见 conversations 登记时机）放行——停止标记本身无害，登记提前后豁口闭合。
    """
    with SessionLocal() as db:
        row = db.get(Conversation, req.session_id)
        if row is not None and row.user_id != user.id:
            raise HTTPException(status_code=403, detail="not conversation owner")
    request_stop(req.session_id)
    return {"stopped": True, "session_id": req.session_id}


@router.post("/chat")
async def chat(req: ChatRequest, request: Request,
               user: User = Depends(get_current_user)):
    """普通对话：跑完整图，返回答复与检索结果。

    rewind=true（重新生成）：先回退到最后一条用户消息，不追加新消息。
    身份取自 Bearer token（P0-1）。
    """
    uid = user.id
    graph = request.app.state.graph
    clear_stop(req.session_id)                 # 上一轮残留的停止标记不带入本轮
    config = {"configurable": {"thread_id": req.session_id},   # 会话级隔离
              "recursion_limit": 100}
    if req.rewind:
        rewound = await _rewind_to_last_user(graph, config)
        # 无 checkpoint 历史（新 thread）时无法回退 → 按普通提问追加消息
        initial = _initial_state(req, uid, append_message=not rewound)
    else:
        initial = _initial_state(req, uid)
    result = await graph.ainvoke(initial, config=config)
    # 登记/刷新会话（跨设备同步会话列表）
    try:
        await run_in_threadpool(_register_conversation, uid,
                                req.session_id,
                                req.title or _conv_title(result.get("messages", [])))
    except Exception as e:
        logger.warning("conversation register failed: %s", e)
    # P1-8：记忆管线后台补跑（不阻塞响应返回）
    _spawn_memory_pipeline(request.app, uid, result,
                           temperature=req.temperature)
    return {
        "answer": result.get("answer", ""),
        "retrievals": result.get("retrievals", []),
    }


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request,
                      user: User = Depends(get_current_user)):
    """SSE 流式：图在后台任务里跑，端点持续 drain 事件队列实时推送。

    token（generate 节点逐字 emit）与图内事件（supervisor/retrieve/memory/trace）
    都经同一队列流出——不依赖 LangGraph messages 模式的 token 转发（节点未接
    config，该模式拿不到逐 token 回调），因此用事件总线更可靠。
    身份取自 Bearer token（P0-1）；前端用 fetch+getReader 消费，可带 header。
    """
    uid = user.id
    graph = request.app.state.graph
    sink: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": req.session_id},   # 会话级隔离
              "recursion_limit": 100}

    async def run_graph():
        set_event_sink(sink)                     # 在本任务上下文里挂队列，节点 emit 可见
        clear_stop(req.session_id)               # 上一轮残留的停止标记不带入本轮
        try:
            if req.rewind:
                rewound = await _rewind_to_last_user(graph, config)
                # 无 checkpoint 历史（新 thread）时无法回退 → 按普通提问追加消息
                initial = _initial_state(req, uid, append_message=not rewound)
            else:
                initial = _initial_state(req, uid)
            result = await graph.ainvoke(initial, config=config)
            # 登记/刷新会话（跨设备同步会话列表；失败不影响对话）
            try:
                await asyncio.to_thread(_register_conversation, uid,
                                        req.session_id,
                                        req.title or _conv_title(result.get("messages", [])))
            except Exception as e:
                logger.warning("conversation register failed: %s", e)
            # 上下文占用推流：窗口（模型响应探测/默认）+ 压缩后的消息估测 tokens
            try:
                svc = request.app.state.llm_service
                if hasattr(svc, "context_window_for"):
                    window = await run_in_threadpool(svc.context_window_for,
                                                     uid)
                else:
                    window = int(request.app.state.settings.llm_context_window)
                await sink.put({"type": "context",
                                **await _context_usage(graph, config, window)})
            except Exception as e:
                logger.warning("context usage emit failed: %s", e)
            await sink.put({"type": "__done__"})
            # P1-8：done 已推给前端，记忆抽取在独立后台任务补跑——
            # 本任务随后被 event_gen 结束/取消也不影响它（独立 Task），
            # 其 emit 落入已无人消费的队列，无害。
            _spawn_memory_pipeline(request.app, uid, result,
                                   temperature=req.temperature)
        except Exception as e:
            from app.abstractions.llm import _is_quota_exhausted
            msg = str(e)
            if _is_quota_exhausted(e):
                # 额度用尽不是限流，重试无效——直接告诉用户去平台解决
                msg = "模型账户额度已用尽（insufficient_quota），请在模型平台提升工作区额度后重试。原始错误：" + msg
            await sink.put({"type": "__error__", "error": msg})
        finally:
            clear_event_sink()

    async def event_gen():
        task = asyncio.create_task(run_graph())
        try:
            while True:
                ev = await sink.get()            # 持续取事件：来一个发一个，实时
                t = ev.get("type")
                if t == "__done__":
                    break
                if t == "__error__":
                    yield _sse({"type": "error", "error": ev.get("error", "")})
                    break
                yield _sse(ev)
            yield _sse({"type": "done"})
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_gen(), media_type="text/event-stream")
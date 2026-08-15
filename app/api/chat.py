import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import RemoveMessage
from pydantic import BaseModel, Field

from app.core.events import clear_event_sink, set_event_sink
from app.core.logging import get_logger
from app.graph.nodes import _estimate_tokens

logger = get_logger("chat")

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str = "u1"
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


def _initial_state(req: ChatRequest, append_message: bool = True) -> dict:
    """组装图初始状态。

    - history：分支会话首次请求携带的历史（dict → LangChain 兼容消息格式）
    - append_message=False（rewind 场景）：不追加用户消息——checkpoint 里
      已回退到最后一条用户消息，直接以其为起点重新生成
    """
    messages = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in req.history]
    if append_message:
        messages.append({"role": "user", "content": req.message})
    return {
        "user_id": req.user_id,
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


async def _context_usage(graph, config: dict, window: int) -> dict:
    """当前会话上下文占用：窗口上限 + 消息 token 估测（自动压缩后）+ 占用比例。"""
    snap = await graph.aget_state(config)
    msgs = list((snap.values or {}).get("messages", [])) if snap else []
    used = _estimate_tokens(msgs) if msgs else 0
    return {"window": window, "used_tokens": used,
            "ratio": round(used / window * 100, 1) if window else 0}


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
async def chat_context(user_id: str, session_id: str, request: Request):
    """当前会话上下文占用（打开/切换会话时前端拉取；对话中由 SSE context 事件刷新）。"""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}
    svc = request.app.state.llm_service
    if hasattr(svc, "context_window_for"):
        # 可能触发同步 httpx /models 探测，放线程池避免阻塞事件循环
        window = await run_in_threadpool(svc.context_window_for, user_id)
    else:
        window = int(request.app.state.settings.llm_context_window)
    return await _context_usage(graph, config, window)


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """普通对话：跑完整图，返回答复与检索结果。

    rewind=true（重新生成）：先回退到最后一条用户消息，不追加新消息。
    """
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": req.session_id},   # 会话级隔离
              "recursion_limit": 100}
    if req.rewind:
        rewound = await _rewind_to_last_user(graph, config)
        # 无 checkpoint 历史（新 thread）时无法回退 → 按普通提问追加消息
        initial = _initial_state(req, append_message=not rewound)
    else:
        initial = _initial_state(req)
    result = await graph.ainvoke(initial, config=config)
    return {
        "answer": result.get("answer", ""),
        "retrievals": result.get("retrievals", []),
    }


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式：图在后台任务里跑，端点持续 drain 事件队列实时推送。

    token（generate 节点逐字 emit）与图内事件（supervisor/retrieve/memory/trace）
    都经同一队列流出——不依赖 LangGraph messages 模式的 token 转发（节点未接
    config，该模式拿不到逐 token 回调），因此用事件总线更可靠。
    """
    graph = request.app.state.graph
    sink: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": req.session_id},   # 会话级隔离
              "recursion_limit": 100}

    async def run_graph():
        set_event_sink(sink)                     # 在本任务上下文里挂队列，节点 emit 可见
        try:
            if req.rewind:
                rewound = await _rewind_to_last_user(graph, config)
                # 无 checkpoint 历史（新 thread）时无法回退 → 按普通提问追加消息
                initial = _initial_state(req, append_message=not rewound)
            else:
                initial = _initial_state(req)
            await graph.ainvoke(initial, config=config)
            # 上下文占用推流：窗口（模型响应探测/默认）+ 压缩后的消息估测 tokens
            try:
                svc = request.app.state.llm_service
                if hasattr(svc, "context_window_for"):
                    window = await run_in_threadpool(svc.context_window_for,
                                                     req.user_id)
                else:
                    window = int(request.app.state.settings.llm_context_window)
                await sink.put({"type": "context",
                                **await _context_usage(graph, config, window)})
            except Exception as e:
                logger.warning("context usage emit failed: %s", e)
            await sink.put({"type": "__done__"})
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
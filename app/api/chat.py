import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.events import clear_event_sink, set_event_sink

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str = "u1"
    session_id: str = "s1"
    message: str = Field(min_length=1)


def _initial_state(req: ChatRequest) -> dict:
    return {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "query": req.message,
        "messages": [{"role": "user", "content": req.message}],
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """普通对话：跑完整图，返回答复与检索结果。"""
    graph = request.app.state.graph
    result = await graph.ainvoke(
        _initial_state(req),
        config={"configurable": {"thread_id": req.session_id}},   # 会话级隔离
    )
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

    async def run_graph():
        set_event_sink(sink)                     # 在本任务上下文里挂队列，节点 emit 可见
        try:
            await graph.ainvoke(
                _initial_state(req),
                config={"configurable": {"thread_id": req.session_id}},   # 会话级隔离
            )
            await sink.put({"type": "__done__"})
        except Exception as e:
            await sink.put({"type": "__error__", "error": str(e)})
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
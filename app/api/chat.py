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
    """SSE 流式：token 实时推送 + 图内事件（supervisor/retrieve）实时推送。"""
    graph = request.app.state.graph
    sink: asyncio.Queue = asyncio.Queue()

    async def event_gen():
        set_event_sink(sink)
        try:
            async for ev in graph.astream(
                    _initial_state(req),
                    config={"configurable": {"thread_id": req.session_id}},
                    stream_mode="messages"):                     # 消息级流式
                while not sink.empty():                          # 先把图内事件发出去
                    yield _sse(sink.get_nowait())
                msg, meta = ev
                if meta.get("langgraph_node") != "generate":
                    continue                                     # 只流 generate 的 token
                text = msg.content
                if isinstance(text, str):
                    tokens = [text]
                elif isinstance(text, list):
                    tokens = [p.get("text", "") for p in text if isinstance(p, dict)]
                else:
                    tokens = []
                for t in tokens:
                    if t:
                        yield _sse({"type": "token", "content": t})
            yield _sse({"type": "done"})
        finally:
            clear_event_sink()

    return StreamingResponse(event_gen(), media_type="text/event-stream")
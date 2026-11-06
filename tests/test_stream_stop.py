"""真实 ChatOpenAI 与 Runnable 包装链在停止时应当场关闭传输流。"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.abstractions.llm import RetryableChatModel
from app.abstractions.reasoning import ReasoningChatOpenAI
from app.core.cancel import clear_stop, request_stop
from app.core.events import clear_event_sink, set_event_sink
from app.graph.native_llm import native_round


@pytest.mark.parametrize("wrapped", [False, True])
def test_stop_closes_wire_stream_before_return(monkeypatch, wrapped):
    """触发 stop 的下一块不进入答案；断流清理不应等待 GC 或下一轮事件循环。"""
    import openai

    sid = f"stop-close-{wrapped}"

    class WireStream:
        def __init__(self):
            self.index = 0
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            await self.close()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return {"choices": [{"index": 0, "finish_reason": None, "delta": {
                    "role": "assistant", "content": "保留的正文",
                    "reasoning_content": "模型推理",
                    "tool_calls": [{"index": 0, "id": "partial-call", "type": "function",
                        "function": {"name": "search", "arguments": '{"q":'}}],
                }}]}
            if self.index == 2:
                request_stop(sid)
                return {"choices": [{"index": 0, "finish_reason": None,
                    "delta": {"content": "不应保留的下一块"}}]}
            raise StopAsyncIteration

        async def close(self):
            self.closed = True

    stream = WireStream()

    class Completions:
        async def create(self, **kwargs):
            return stream

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    model = ReasoningChatOpenAI(model="m", api_key="k", base_url="http://t/v1",
                                max_retries=0, stream_usage=True)
    if wrapped:
        model = RetryableChatModel(model)
    tools = [{"name": "search", "description": "检索", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}}}]

    async def exercise():
        clear_stop(sid)
        sink = asyncio.Queue()
        set_event_sink(sink)
        try:
            response, usage, stopped = await native_round(
                None, [HumanMessage(content="研究问题")], tools, 0.3, sid, model=model)
            facts = {"wrapped": wrapped, "closed_at_return": stream.closed,
                     "stopped": stopped, "content": response.content,
                     "tool_calls": response.tool_calls,
                     "invalid_tool_calls": response.invalid_tool_calls,
                     "additional_kwargs": response.additional_kwargs}
            assert stopped and response.content == "保留的正文", facts
            assert response.tool_calls == [] and response.invalid_tool_calls == [], facts
            assert "reasoning_content" not in response.additional_kwargs, facts
            # 在这里立即断言；不能先 await sleep 让异步生成器回收掩盖关闭延迟。
            assert stream.closed, facts
        finally:
            clear_stop(sid)
            clear_event_sink()
            # 回归失败时也让已经排队的异步生成器 finalizer 完成清理。
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    asyncio.run(exercise())

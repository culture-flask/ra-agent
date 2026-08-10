"""流式事件总线：把图运行中的事件推给当前请求的 SSE 队列。

原理：contextvars（上下文变量）——每个请求有自己独立的上下文，
set_event_sink 把本请求的队列"钉"进上下文；图节点里 emit() 读取上下文
把事件塞进队列；SSE 端点一边收 token 一边把队列里的自定义事件发出去。
"""

import asyncio
import contextvars

_event_sink: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "event_sink", default=None
)


def set_event_sink(sink: asyncio.Queue) -> None:
    _event_sink.set(sink)


def clear_event_sink() -> None:
    _event_sink.set(None)


def emit(kind: str, payload: dict) -> None:
    """将运行期事件（retrieve / tool ...）推入当前请求的流式队列。"""
    sink = _event_sink.get()
    if sink is not None:
        try:
            sink.put_nowait({"type": kind, **payload})
        except Exception:
            pass
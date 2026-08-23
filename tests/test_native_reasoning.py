"""generate 原生路径推理捕获：双方言优先级与降级行为。

覆盖 nodes._native_round 的字段提取约定：
- reasoning_content 优先（DeepSeek 系方言），命中即不再看 reasoning
- reasoning（新事实标准：字符串或分片数组）作为回退
- 两者皆无 → 不发 reasoning 事件，答案流照常（零命中降级）
"""
import asyncio
from types import SimpleNamespace

from app.core.events import clear_event_sink, set_event_sink
from app.graph.nodes import _native_round

# 与 test_graph 相同的常驻循环约定
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


CFG = SimpleNamespace(api_key="k", base_url="http://t/v1", model_id="m")


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        pass


def _install_stream(monkeypatch, chunks):
    """替换 openai.AsyncOpenAI：create() 直接返回预置 chunk 流。"""
    import openai

    class _Completions:
        async def create(self, **kwargs):
            return _Stream(chunks)

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)


def _delta(content=None, tool_calls=None, **extra):
    d = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    for k, v in extra.items():
        setattr(d, k, v)
    return d


def _drain(q: asyncio.Queue):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _round_with_events(monkeypatch, chunks):
    """跑一轮原生流式并回收全部 SSE 事件。

    传入元素若没有 choices 属性则视为 delta，自动包装成 chunk 形态。
    """
    def _wrap(c):
        if getattr(c, "choices", None) is None:
            return SimpleNamespace(choices=[SimpleNamespace(delta=c)], usage=None)
        return c

    chunks = [_wrap(c) for c in chunks]
    q = asyncio.Queue()
    set_event_sink(q)
    try:
        _install_stream(monkeypatch, chunks)
        resp, _usage, stopped = _run(_native_round(CFG, [], None, 0.3, "s1"))
    finally:
        clear_event_sink()
    events = _drain(q)
    reasoning = [e["content"] for e in events if e["type"] == "reasoning"]
    tokens = [e["content"] for e in events if e["type"] == "token"]
    return resp, reasoning, tokens


def test_priority_reasoning_content_wins(monkeypatch):
    """两字段同时在 → 只取 reasoning_content（有就不拿）。"""
    resp, reasoning, tokens = _round_with_events(monkeypatch, [
        _delta(reasoning_content="甲", reasoning="乙"),
        _delta(content="答案"),
    ])
    assert reasoning == ["甲"]
    assert tokens == ["答案"]
    assert resp.content == "答案"


def test_reasoning_fallback_string(monkeypatch):
    """只有新方言 reasoning（字符串）→ 正常捕获。"""
    resp, reasoning, _tokens = _round_with_events(monkeypatch, [
        _delta(reasoning="思考中"),
        _delta(content="ok"),
    ])
    assert reasoning == ["思考中"]
    assert resp.content == "ok"


def test_reasoning_fallback_list_shards(monkeypatch):
    """reasoning 为分片数组（部分网关形态）→ 拼接后透传。"""
    resp, reasoning, _tokens = _round_with_events(monkeypatch, [
        _delta(reasoning=[{"text": "x"}, {"text": "y"}]),
        _delta(content="done"),
    ])
    assert reasoning == ["xy"]


def test_reasoning_list_summary_key(monkeypatch):
    """分片数组用 summary 键（OpenRouter reasoning_details 形态）→ 兜底拼接。"""
    resp, reasoning, _tokens = _round_with_events(monkeypatch, [
        _delta(reasoning=[{"type": "summary", "summary": "概要"}]),
        _delta(content="done"),
    ])
    assert reasoning == ["概要"]


def test_neither_field_degrades_silently(monkeypatch):
    """两个字段都没有 → 零 reasoning 事件，token 流照常（正常降级）。"""
    resp, reasoning, tokens = _round_with_events(monkeypatch, [
        _delta(content="hi"),
    ])
    assert reasoning == []
    assert tokens == ["hi"]
    assert resp.content == "hi"

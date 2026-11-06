"""LangChain 流式适配器：双方言推理捕获与降级行为。

覆盖 nodes._native_round 的字段提取约定：
- reasoning_content 优先（DeepSeek 系方言），命中即不再看 reasoning
- reasoning（新事实标准：字符串或分片数组）作为回退
- 两者皆无 → 不发 reasoning 事件，答案流照常（零命中降级）
"""
import asyncio
from types import SimpleNamespace

from app.core.events import clear_event_sink, set_event_sink
from app.graph.native_llm import native_round as _native_round

# 与 test_graph 相同的常驻循环约定
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


CFG = SimpleNamespace(api_key="k", base_url="http://t/v1", model_id="m")


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            chunk = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(chunk, BaseException):
            raise chunk
        return _wire(chunk)

    async def close(self):
        self.closed = True


def _wire(value):
    """SDK 传输替身返回原始协议字典，交由真实 ChatOpenAI 转换。"""
    if isinstance(value, SimpleNamespace):
        return {key: _wire(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wire(item) for item in value]
    return value


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
            return SimpleNamespace(choices=[SimpleNamespace(delta=c, finish_reason=None)], usage=None)
        return c

    chunks = [_wrap(c) for c in chunks]
    # 收尾 chunk：finish_reason=stop 表示正常完成（否则截断检测会整轮重试）
    chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason="stop")],
                                  usage=None))
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

# ---------- 截断检测（P3-34 后续）：无 finish_reason/usage = 网关静默断流 ----------

def _install_multi_streams(monkeypatch, streams):
    """create() 按序返回预置的多个流（用于验证截断重试次数）。"""
    import openai

    calls = {"n": 0}

    class _Completions:
        async def create(self, **kwargs):
            i = min(calls["n"], len(streams) - 1)
            calls["n"] += 1
            return streams[i]

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    return calls


def test_truncated_stream_retries_once_then_partial(monkeypatch):
    """流静默断流（无 finish_reason/usage）→ 自动整轮重试一次；
    重试仍截断 → 接受部分内容（不抛异常、不无限循环）。"""
    def truncated_stream(text):
        return _Stream([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=_delta(content=text), finish_reason=None)], usage=None),
        ])

    s1 = truncated_stream("半截一")
    s2 = truncated_stream("半截二")
    calls = _install_multi_streams(monkeypatch, [s1, s2])

    q = asyncio.Queue()
    set_event_sink(q)
    try:
        resp, usage, stopped = _run(_native_round(CFG, [], None, 0.3, "s-trunc"))
    finally:
        clear_event_sink()

    assert calls["n"] == 2                    # 恰好重试一次
    assert resp.content == "半截二"           # 返回的是重试轮的部分内容
    assert usage is None                      # 截断流没有末块 usage
    assert stopped is False


def test_normal_finish_no_retry(monkeypatch):
    """有 finish_reason 的正常完成 → 不触发重试（create 仅一次）。"""
    stream = _Stream([
        SimpleNamespace(choices=[SimpleNamespace(
            delta=_delta(content="完整"), finish_reason=None)], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=None, finish_reason="stop")], usage=None),
    ])
    calls = _install_multi_streams(monkeypatch, [stream])

    q = asyncio.Queue()
    set_event_sink(q)
    try:
        resp, _usage, stopped = _run(_native_round(CFG, [], None, 0.3, "s-ok"))
    finally:
        clear_event_sink()

    assert calls["n"] == 1
    assert resp.content == "完整"


def test_real_langchain_conversion_and_callback_keep_reasoning(monkeypatch):
    """真实 bind_tools/astream/回调链保留网关推理字段，不伪造转换后消息。"""
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import HumanMessage

    from app.abstractions.reasoning import ReasoningChatOpenAI

    received = []

    class Capture(AsyncCallbackHandler):
        async def on_llm_new_token(self, token, *, chunk=None, **kwargs):
            received.append((token, dict(chunk.message.additional_kwargs)))

    stream = _Stream([
        {"choices": [{"index": 0, "delta": {"role": "assistant",
            "reasoning": [{"text": "模型思考"}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "答案"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
            "id": "call-search", "type": "function", "function": {
                "name": "search", "arguments": '{"q":"论文"}'}}]},
            "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2,
            "total_tokens": 7}},
    ])
    calls = _install_multi_streams(monkeypatch, [stream])
    model = ReasoningChatOpenAI(api_key="k", base_url="http://t/v1", model="m",
                                temperature=0.3, max_retries=0)
    bound = model.bind_tools([{"name": "search", "description": "检索",
                              "parameters": {"type": "object", "properties": {
                                  "q": {"type": "string"}}}}])

    async def collect():
        chunks = []
        async for chunk in bound.astream([HumanMessage(content="研究问题")],
                                        config={"callbacks": [Capture()],
                                                "metadata": {"session_id": "wire-test"}}):
            chunks.append(chunk)
        return chunks

    chunks = _run(collect())
    assert calls["n"] == 1 and stream.closed
    assert any(extra.get("reasoning_content") == "模型思考" for _, extra in received)
    assert any(token == "答案" for token, _ in received)
    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk
    assert full.content == "答案"
    assert full.tool_calls[0]["name"] == "search"
    assert full.tool_calls[0]["args"] == {"q": "论文"}
    assert full.usage_metadata["total_tokens"] == 7


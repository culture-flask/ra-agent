"""LLM 退避重试测试：假模型 + monkeypatch sleep，不依赖网络与真实模型。"""

import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.abstractions.llm import RetryableChatModel, _is_retryable


class RateLimitError(Exception):
    """模拟 openai.RateLimitError（429）。"""
    status_code = 429


class ServerError(Exception):
    """模拟 5xx。"""
    status_code = 500


class AuthError(Exception):
    """模拟 401，重试无意义。"""
    status_code = 401


class FlakyModel:
    """前 fail_times 次调用抛 exc，之后正常；记录调用次数。"""

    def __init__(self, exc, fail_times=1, answer="ok"):
        self._exc = exc
        self._fail_times = fail_times
        self._answer = answer
        self.calls = 0
        self.model_name = "flaky"

    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return AIMessage(content=self._answer)

    async def astream(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            yield AIMessage(content="半截内容")   # 流中断前已吐出的部分
            raise self._exc
        yield AIMessage(content=self._answer)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """禁用退避等待，让测试即时完成。"""
    async def _noop(_):
        pass
    monkeypatch.setattr(asyncio, "sleep", _noop)


def _wrap(model, max_retries=3):
    return RetryableChatModel(model, max_retries=max_retries,
                              base_delay=1.0, label="flaky")


# ---------- 异常分类 ----------
def test_retryable_classification():
    assert _is_retryable(httpx.TimeoutException("t"))
    assert _is_retryable(httpx.ConnectError("c"))
    assert _is_retryable(RateLimitError())        # 429
    assert _is_retryable(ServerError())           # 5xx
    assert not _is_retryable(AuthError())         # 4xx 不重试
    assert not _is_retryable(ValueError("bad"))


# ---------- ainvoke ----------
def test_retry_succeeds_after_transient_errors():
    """网络错误重试 2 次后成功：总共调用 3 次，返回正常结果。"""
    flaky = FlakyModel(httpx.ConnectError("boom"), fail_times=2)
    model = _wrap(flaky)
    resp = asyncio.run(model.ainvoke([HumanMessage(content="hi")]))
    assert resp.content == "ok"
    assert flaky.calls == 3


def test_retry_on_rate_limit_and_server_error():
    """429 / 5xx 都重试。"""
    for exc in (RateLimitError(), ServerError()):
        flaky = FlakyModel(exc, fail_times=1)
        resp = asyncio.run(_wrap(flaky).ainvoke([HumanMessage(content="hi")]))
        assert resp.content == "ok"
        assert flaky.calls == 2


def test_no_retry_on_auth_error():
    """401 等不可重试错误：只调 1 次，原样抛出。"""
    flaky = FlakyModel(AuthError(), fail_times=99)
    with pytest.raises(AuthError):
        asyncio.run(_wrap(flaky).ainvoke([HumanMessage(content="hi")]))
    assert flaky.calls == 1


def test_retries_exhausted_raises():
    """始终失败且重试耗尽：调用 max_retries+1 次后抛原异常。"""
    flaky = FlakyModel(httpx.TimeoutException("t"), fail_times=99)
    with pytest.raises(httpx.TimeoutException):
        asyncio.run(_wrap(flaky, max_retries=2).ainvoke([HumanMessage(content="hi")]))
    assert flaky.calls == 3


# ---------- astream ----------
def test_astream_retries_whole_stream():
    """流式中断：整轮重试，第二次完整输出。

    已 yield 给调用方的半截内容无法收回（异步生成器语义），
    前端会短暂看到重复开头——这是整轮重试的预期代价。
    """
    flaky = FlakyModel(httpx.RemoteProtocolError("broken"), fail_times=1)
    model = _wrap(flaky)

    async def collect():
        chunks = []
        async for c in model.astream([HumanMessage(content="hi")]):
            chunks.append(str(c.content))
        return chunks

    chunks = asyncio.run(collect())
    assert chunks == ["半截内容", "ok"]         # 首次半截 + 重试后的完整结果
    assert flaky.calls == 2                     # 重试确实发生


def test_bind_tools_keeps_retry():
    """bind_tools 之后仍保留重试能力（工具循环的第二次生成）。"""
    flaky = FlakyModel(RateLimitError(), fail_times=1)
    model = _wrap(flaky).bind_tools([{"name": "add", "description": "x"}])
    resp = asyncio.run(model.ainvoke([HumanMessage(content="hi")]))
    assert resp.content == "ok"
    assert flaky.calls == 2
    assert model.model_name == "flaky"          # 模型名透传

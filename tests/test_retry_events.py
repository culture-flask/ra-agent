"""重试事件透出测试：native_round 网络重试/截断重试 → retry 事件（前缀 +
agent 标注 + 倒计时秒数）；agent_speak 重试耗尽 → agent_fail 事件（失败原因）。"""

import asyncio
import httpx
from types import SimpleNamespace

from app.core.events import clear_event_sink, set_event_sink
from app.graph import agent_runtime
from app.graph.native_llm import native_round, retry_event_name

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
_run = _loop.run_until_complete

CFG = SimpleNamespace(api_key="k", base_url="http://t/v1", model_id="m")


async def _nosleep(_):
    return None


def _chunk(content="", reasoning=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning,
                            tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FlakyStream:
    """前 fail_times 次迭代抛可重试传输错误，之后正常吐 chunks。"""

    def __init__(self, chunks, fail_times):
        self._chunks, self._fail_times = list(chunks), fail_times

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise httpx.ReadTimeout("read timed out")
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        pass


def _install_flaky(monkeypatch, chunks, fail_times):
    """前 fail_times 次 create() 调用都返回"首块即抛可重试错误"的流
    （流中断后整轮废弃重建，所以失败次数按调用次数计），之后正常。"""
    import openai
    calls = {"n": 0}

    class _Completions:
        async def create(self, **kwargs):
            calls["n"] += 1
            return _FlakyStream(chunks, fail_times if calls["n"] <= fail_times
                                else 0)

    class _Client:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    return calls


def _drain(sink) -> list[dict]:
    out = []
    while not sink.empty():
        out.append(sink.get_nowait())
    return out


def test_retry_event_name_mapping():
    assert retry_event_name("token") == "retry"
    assert retry_event_name("bs_token") == "bs_retry"
    assert retry_event_name("sem_token") == "sem_retry"
    assert retry_event_name("dr_token") == "dr_retry"
    assert retry_event_name("bs_internal") == "bs_retry"


def test_native_round_emits_retry_events(monkeypatch):
    """流中断重试：每次重试发 retry 事件（attempt/max/delay/error 齐全）。"""
    monkeypatch.setattr(asyncio, "sleep", _nosleep)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                            prompt_tokens_details=None,
                            prompt_cache_hit_tokens=None)
    _install_flaky(monkeypatch,
                   [_chunk("你"), _chunk("好", finish="stop", usage=usage)],
                   fail_times=2)
    sink = asyncio.Queue()
    set_event_sink(sink)
    try:
        resp, _usage, stopped = _run(native_round(CFG, [], None, 0.3, "s1"))
    finally:
        clear_event_sink()
    assert resp.content == "你好" and not stopped
    retries = [e for e in _drain(sink) if e["type"] == "retry"]
    assert len(retries) == 2
    assert retries[0]["attempt"] == 1 and retries[0]["max"] == 5
    assert retries[0]["delay"] == 1 and retries[1]["delay"] == 2
    assert "timed out" in retries[0]["error"]


def test_retry_event_prefix_and_agent(monkeypatch):
    """多 Agent：事件带团队前缀 + agent 标注（气泡路由依据）。"""
    monkeypatch.setattr(asyncio, "sleep", _nosleep)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                            prompt_tokens_details=None,
                            prompt_cache_hit_tokens=None)
    _install_flaky(monkeypatch,
                   [_chunk("答", finish="stop", usage=usage)], fail_times=1)
    sink = asyncio.Queue()
    set_event_sink(sink)
    try:
        _run(native_round(CFG, [], None, 0.3, "s1", token_event="bs_token",
                          reasoning_event="bs_reasoning",
                          emit_extra={"agent": "innovator"}))
    finally:
        clear_event_sink()
    retries = [e for e in _drain(sink) if str(e["type"]).endswith("retry")]
    assert len(retries) == 1
    assert retries[0]["type"] == "bs_retry"
    assert retries[0]["agent"] == "innovator"


def test_truncation_retry_emits_event(monkeypatch):
    """截断重试（无 finish_reason/usage）也发 retry 事件。"""
    monkeypatch.setattr(asyncio, "sleep", _nosleep)
    calls = _install_flaky(monkeypatch, [_chunk("半截")], fail_times=0)
    sink = asyncio.Queue()
    set_event_sink(sink)
    try:
        resp, _u, _s = _run(native_round(CFG, [], None, 0.3, "s1"))
    finally:
        clear_event_sink()
    assert calls["n"] == 2                       # 截断后整轮重试了一次
    assert resp.content == "半截"                # 重试仍截断 → 接受部分内容
    retries = [e for e in _drain(sink) if e["type"] == "retry"]
    assert len(retries) == 1 and "截断" in retries[0]["error"]


def test_agent_speak_emits_agent_fail(monkeypatch):
    """agent_speak 失败外抛前发 agent_fail（重试耗尽后的最终原因）。"""

    async def boom(*a, **k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(agent_runtime, "native_round", boom)
    sink = asyncio.Queue()
    set_event_sink(sink)
    ctx = SimpleNamespace(settings=SimpleNamespace(), llm_service=None,
                          kb_service=None, mcp_adapter=None)
    state = {"user_id": "u1", "session_id": "s1"}
    raised = False
    try:
        _run(agent_runtime.agent_speak(ctx, state, "innovator", "sys", "hi",
                                       0.5, None, 2,
                                       cfg=SimpleNamespace(model_id="m")))
    except RuntimeError:
        raised = True
    finally:
        clear_event_sink()
    assert raised
    fails = [e for e in _drain(sink) if e["type"] == "bs_agent_fail"]
    assert len(fails) == 1
    assert fails[0]["agent"] == "innovator"
    assert fails[0]["error"]                     # short_reason 产出非空


def test_control_calls_do_not_emit_retry(monkeypatch):
    """控制类调用（stream_events=False）不打扰前端：不发 retry 事件。"""
    monkeypatch.setattr(asyncio, "sleep", _nosleep)
    _install_flaky(monkeypatch, [_chunk("ok")], fail_times=1)
    sink = asyncio.Queue()
    set_event_sink(sink)
    try:
        _run(native_round(CFG, [], None, 0.3, "s1", token_event="bs_internal",
                          stream_events=False))
    finally:
        clear_event_sink()
    assert all(e["type"] != "bs_retry" for e in _drain(sink))


def test_llm_text_emits_retry_for_writer(monkeypatch):
    """llm_text 的 emit_retry 透传：溯源社撰写人（不流式）也能看到重试倒计时，
    空完成重试同样发事件（带 agent 路由到气泡）。"""
    calls = {"n": 0}

    async def fake_native_round(cfg, payload, tools, temperature, sid, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(content=""), None, False   # 空完成
        return SimpleNamespace(content="ok"), {"total_tokens": 7}, False

    monkeypatch.setattr(agent_runtime, "native_round", fake_native_round)
    monkeypatch.setattr(agent_runtime, "effective_cfg",
                        lambda ctx, uid: SimpleNamespace(model_id="m"))
    monkeypatch.setattr(asyncio, "sleep", _nosleep)
    sink = asyncio.Queue()
    set_event_sink(sink)
    try:
        text, _used = _run(agent_runtime.llm_text(
            None, {"user_id": "u1", "session_id": "s1"}, "sys", "hi",
            0.2, prefix="dr_", emit_retry=True, emit_extra={"agent": "writer"}))
    finally:
        clear_event_sink()
    assert text == "ok"
    retries = [e for e in _drain(sink) if e["type"] == "dr_retry"]
    assert len(retries) == 1
    assert retries[0]["agent"] == "writer"
    assert retries[0]["error"] == "输出为空，自动重试"


def test_dr_write_frame_emits_agent_fail(monkeypatch):
    """溯源社撰写人两段契约全空/异常 → dr_agent_fail 透出失败原因；
    重试透传参数（emit_retry/emit_extra）确实传给了底层调用。"""
    from app.graph.deep_research import make_write_frame_node
    calls = {"n": 0}

    async def fake_llm_text(ctx, state, sys_p, human, temperature=None,
                            prefix="dr_", **kw):
        calls["n"] += 1
        captured["kw"] = kw
        if calls["n"] == 1:
            raise RuntimeError("connection reset by peer")
        return "", 0                                   # 空输出 → 契约失败

    captured = {}
    monkeypatch.setattr("app.graph.deep_research._llm_text", fake_llm_text)
    monkeypatch.setattr("app.graph.deep_research._role_cfg",
                        lambda ctx, state, rid: SimpleNamespace(model_id="m"))
    sink = asyncio.Queue()
    set_event_sink(sink)
    ctx = SimpleNamespace(settings=SimpleNamespace())
    state = {"user_id": "u1", "session_id": "dr-rw", "topic": "课题",
             "roles": [], "outline": {"title": "报告"},
             "chapters": [{"index": 1, "title": "背景", "content": "c"}]}
    try:
        out = _run(make_write_frame_node(ctx)(state))
    finally:
        clear_event_sink()
    assert out["toc"] == "" and out["introduction"] == ""
    assert captured["kw"].get("emit_retry") is True
    assert captured["kw"].get("emit_extra") == {"agent": "writer"}
    evs = _drain(sink)
    fails = [e for e in evs if e["type"] == "dr_agent_fail"]
    assert len(fails) == 1 and fails[0]["agent"] == "writer"
    assert fails[0]["error"]                            # short_reason 产出非空
    assert any(e["type"] == "dr_agent_end" for e in evs)  # 节点照常收尾（发布员降级汇编）

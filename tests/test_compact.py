"""自动上下文压缩测试：轮数/窗口触发、总结拼接、失败兜底。"""

import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.core.tokens import estimate_tokens
from app.graph.nodes import (
    COMPACT_KEEP_ROUNDS,
    COMPACT_MIN_ROUNDS,
    COMPACT_MIN_TOKEN_RATIO,
    _build_system_prompt,
    _round_count,
    _split_keep_and_old,
)
from app.graph.state import AgentState
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


def _rounds(n: int, prefix: str = "轮"):
    """构造 n 轮消息（user+assistant 对）。"""
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"{prefix}提问{i}"))
        msgs.append(AIMessage(content=f"{prefix}回答{i}"))
    return msgs


# ---------- 辅助函数 ----------
def test_round_count_and_estimate():
    msgs = _rounds(5)
    assert _round_count(msgs) == 5
    assert estimate_tokens(msgs) > 0
    assert estimate_tokens([]) == 0


def test_split_keep_and_old():
    msgs = _rounds(10)
    keep, old = _split_keep_and_old(msgs, COMPACT_KEEP_ROUNDS)
    assert _round_count(keep) == 4
    assert _round_count(old) == 6
    assert keep[0].content == "轮提问6"           # 保留最近 4 轮原文
    assert old[-1].content == "轮回答5"


def test_build_system_prompt_includes_summary():
    state = {"conversation_summary": "用户在研究量子计算"}
    prompt = _build_system_prompt(state)
    assert "[历史对话总结]" in prompt
    assert "量子计算" in prompt


# ---------- 图内集成 ----------
class _FakeModel:
    def __init__(self, answer, route_json, summary):
        self._answer, self._route_json, self._summary = answer, route_json, summary

    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages):
        system = next((m.content for m in messages
                       if getattr(m, "type", "") == "system"), "")
        if "问答路由" in str(system):
            return AIMessage(content=self._route_json)
        if "对话总结助手" in str(system):
            return AIMessage(content=self._summary)
        return AIMessage(content=self._answer)

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class _FakeLLMService:
    def __init__(self, answer, route_json='{"needs_retrieval": false, "kbs": []}',
                 summary="历史总结：用户研究了量子计算与叠加态"):
        self._answer, self._route_json, self._summary = answer, route_json, summary

    def get_chat_model(self, user_id: str, temperature=None):
        return _FakeModel(self._answer, self._route_json, self._summary)


def _make_ctx_local(answer="压缩后回答", route_json='{"needs_retrieval": false, "kbs": []}'):
    from app.graph.nodes import WorkflowContext
    from app.services.kb_service import KBService
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    kb_service = KBService(settings)
    ctx = WorkflowContext(settings, _FakeLLMService(answer, route_json), kb_service)
    return ctx


def test_compact_triggers_over_20_rounds():
    """轮数 >20 且占用 ≥ 窗口 10% 触发压缩：旧轮次删除，总结进系统提示词。

    P3-25 后轮数路径有最小占用门槛，这里把窗口调小（500）：21 轮短消息
    约 190 字符 ≈ 95 tokens，落在 50（10%×500）~400（80%×500）之间，
    确保走的是轮数兜底触发而不是 token 主路径。
    """
    from app.graph.workflow import build_graph

    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "llm_context_window": 500,
    })
    from app.graph.nodes import WorkflowContext
    from app.services.kb_service import KBService
    ctx = WorkflowContext(settings, _FakeLLMService("压缩后回答"), KBService(settings))
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "compact-1"}}
    msgs = _rounds(COMPACT_MIN_ROUNDS + 1)          # 21 轮
    state = {"user_id": "u1", "session_id": "compact-1", "query": "继续",
             "messages": msgs}
    result = _run(graph.ainvoke(state, config=cfg))

    # 压缩后：conversation_summary 落状态，messages 只剩最近 4 轮
    assert result.get("conversation_summary") == "历史总结：用户研究了量子计算与叠加态"
    assert _round_count(result["messages"]) == 4     # 只保留最近 4 轮
    # 旧轮次确实被删除：保留的是第 18~21 轮（索引17起），最早的提问已不在
    contents = [m.content for m in result["messages"]]
    assert "轮提问17" in contents                    # 保留轮的第一条提问
    assert "轮提问16" not in contents                # 第 17 轮及之前已被总结替换


def test_no_compact_below_threshold():
    """轮数 ≤20 且 token 未达 80% → 不压缩。"""
    from app.graph.workflow import build_graph

    ctx = _make_ctx_local()
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "compact-2"}}
    msgs = _rounds(COMPACT_MIN_ROUNDS)              # 正好 20 轮
    result = _run(graph.ainvoke({"user_id": "u1", "session_id": "compact-2",
                                 "query": "继续", "messages": msgs}, config=cfg))
    assert not result.get("conversation_summary")
    assert _round_count(result["messages"]) == COMPACT_MIN_ROUNDS   # 全部保留


def test_no_compact_low_token_many_rounds():
    """P3-25：轮数超限但占用极低（< 窗口 10%）→ 不压缩，保住全部细节。

    默认 256k 窗口下 30 轮短对话仅占窗口千分之几——改造前这里会白白把
    26 轮历史总结掉；现在门槛生效，一条消息都不删。
    """
    from app.graph.workflow import build_graph

    ctx = _make_ctx_local()
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "compact-lowtok"}}
    result = _run(graph.ainvoke(
        {"user_id": "u1", "session_id": "compact-lowtok", "query": "继续",
         "messages": _rounds(COMPACT_MIN_ROUNDS + 10)}, config=cfg))
    assert not result.get("conversation_summary")           # 未产生总结
    assert _round_count(result["messages"]) == COMPACT_MIN_ROUNDS + 10  # 全部保留


def test_compact_triggers_on_window_80_percent():
    """token 估测达上下文窗口 80% → 即使轮数少也压缩。"""
    from app.graph.workflow import build_graph

    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
        "llm_context_window": 60,                    # 小窗口便于触发
    })
    from app.graph.nodes import WorkflowContext
    from app.services.kb_service import KBService
    ctx = WorkflowContext(settings, _FakeLLMService("答", summary="小窗总结"),
                          KBService(settings))
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "compact-3"}}
    # 5 轮 * 每条约 14 字符 ≈ 70 字符 ≈ 35 tokens > 60*0.8=48? 不够 → 用更长文本
    msgs = []
    for i in range(5):
        msgs.append(HumanMessage(content="这是一段用于撑大token估算的中文对话内容第%d号提问" % i))
        msgs.append(AIMessage(content="这是一段用于撑大token估算的中文回复内容第%d号回答" % i))
    result = _run(graph.ainvoke({"user_id": "u1", "session_id": "compact-3",
                                 "query": "继续", "messages": msgs}, config=cfg))
    assert result.get("conversation_summary") == "小窗总结"
    assert _round_count(result["messages"]) == COMPACT_KEEP_ROUNDS  # 保留最近 4 轮


def test_context_usage_after_graph_run():
    """对话一轮后：上下文占用 = 窗口上限 + 压缩后消息的 token 估测 + 比例。"""
    from app.api.chat import ChatRequest, _context_usage, _initial_state
    from app.graph.workflow import build_graph
    from app.core.tokens import estimate_tokens as _estimate_tokens

    ctx = _make_ctx_local()
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "ctx-usage-1"}}
    req = ChatRequest(session_id="ctx-usage-1", message="你好")
    _run(graph.ainvoke(_initial_state(req, "u1"), config=cfg))

    usage = _run(_context_usage(graph, cfg, window=32768))
    assert usage["window"] == 32768
    assert usage["used_tokens"] == _estimate_tokens(
        [m for m in _run(graph.aget_state(cfg)).values["messages"]])
    assert usage["used_tokens"] > 0                  # 有消息占用
    assert 0 <= usage["ratio"] < 100                 # 短消息比例趋近 0 也合法

    # 空 thread：占用为 0，比例 0
    empty = _run(_context_usage(graph, {"configurable": {"thread_id": "ctx-empty"}},
                                window=32768))
    assert empty["used_tokens"] == 0 and empty["ratio"] == 0


def test_chat_context_endpoint(monkeypatch, auth_factory):
    """GET /chat/context：打开会话时前端拉取上下文占用（不依赖 SSE 事件）。"""
    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "deepseek-v4-flash", "context_length": 32768}]}

        def raise_for_status(self):
            pass

    monkeypatch.setattr("app.abstractions.llm.httpx.get",
                        lambda url, headers=None, timeout=None: FakeResp())

    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.db import SessionLocal
    from app.models import User
    with SessionLocal() as db:
        if not db.get(User, "ctx-ep-user"):
            db.add(User(id="ctx-ep-user", username="ctx-ep-user", password_hash="x"))
            db.commit()
    with TestClient(app) as c:
        r = c.get("/api/v1/chat/context",
                  headers=auth_factory("ctx-ep-user"),
                  params={"session_id": "ctx-ep-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["window"] == 32768          # 探测到模型元数据里的窗口
        assert body["used_tokens"] == 0         # 空会话
        assert body["ratio"] == 0


def test_compact_failure_skips_gracefully():
    """压缩 LLM 调用失败 → 静默跳过，不阻断主对话。

    同样用小窗口（500）确保真的走到总结调用那一步——否则 P3-25 的
    低占用门槛会先把它拦下，测的就不是失败兜底了。
    """
    from app.graph.workflow import build_graph

    class BoomModel(_FakeModel):
        async def ainvoke(self, messages):
            system = next((m.content for m in messages
                           if getattr(m, "type", "") == "system"), "")
            if "对话总结助手" in str(system):
                raise RuntimeError("summarizer down")
            return AIMessage(content="回答")

    class BoomService:
        def get_chat_model(self, user_id: str, temperature=None):
            return BoomModel("回答", '{"needs_retrieval": false, "kbs": []}', "x")

    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
        "llm_context_window": 500,
    })
    from app.graph.nodes import WorkflowContext
    from app.services.kb_service import KBService
    ctx = WorkflowContext(settings, BoomService(), KBService(settings))
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "compact-4"}}
    result = _run(graph.ainvoke({"user_id": "u1", "session_id": "compact-4",
                                 "query": "继续", "messages": _rounds(25)}, config=cfg))
    assert result["answer"] == "回答"                # 主对话正常
    assert not result.get("conversation_summary")
    assert _round_count(result["messages"]) == 25    # 没有删除任何消息

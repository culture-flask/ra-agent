"""用量计量测试：流式 usage 虚高修复 + llm_usage 落库 + 汇总端点（P3-20）。"""

import asyncio

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, HumanMessage
from sqlalchemy import select

from app.core.db import SessionLocal
from app.graph.nodes import WorkflowContext, generate_node
from app.main import app
from app.models import LLMUsage
from app.services.kb_service import KBService
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


USAGE = {"input_tokens": 800, "output_tokens": 200, "total_tokens": 1000}


class DupUsageModel:
    """复刻问题供应商：流式时**每个 chunk 都带同一份 usage**。

    LangChain 的 AIMessageChunk 相加会累加 usage_metadata——聚合后的 resp
    携带 chunks 数 × 真实值，这正是"1M 窗口统计出千万级用量"的根因。
    """

    def __init__(self, n_chunks=5):
        self._n = n_chunks
        self.model_name = "dup-usage-model"

    def bind_tools(self, schemas):
        return self

    async def astream(self, messages):
        for i in range(self._n):
            yield AIMessageChunk(content="片段%d" % i,
                                 usage_metadata=dict(USAGE))


class DupLLMService:
    model_name = "dup-usage-model"

    def get_chat_model(self, user_id, temperature=None):
        return DupUsageModel()


def _make_ctx():
    settings = Settings.load()
    ctx = WorkflowContext(settings, DupLLMService(), KBService(settings),
                          None, None, None)
    return ctx


def _run_generate(user_id, session_id):
    ctx = _make_ctx()
    state = {"user_id": user_id, "session_id": session_id,
             "query": "你好", "messages": [HumanMessage(content="你好")],
             "temperature": None}
    return _run(generate_node(ctx, state))


def test_streaming_usage_not_inflated():
    """末块权威值修复：N 个重复 usage 的 chunk → 统计为单份而非 N 倍。"""
    result = _run_generate("usage-u1", "usage-s1")
    usage = result["last_usage"]
    assert usage["total_tokens"] == 1000          # 不是 5000（5 个 chunk 相加）
    assert usage["input_tokens"] == 800
    assert result["answer"] == "片段0片段1片段2片段3片段4"


def test_usage_persisted_to_db():
    """generate 后 best-effort 落库：llm_usage 出现一行真实值。"""
    _run_generate("usage-u2", "usage-s2")
    with SessionLocal() as db:
        rows = db.scalars(select(LLMUsage).where(
            LLMUsage.user_id == "usage-u2")).all()
        assert len(rows) == 1
        assert rows[0].model == "dup-usage-model"
        assert rows[0].total_tokens == 1000
        assert rows[0].input_tokens == 800


def test_summary_endpoint_scoped_and_aggregated(auth_factory):
    """/usage/summary：按天×模型聚合；仅本人可见；无 token 401。"""
    from datetime import timedelta

    from app.models.base import utcnow

    def seed(user, model, days_ago, total):
        with SessionLocal() as db:
            db.add(LLMUsage(user_id=user, session_id="s", model=model,
                            input_tokens=total // 2,
                            output_tokens=total - total // 2,
                            total_tokens=total,
                            created_at=utcnow() - timedelta(days=days_ago)))
            db.commit()

    seed("sum-u1", "model-a", 0, 100)
    seed("sum-u1", "model-b", 2, 300)
    seed("sum-other", "model-a", 0, 999)          # 他人数据不得泄漏

    with TestClient(app) as c:
        h = auth_factory("sum-u1")
        r = c.get("/api/v1/usage/summary?days=30", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["grand"]["total_tokens"] == 400
        assert body["grand"]["calls"] == 2
        assert len(body["items"]) == 2
        newest = body["items"][0]                  # 按日期倒序
        assert newest["date"] >= body["items"][-1]["date"]
        models_newest = {m["model"]: m["total_tokens"]
                         for m in newest["models"]}
        assert models_newest == {"model-a": 100}   # 他人的 999 不混入

    with TestClient(app) as c:
        assert c.get("/api/v1/usage/summary").status_code == 401


class EmptyThenGoodModel:
    """第一次流式调用返回纯空 chunk（复刻供应商空完成），第二次恢复正常。"""

    def __init__(self):
        self.calls = 0
        self.model_name = "flaky-empty"

    def bind_tools(self, schemas):
        return self

    async def astream(self, messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(content="")           # 零内容、零 tool_calls
        else:
            yield AIMessageChunk(content="恢复后的回答",
                                 usage_metadata=dict(USAGE))


class EmptyLLMService:
    def __init__(self):
        self._impl = None

    def get_chat_model(self, user_id, temperature=None):
        return self._impl


def test_empty_response_retried_and_recovered():
    """P3-31：首轮空响应 → 快速退避整轮重试 → 第二轮正常出内容。"""
    svc = EmptyLLMService()
    svc._impl = EmptyThenGoodModel()
    settings = Settings.load()
    ctx = WorkflowContext(settings, svc, KBService(settings), None, None, None)
    state = {"user_id": "empty-u1", "session_id": "empty-s1",
             "query": "q", "messages": [HumanMessage(content="q")],
             "temperature": None}
    result = _loop.run_until_complete(generate_node(ctx, state))
    assert result["answer"] == "恢复后的回答"
    assert svc._impl.calls == 2                        # 确实整轮重试过一次
    # 只有有效轮带 usage → 落库恰好一行真实值
    with SessionLocal() as db:
        rows = db.scalars(select(LLMUsage).where(
            LLMUsage.user_id == "empty-u1")).all()
        assert len(rows) == 1 and rows[0].total_tokens == 1000


class AlwaysEmptyModel:
    model_name = "always-empty"

    def bind_tools(self, schemas):
        return self

    async def astream(self, messages):
        yield AIMessageChunk(content="")


def test_empty_response_exhausted_raises(caplog):
    """P3-31：重试耗尽仍为空 → 显式抛错走 error 通道，绝不静默结束。"""
    svc = EmptyLLMService()
    svc._impl = AlwaysEmptyModel()
    settings = Settings.load()
    ctx = WorkflowContext(settings, svc, KBService(settings), None, None, None)
    state = {"user_id": "empty-u2", "session_id": "empty-s2",
             "query": "q", "messages": [HumanMessage(content="q")],
             "temperature": None}
    try:
        _loop.run_until_complete(generate_node(ctx, state))
        raise AssertionError("应当抛出 RuntimeError")
    except RuntimeError as e:
        assert "空响应" in str(e)

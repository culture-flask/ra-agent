"""长记忆测试：跨会话读取、用户隔离、写入前审核、API。"""

import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.core.db import SessionLocal
from app.core.tracing import Tracer
from app.graph.nodes import WorkflowContext
from app.models import User
from app.graph.workflow import build_graph
from app.services.kb_service import KBService
from app.services.memory_service import MemoryService
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _ensure_user(user_id: str):
    """memories.user_id 有外键指向 users，写记忆前先建用户（第 5 天同款）。"""
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()


def _run(coro):
    return _loop.run_until_complete(coro)


class ScriptedModel:
    """脚本化假模型：依次返回预置消息；可记录收到的 system 提示词。

    注意：responses 必须共享列表（不拷贝）——第 6 天踩过的坑：
    拷贝后每次 get_chat_model 都从第一条开始弹，后面的节点拿到错误消息。
    """

    def __init__(self, responses, captured=None):
        self._responses = responses          # 共享引用，不拷贝
        self._captured = captured

    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages):
        if self._captured is not None and messages:
            self._captured.append(str(messages[0].content))   # 记录 system prompt
        return self._responses.pop(0) if self._responses \
            else AIMessage(content='{"memories":[]}')

    async def astream(self, messages):
        yield await self.ainvoke(messages)    # generate 节点用 astream 逐 token 流式


class ScriptedLLMService:
    def __init__(self, responses, captured=None):
        self._responses = responses          # 共享
        self._captured = captured

    def get_chat_model(self, user_id: str, temperature=None):
        return ScriptedModel(self._responses, self._captured)


def _make_ctx(responses, captured=None):
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    kb_service = KBService(settings)
    tracer = Tracer()
    ms = MemoryService()
    ctx = WorkflowContext(settings, ScriptedLLMService(responses, captured),
                          kb_service, None, tracer, ms)
    return ctx, ms


def _chat(ctx, session_id: str, message: str):
    graph = _run(build_graph(ctx))
    return _run(graph.ainvoke(
        {"user_id": "u1", "session_id": session_id, "query": message,
         "messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": session_id}}))


def test_memory_saved_after_chat():
    _ensure_user("u1")
    """对话结束后抽取并落库：get_all 能看到记忆。"""
    ctx, ms = _make_ctx([
        AIMessage(content="好的"),
        AIMessage(content='{"memories":[{"key":"research_topic","value":"量子计算"}]}'),
    ])
    _chat(ctx, "m1", "我最近在研究量子计算")
    memory = ms.get_all("u1")
    assert memory["research_topic"] == {"v": "量子计算"}


def test_memory_cross_session_loaded():
    _ensure_user("u1")
    """跨会话：第二个会话的 system prompt 里带上了记忆（load_memory 生效）。"""
    captured = []
    ctx, ms = _make_ctx([
        AIMessage(content="好的"),
        AIMessage(content='{"memories":[{"key":"research_topic","value":"量子计算"}]}'),
    ], captured=captured)
    _chat(ctx, "m2a", "我最近在研究量子计算")

    ctx2, _ = _make_ctx([AIMessage(content="回答")], captured=captured)
    _chat(ctx2, "m2b", "你记得我的研究方向吗？")
    assert any("[用户记忆]" in p and "量子计算" in p for p in captured)


def test_memory_user_isolation():
    _ensure_user("u1")
    """用户级隔离：u1 的记忆对 u2 不可见。"""
    ctx, ms = _make_ctx([
        AIMessage(content="好"),
        AIMessage(content='{"memories":[{"key":"research_topic","value":"量子计算"}]}'),
    ])
    _chat(ctx, "m3", "我研究量子计算")
    assert ms.get_all("u1").get("research_topic")
    assert ms.get_all("u2") == {}


def test_review_rejects_bad_memory():
    """审核规则：非法键名/太短的值不落库。"""
    ctx, ms = _make_ctx([
        AIMessage(content="好"),
        AIMessage(content='{"memories":[{"key":"Bad Key!","value":"量子计算"},'
                           '{"key":"ok_key","value":"x"}]}'),
    ])
    _chat(ctx, "m4", "测试")
    assert ms.get_all("u1") == {}


def test_memory_api():
    """GET /api/v1/memories 返回用户记忆。"""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        r = c.get("/api/v1/memories", params={"user_id": "u1"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)

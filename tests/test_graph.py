import asyncio
import tempfile
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.core.db import SessionLocal
from app.graph.nodes import WorkflowContext
from app.graph.workflow import build_graph
from app.models import User
from app.services.kb_service import KBService
from app.settings import Settings


def _ensure_user(user_id: str):
    """kbs.owner_user_id 有外键指向 users，私人库测试前先建用户。"""
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()

# 关键：AsyncPostgresSaver 的连接绑定在构建它的事件循环上。
# asyncio.run() 每次新建并销毁循环，会导致"bound to a different event loop"。
# 所以全测试共享一个常驻循环。
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


class FakeLLMService:
    """替身 LLM 服务：get_chat_model 返回固定回答的假模型。"""

    def __init__(self, answer: str):
        self._answer = answer

    def get_chat_model(self, user_id: str):
        return FakeMessagesListChatModel(
            responses=[AIMessage(content=self._answer)])


def _make_ctx(answer: str = "这是假模型回答"):
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",   # 离线嵌入，测试确定性
    })
    kb_service = KBService(settings)
    ctx = WorkflowContext(settings, FakeLLMService(answer), kb_service)
    return ctx, kb_service


def _run_graph(graph, state: dict):
    return _run(graph.ainvoke(
        state, config={"configurable": {"thread_id": state["session_id"]}}))


def test_retrieve_chain_with_kb():
    """有知识库 → supervisor 走 retrieve → generate，检索结果带 scope 标签。"""
    ctx, kb_service = _make_ctx("答案")
    kb_service.create_kb("测试库", "public", "u1",
                         ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t1",
                                "query": "叠加态是什么",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert len(result["retrievals"]) == 1
    assert result["retrievals"][0]["scope"] == "public"
    assert result["retrievals"][0]["kb_name"] == "测试库"
    assert result["answer"] == "答案"


def test_no_kb_skips_retrieve():
    """没有知识库 → supervisor 直接走 generate，不检索。"""
    ctx, _ = _make_ctx("无库回答")
    graph = _run(build_graph(ctx))
    result = _run_graph(graph, {"user_id": "u1", "session_id": "t2",
                                "query": "你好",
                                "messages": [HumanMessage(content="你好")]})
    assert result.get("retrievals", []) == []
    assert result["answer"] == "无库回答"


def test_private_kb_not_visible_to_others():
    """u1 的私人库对 u2 不可见（越权防护在图层生效）。"""
    ctx, kb_service = _make_ctx("答案")
    _ensure_user("u1")
    kb_service.create_kb("u1私密", "private", "u1", ["我的实验pH=7.2"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u2", "session_id": "t3",
                                "query": "实验参数",
                                "messages": [HumanMessage(content="实验参数")]})
    assert result.get("retrievals", []) == []   # u2 什么都查不到


def test_checkpointer_continues_session():
    """同 thread_id 两次调用 → 消息累积（短期记忆/会话级隔离）。"""
    ctx, _ = _make_ctx("回答")
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "same-session"}}

    _run(graph.ainvoke({"user_id": "u1", "session_id": "x",
                        "query": "第一问",
                        "messages": [HumanMessage(content="第一问")]}, config=cfg))
    result = _run(graph.ainvoke({"user_id": "u1", "session_id": "x",
                                 "query": "第二问",
                                 "messages": [HumanMessage(content="第二问")]},
                                config=cfg))
    texts = [m.content for m in result["messages"]]
    assert "第一问" in texts and "第二问" in texts   # 历史消息都在
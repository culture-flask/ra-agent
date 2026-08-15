import asyncio
import tempfile
from pathlib import Path

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


class RouterAwareFakeModel:
    """替身模型：按系统提示词区分调用类型。

    - 系统提示词含「问答路由」→ 返回 route_json（模拟 LLM 意图判断）
    - 其他（生成 / 记忆抽取）→ 返回 answer
    """

    def __init__(self, answer: str, route_json: str):
        self._answer = answer
        self._route_json = route_json

    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages):
        system = next((m.content for m in messages
                       if getattr(m, "type", "") == "system"), "")
        if "问答路由" in str(system):
            return AIMessage(content=self._route_json)
        return AIMessage(content=self._answer)

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class FakeLLMService:
    """替身 LLM 服务：路由/生成按提示词分别返回预置内容。"""

    def __init__(self, answer: str, route_json: str):
        self._answer = answer
        self._route_json = route_json

    def get_chat_model(self, user_id: str):
        return RouterAwareFakeModel(self._answer, self._route_json)


def _make_ctx(answer: str = "这是假模型回答",
              route_json: str = '{"needs_retrieval": true, "kbs": []}'):
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",   # 离线嵌入，测试确定性
    })
    kb_service = KBService(settings)
    ctx = WorkflowContext(settings, FakeLLMService(answer, route_json), kb_service)
    return ctx, kb_service


def _run_graph(graph, state: dict):
    return _run(graph.ainvoke(
        state, config={"configurable": {"thread_id": state["session_id"]}}))


def test_retrieve_chain_with_kb():
    """有知识库 + LLM 判断需要检索 → 按选中的库检索，结果带 scope 标签。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "测试库", "scope": "public"}]}')
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


def test_llm_skips_retrieval_for_chitchat():
    """有知识库，但 LLM 判断无需检索（闲聊）→ 不检索直接生成。"""
    ctx, kb_service = _make_ctx(
        "你好呀", '{"needs_retrieval": false, "kbs": []}')
    kb_service.create_kb("测试库", "public", "u1",
                         ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t5",
                                "query": "你好",
                                "messages": [HumanMessage(content="你好")]})
    assert result.get("retrievals", []) == []
    assert result["answer"] == "你好呀"


def test_llm_selects_only_chosen_kbs():
    """LLM 只选公共库 → 私人库不被检索（不再全库覆盖）。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "公共文档", "scope": "public"}]}')
    _ensure_user("u1")
    kb_service.create_kb("公共文档", "public", "u1", ["公共资料：量子比特叠加态"])
    kb_service.create_kb("我的实验", "private", "u1", ["私人实验：pH=7.2"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t6",
                                "query": "叠加态",
                                "messages": [HumanMessage(content="叠加态")]})
    assert result["retrievals"]
    assert all(r["kb_name"] == "公共文档" for r in result["retrievals"])


def test_route_failure_falls_back_to_all_visible():
    """路由输出无法解析 → 降级为全部可见库检索（RAG 兜底）。"""
    ctx, kb_service = _make_ctx("答案", "抱歉，我无法理解你的意思")
    kb_service.create_kb("测试库", "public", "u1",
                         ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t7",
                                "query": "叠加态是什么",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert len(result["retrievals"]) == 1

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
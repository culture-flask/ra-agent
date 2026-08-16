"""会话跨设备同步：conversations 登记 + 列表/历史/删除 API。"""

import asyncio
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.chat import _ATT_MARK, _register_conversation
from app.api.conversations import _split_attachments
from app.core.db import SessionLocal, engine
from app.main import app
from app.models import Conversation
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

Conversation.__table__.create(engine, checkfirst=True)   # 幂等建表（登记先于 lifespan）


def _run(coro):
    return _loop.run_until_complete(coro)


def _del_rows(*ids):
    with SessionLocal() as db:
        for sid in ids:
            row = db.get(Conversation, sid)
            if row is not None:
                db.delete(row)
        db.commit()


def test_split_attachments():
    text = "总结一下" + _ATT_MARK + "[附件：a.pdf]\n正文内容"
    q, names = _split_attachments(text)
    assert q == "总结一下" and names == ["a.pdf"]
    q2, names2 = _split_attachments("普通消息")       # 无标记的老消息原样返回
    assert q2 == "普通消息" and names2 == []


def test_register_and_list():
    _register_conversation("conv-u1", "conv-list-a", "第一个会话")
    _register_conversation("conv-u1", "conv-list-b", "第二个会话")
    _register_conversation("conv-u2", "conv-list-c", "别人的会话")
    try:
        time.sleep(0.02)                            # 确保 updated_at 分得开
        _register_conversation("conv-u1", "conv-list-a", "不该覆盖")  # 仅刷新时间
        with TestClient(app) as c:
            r = c.get("/api/v1/conversations", params={"user_id": "conv-u1"})
            assert r.status_code == 200
            convs = r.json()["conversations"]
            ids = [x["session_id"] for x in convs]
            assert {"conv-list-a", "conv-list-b"} <= set(ids)
            assert "conv-list-c" not in ids          # 用户隔离
            a = next(x for x in convs if x["session_id"] == "conv-list-a")
            b = next(x for x in convs if x["session_id"] == "conv-list-b")
            assert a["title"] == "第一个会话"         # 标题不随刷新变化
            assert a["updated_at"] >= b["updated_at"]  # 最近活跃在前
    finally:
        _del_rows("conv-list-a", "conv-list-b", "conv-list-c")


class _FakeModel:
    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="历史回答")

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class _FakeLLMService:
    def get_chat_model(self, user_id, temperature=None):
        return _FakeModel()


def test_messages_history_and_delete():
    """假 LLM 图跑一轮（写真实 checkpointer）→ API 读回历史（附件拆 chips）→ 删除清理。"""
    from app.graph.nodes import WorkflowContext
    from app.graph.workflow import build_graph
    from app.services.kb_service import KBService

    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    ctx = WorkflowContext(settings, _FakeLLMService(), KBService(settings))
    graph = _run(build_graph(ctx))
    sid = "conv-hist-1"
    cfg = {"configurable": {"thread_id": sid}}
    user_content = "这是什么方法？" + _ATT_MARK + "[附件：note.txt]\n文件正文ABC"
    try:
        _run(graph.ainvoke({"user_id": "conv-u1", "session_id": sid,
                            "query": "这是什么方法？",
                            "messages": [{"role": "user", "content": user_content}]},
                           config=cfg))
        _register_conversation("conv-u1", sid, "这是什么方法？")
        with TestClient(app) as c:
            app.state.graph = graph                 # 换成假 LLM 图（同一 checkpointer）
            r = c.get(f"/api/v1/conversations/{sid}/messages",
                      params={"user_id": "conv-u1"})
            assert r.status_code == 200
            msgs = r.json()["messages"]
            assert msgs[0]["role"] == "user"
            assert msgs[0]["content"] == "这是什么方法？"     # 附件块拆出正文
            assert msgs[0]["files"] == [{"name": "note.txt"}]  # 还原为文件 chips
            assert msgs[-1]["role"] == "assistant"
            assert msgs[-1]["content"] == "历史回答"

            r2 = c.get(f"/api/v1/conversations/{sid}/messages",
                       params={"user_id": "intruder"})
            assert r2.status_code == 403             # 他人不可读

            r3 = c.delete(f"/api/v1/conversations/{sid}",
                          params={"user_id": "conv-u1"})
            assert r3.status_code == 200             # 行 + checkpoint 一并删
            r4 = c.get(f"/api/v1/conversations/{sid}/messages",
                       params={"user_id": "conv-u1"})
            assert r4.json()["messages"] == []       # 行已删 → 视为没聊过
    finally:
        _del_rows(sid)
        try:
            _run(graph.adelete_thread(sid))
        except Exception:
            pass

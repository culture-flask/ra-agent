"""普通和团队入口均不能接管其他用户的登记或遗留 checkpoint。"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.core.db import SessionLocal
from app.main import app
from app.models import BrainstormSession, Conversation


class _Model:
    model_name = "offline-checkpoint-owner"

    def __init__(self, calls):
        self.calls = calls

    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, config=None):
        self.calls.append(messages)
        yield AIMessageChunk(content="owner-only reply", response_metadata={"finish_reason": "stop"})

    async def ainvoke(self, messages, config=None):
        return AIMessage(content='{"memories":[]}')


@pytest.mark.parametrize("endpoint,body", [
    ("chat", {"message": "take over"}),
    ("chat/stream", {"message": "take over"}),
    ("chat", {"message": "take over", "rewind": True}),
    ("chat/stop", {}),
    ("chat/context", {}),
    ("brainstorm/stream", {"topic": "take over"}),
    ("seminar/stream", {"topic": "take over"}),
    ("deep-research/stream", {"topic": "take over"}),
])
def test_orphan_checkpoint_cannot_be_claimed_by_other_user(monkeypatch, auth_factory, endpoint, body):
    calls = []
    with TestClient(app) as client:
        service = app.state.llm_service
        monkeypatch.setattr(service, "get_chat_model", lambda *a, **kw: _Model(calls))
        monkeypatch.setattr(service, "effective_config", lambda *a, **kw: None)
        monkeypatch.setattr(service, "context_window_for", lambda *a: 128000)
        alice = auth_factory("checkpoint-alice")
        bob = auth_factory("checkpoint-bob")
        session_id = "orphan-private-state"
        assert client.post("/api/v1/chat", headers=alice, json={
            "session_id": session_id, "message": "private owner material",
        }).status_code == 200
        with SessionLocal() as db:
            row = db.get(Conversation, session_id)
            original_owner = row.user_id
            # 模拟会话删除时 SQL 已提交、checkpoint 清理失败的最终状态。
            db.delete(row)
            db.commit()
        before = len(calls)
        if endpoint == "chat/context":
            response = client.get("/api/v1/chat/context", headers=bob,
                                  params={"session_id": session_id})
        else:
            response = client.post("/api/v1/" + endpoint, headers=bob,
                                   json={"session_id": session_id, **body})
        assert response.status_code == 403
        assert len(calls) == before
        with SessionLocal() as db:
            assert db.get(Conversation, session_id) is None
            assert db.get(BrainstormSession, session_id) is None
        snapshot = client.portal.call(app.state.graph.aget_state,
                                      {"configurable": {"thread_id": session_id}})
        assert snapshot.values["user_id"] == original_owner
        assert "private owner material" in str(snapshot.values["messages"])


def test_chat_auxiliary_endpoints_respect_team_ownership(auth_factory):
    from app.core.cancel import clear_stop, is_stopped

    with TestClient(app) as client:
        alice, bob = auth_factory("team-alice"), auth_factory("team-bob")
        session_id = "team-private-state"
        with SessionLocal() as db:
            db.add(BrainstormSession(session_id=session_id, user_id="team-alice",
                team="seminar", topic="private team", roles=[], status="running"))
            db.commit()
        clear_stop(session_id)
        assert client.post("/api/v1/chat/stop", headers=bob,
            json={"session_id": session_id}).status_code == 403
        assert not is_stopped(session_id)
        assert client.get("/api/v1/chat/context", headers=bob,
            params={"session_id": session_id}).status_code == 403
        assert client.post("/api/v1/chat/stop", headers=alice,
            json={"session_id": session_id}).status_code == 200
        assert is_stopped(session_id)
        clear_stop(session_id)

"""会讲 API 测试：SSE 事件序列 / 鉴权 / team 过滤 / 详情。"""

import json

from fastapi.testclient import TestClient

from app.main import app
from test_seminar_graph import _make_ctx, _run, build_seminar_graph


def _install_fake_graph():
    app.state.seminar_graph = _run(build_seminar_graph(_make_ctx()))


def test_stream_requires_auth():
    with TestClient(app) as c:
        r = c.post("/api/v1/seminar/stream",
                   json={"session_id": "sem-x", "topic": "t"})
        assert r.status_code == 401


def test_stream_sse_event_sequence(auth_factory):
    with TestClient(app) as c:
        _install_fake_graph()
        with c.stream("POST", "/api/v1/seminar/stream",
                      headers=auth_factory(),
                      json={"session_id": "sem-api-1",
                            "topic": "RAG评测可靠性"}) as resp:
            assert resp.status_code == 200
            types = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    types.append(json.loads(line[6:]).get("type"))
    assert "sem_curate" in types          # 六阶段事件齐全
    assert "sem_agent_start" in types
    assert "sem_token" in types
    assert "sem_insight" in types
    assert "sem_idea_card" in types
    assert "sem_ranking" in types
    assert "sem_plan" in types
    assert types[-1] == "done"


def test_sessions_filtered_by_team(auth_factory):
    """team 过滤：争鸣社会话不出现在会讲列表（两队分家）。"""
    from app.core.db import SessionLocal
    from app.models import BrainstormSession, User

    with TestClient(app) as c:
        _install_fake_graph()
        h = auth_factory("u1")
        # 直接登记一条争鸣社（team 默认 debate）会话
        from app.core.security import hash_password
        with SessionLocal() as db:
            if not db.get(User, "u1"):
                db.add(User(id="u1", username="t-u1",
                            password_hash=hash_password("x")))
            db.add(BrainstormSession(session_id="bs-team-check",
                                     user_id="u1", topic="争鸣议题",
                                     status="done", team="debate"))
            db.commit()
        r = c.get("/api/v1/seminar/sessions", headers=h)
        assert r.status_code == 200
        ids = [s["session_id"] for s in r.json()["sessions"]]
        assert "bs-team-check" not in ids          # 争鸣社不混入会讲列表
        # 反向：争鸣社列表也不含会讲
        r2 = c.get("/api/v1/brainstorm/sessions", headers=h)
        ids2 = [s["session_id"] for s in r2.json()["sessions"]]
        assert "bs-team-check" in ids2


def test_seminar_detail_shape(auth_factory):
    """跑完一场：详情含会讲特有字段（笔记/洞见池/构想卡/评分/成稿）。"""
    with TestClient(app) as c:
        _install_fake_graph()
        h = auth_factory("u1")
        with c.stream("POST", "/api/v1/seminar/stream",
                      headers=h,
                      json={"session_id": "sem-api-2",
                            "topic": "RAG评测可靠性"}) as resp:
            assert resp.status_code == 200
            for _ in resp.iter_lines():
                pass
        r = c.get("/api/v1/seminar/sem-api-2", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert len(body["reading_notes"]) == 4
        assert body["insight_board"]
        assert body["card_registry"]
        assert body["final_proposal"]
        # 他人访问 → 403
        assert c.get("/api/v1/seminar/sem-api-2",
                     headers=auth_factory("u2")).status_code == 403


def test_ask_endpoint_routing_stream_and_ownership(auth_factory):
    """会讲追问：点名学者路由正确（最早命中优先）、未点名 → 主席；
    SSE 流 ask_start/token/done；属主校验 403、未知会话 404。"""
    from app.main import app

    class FakeGraph:
        async def aget_state(self, config):
            class _Snap:
                values = {
                    "user_id": "u1", "topic": "RAG评测",
                    "roles": [{"id": "historian"}],
                    "reading_notes": [{"agent_id": "historian",
                                       "content": "研读笔记", "failed": False}],
                    "qa_transcript": [], "insight_board": [], "open_questions": [],
                    "evidence_pool": [],
                    "card_registry": [{"card_id": "C1", "author": "visitor",
                                       "title": "迁移构想"}],
                    "idea_ranking": [], "final_proposal": "# 会讲报告",
                }
            return _Snap()

    from app.core.db import SessionLocal
    from app.models import BrainstormSession, User
    from app.core.security import hash_password

    def _setup():
        with SessionLocal() as db:
            for uid in ("u1", "u2"):
                if not db.get(User, uid):
                    db.add(User(id=uid, username=f"t-{uid}",
                                password_hash=hash_password("x")))
            if not db.get(BrainstormSession, "sem-ask-1"):
                db.add(BrainstormSession(session_id="sem-ask-1", user_id="u1",
                                         topic="议题", status="done",
                                         team="seminar"))
            db.commit()

    _setup()
    with TestClient(app) as c:
        saved = app.state.seminar_graph
        app.state.seminar_graph = FakeGraph()
        try:
            h1, h2 = auth_factory("u1"), auth_factory("u2")
            r = c.post("/api/v1/seminar/sem-none/ask", headers=h1,
                       json={"question": "文献学家在想什么"})
            assert r.status_code == 404
            r = c.post("/api/v1/seminar/sem-ask-1/ask", headers=h2,
                       json={"question": "q"})
            assert r.status_code == 403
            with c.stream("POST", "/api/v1/seminar/sem-ask-1/ask",
                          headers=h1,
                          json={"question": "文献学家在想什么"}) as resp:
                assert resp.status_code == 200
                evs = [json.loads(l[6:]) for l in resp.iter_lines()
                       if l.startswith("data: ")]
            types = [e["type"] for e in evs]
            assert types[0] == "ask_start" and evs[0]["agent"] == "historian"
            assert "token" in types and types[-1] == "done"
            # 未点名 → 主席
            with c.stream("POST", "/api/v1/seminar/sem-ask-1/ask",
                          headers=h1, json={"question": "总结一下"}) as resp:
                evs = [json.loads(l[6:]) for l in resp.iter_lines()
                       if l.startswith("data: ")]
            assert evs[0]["agent"] == "chair"
        finally:
            app.state.seminar_graph = saved

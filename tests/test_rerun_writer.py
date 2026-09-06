"""仅重跑执笔人端点测试（三团队）：checkpoint 恢复 → 只跑执笔节点 →
落库/回写/入库；属主与状态闸门。图与节点全部用假实现隔离。"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.models import BrainstormSession


def _make_session(user_id: str, sid: str, team: str, status: str = "failed"):
    from app.core.db import SessionLocal
    with SessionLocal() as db:
        db.add(BrainstormSession(session_id=sid, user_id=user_id,
                                 topic="测试议题", status=status, roles=[],
                                 team=team))
        db.commit()


def _row(sid: str) -> BrainstormSession:
    from app.core.db import SessionLocal
    with SessionLocal() as db:
        return db.get(BrainstormSession, sid)


def _fake_graph(values: dict):
    """aget_state 返回给定 checkpoint 状态；记录 aupdate_state 回写。"""
    class FakeGraph:
        def __init__(self):
            self.updated = None
        async def aget_state(self, config):
            return SimpleNamespace(values=values)
        async def aupdate_state(self, config, update, as_node=None):
            self.updated = (update, as_node)
    return FakeGraph()


def _post(client, headers, sid, team):
    prefix = {"debate": "/api/v1/brainstorm", "seminar": "/api/v1/seminar",
              "deep_research": "/api/v1/deep-research"}[team]
    return client.post(f"{prefix}/rerun-writer", headers=headers,
                       json={"session_id": sid})


def _events(text: str) -> list[str]:
    return [ln[6:] for ln in text.splitlines() if ln.startswith("data: ")]


def test_debate_rerun_writer_happy_path(auth_factory, monkeypatch):
    h1 = auth_factory("u1")
    _make_session("u1", "bs-rw-1", "debate", status="failed")
    state = {"positions": [{"agent_name": "A", "content": "x"}],
             "transcript": [{"agent_name": "A", "content": "y"}],
             "evidence_pool": [], "moderator_notes": {},
             "token_budget_used": 100}
    graph = _fake_graph(state)
    calls = {}

    async def fake_synthesis(ctx, st):
        calls["state"] = st
        return {"final_proposal": "NEW PROPOSAL", "token_budget_used": 5}

    archived = []
    monkeypatch.setattr("app.api.brainstorm.aget_state_retry",
                        lambda g, c: _async(SimpleNamespace(values=state)))
    monkeypatch.setattr("app.api.brainstorm.synthesis_node", fake_synthesis)
    monkeypatch.setattr("app.api.brainstorm._archive_proposal_to_kb",
                        lambda uid, sid, topic, p: archived.append((uid, sid, p)))
    with TestClient(app) as c:
        c.app.state.brainstorm_graph = graph
        r = _post(c, h1, "bs-rw-1", "debate")
    assert r.status_code == 200
    evs = [e for e in (_events(r.text))]
    assert any('"type": "done"' in e or "'type': 'done'" in e or "done" in e
               for e in evs), evs
    # 节点收到 checkpoint 恢复的状态，且停止标记被复位
    assert calls["state"]["stopped"] is False
    assert calls["state"]["positions"][0]["agent_name"] == "A"
    # 落库：成稿更新 + 状态 done
    row = _row("bs-rw-1")
    assert row.status == "done" and row.final_proposal == "NEW PROPOSAL"
    assert row.stats["positions"] == 1
    # checkpoint 回写（详情回放以 checkpoint 为准）
    assert graph.updated[1] == "synthesis"
    assert graph.updated[0]["final_proposal"] == "NEW PROPOSAL"
    # 成稿入库（知识飞轮）
    assert archived == [("u1", "bs-rw-1", "NEW PROPOSAL")]


def test_rerun_writer_guards(auth_factory, monkeypatch):
    h1, h2 = auth_factory("u1"), auth_factory("u2")
    _make_session("u1", "bs-rw-2", "debate")
    monkeypatch.setattr("app.api.brainstorm.aget_state_retry",
                        lambda g, c: _async(SimpleNamespace(values={})))
    with TestClient(app) as c:
        # 他人会话 → 403
        assert _post(c, h2, "bs-rw-2", "debate").status_code == 403
        # 不存在 → 404
        assert _post(c, h1, "bs-none", "debate").status_code == 404
        # 无中间状态 → 404（提示整场重跑）
        r = _post(c, h1, "bs-rw-2", "debate")
        assert r.status_code == 404 and "整场重跑" in r.json()["detail"]
        # running 中 → 409
        _make_session("u1", "bs-rw-2b", "debate", status="running")
        monkeypatch.setattr("app.api.brainstorm.aget_state_retry",
                            lambda g, c: _async(SimpleNamespace(
                                values={"positions": [1]})))
        assert _post(c, h1, "bs-rw-2b", "debate").status_code == 409


def test_seminar_rerun_writer(auth_factory, monkeypatch):
    h1 = auth_factory("u1")
    _make_session("u1", "sem-rw-1", "seminar")
    state = {"reading_notes": [{"agent_id": "historian", "content": "n"}],
             "card_registry": [], "qa_transcript": [], "insight_board": []}
    graph = _fake_graph(state)
    calls = {}

    def fake_rapporteur(ctx):
        async def _node(st):
            calls["state"] = st
            return {"final_proposal": "SEM NEW", "token_budget_used": 3}
        return _node

    archived = []
    monkeypatch.setattr("app.api.seminar.aget_state_retry",
                        lambda g, c: _async(SimpleNamespace(values=state)))
    monkeypatch.setattr("app.api.seminar.make_rapporteur_node", fake_rapporteur)
    monkeypatch.setattr("app.api.seminar._archive_report_to_kb",
                        lambda uid, sid, topic, p: archived.append((uid, sid, p)))
    with TestClient(app) as c:
        c.app.state.seminar_graph = graph
        r = _post(c, h1, "sem-rw-1", "seminar")
    assert r.status_code == 200
    assert calls["state"]["stopped"] is False
    row = _row("sem-rw-1")
    assert row.status == "done" and row.final_proposal == "SEM NEW"
    assert graph.updated[1] == "rapporteur"
    assert archived == [("u1", "sem-rw-1", "SEM NEW")]


def test_deep_research_rerun_writer(auth_factory, monkeypatch):
    h1 = auth_factory("u1")
    _make_session("u1", "dr-rw-1", "deep_research")
    state = {"chapters": [{"index": 1, "title": "背景", "content": "c1"}],
             "outline": {"title": "报告"}, "source_pool": [{"title": "s",
                                                           "url": "https://a"}]}
    graph = _fake_graph(state)
    calls = {}

    def fake_frame(ctx):
        async def _node(st):
            calls["frame_state"] = st
            return {"toc": "目录", "introduction": "引言", "conclusion": "结论"}
        return _node

    async def fake_publish(st):
        calls["publish_state"] = st
        return {"final_report": "REPORT", "references": []}

    archived = []
    monkeypatch.setattr("app.api.deep_research.aget_state_retry",
                        lambda g, c: _async(SimpleNamespace(values=state)))
    monkeypatch.setattr("app.api.deep_research.make_write_frame_node", fake_frame)
    monkeypatch.setattr("app.api.deep_research.publish_node", fake_publish)
    monkeypatch.setattr("app.api.deep_research._archive_report_to_kb",
                        lambda uid, sid, topic, p: archived.append((uid, sid, p)))
    with TestClient(app) as c:
        c.app.state.deep_research_graph = graph
        r = _post(c, h1, "dr-rw-1", "deep_research")
    assert r.status_code == 200
    # publish 收到 write_frame 的产出（章节 + 引言/结论装配）
    assert calls["publish_state"]["toc"] == "目录"
    assert calls["publish_state"]["chapters"][0]["title"] == "背景"
    row = _row("dr-rw-1")
    assert row.status == "done" and row.final_proposal == "REPORT"
    assert graph.updated[1] == "publish"
    assert archived == [("u1", "dr-rw-1", "REPORT")]


async def _async(v):
    return v

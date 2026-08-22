"""用户反馈 API：写入持久化、越权隔离、参数校验、payload 截断（P3-19）。"""

from fastapi.testclient import TestClient

from app.core.db import SessionLocal
from app.main import app
from app.models import Feedback


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _register(c, username):
    reg = c.post("/api/v1/auth/register",
                 json={"username": username, "password": "pass1234"}).json()
    return reg["access_token"]


def test_post_requires_auth():
    with TestClient(app) as c:
        assert c.post("/api/v1/feedbacks", json={
            "session_id": "fb-s", "rating": 1}).status_code == 401
        bad = {"Authorization": "Bearer forged-token"}
        assert c.post("/api/v1/feedbacks", headers=bad, json={
            "session_id": "fb-s", "rating": 1}).status_code == 401


def test_post_persists_with_token_identity(auth_factory):
    """落库的 user_id 来自 token 而非任何客户端字段；hits 原样回读。"""
    with TestClient(app) as c:
        tok = _register(c, "fb-persist")
        r = c.post("/api/v1/feedbacks", headers=_auth(tok), json={
            "session_id": "fb-s1", "rating": 1,
            "question": "HDND 高阶差分神经区分器是什么",
            "answer": "它是一种……",
            "hits": [{"kb_name": "密码学库", "scope": "public",
                      "source": "HDND.pdf", "page": 3, "score": 0.0123}],
        })
        assert r.status_code == 201
        fid = r.json()["feedback_id"]

        rows = c.get("/api/v1/feedbacks", headers=_auth(tok)).json()
        assert len(rows) == 1 and rows[0]["id"] == fid

        detail = c.get(f"/api/v1/feedbacks/{fid}", headers=_auth(tok)).json()
        assert detail["question"].startswith("HDND")
        assert detail["hits"][0]["source"] == "HDND.pdf"

        with SessionLocal() as db:
            row = db.get(Feedback, fid)
            assert row is not None and row.rating == 1
            assert row.hits[0]["source"] == "HDND.pdf"


def test_list_and_detail_user_isolated(auth_factory):
    """换 token 即换视角：B 看不到 A 的反馈，B 取 A 的详情按 404 处理。"""
    with TestClient(app) as c:
        tok_a = _register(c, "fb-iso-a")
        r = c.post("/api/v1/feedbacks", headers=_auth(tok_a), json={
            "session_id": "fb-s2", "rating": -1,
            "question": "q", "answer": "a"})
        fid = r.json()["feedback_id"]
        tok_b = _register(c, "fb-iso-b")

        assert c.get("/api/v1/feedbacks", headers=_auth(tok_b)).json() == []
        assert c.get(f"/api/v1/feedbacks/{fid}",
                     headers=_auth(tok_b)).status_code == 404
        mine = c.get("/api/v1/feedbacks", headers=_auth(tok_a)).json()
        assert len(mine) == 1 and mine[0]["rating"] == -1


def test_rating_bounds_rejected(auth_factory):
    """rating 只接受 -1/1；0 无信息量、越界值一律 422。"""
    with TestClient(app) as c:
        h = _auth(_register(c, "fb-bounds"))
        for bad in (0, 2, -2):
            r = c.post("/api/v1/feedbacks", headers=h,
                       json={"session_id": "s", "rating": bad})
            assert r.status_code == 422
        assert c.post("/api/v1/feedbacks", headers=h,
                      json={"session_id": "s", "rating": -1}).status_code == 201


def test_payload_truncated_and_sanitized(auth_factory):
    """超长 question/answer 截断、hits 条数截断、非标量字段剔除。"""
    with TestClient(app) as c:
        h = _auth(_register(c, "fb-trunc"))
        big_hits = [{"source": f"f{i}.pdf", "junk": {"nested": True},
                     "score": 0.1} for i in range(50)]
        r = c.post("/api/v1/feedbacks", headers=h, json={
            "session_id": "s", "rating": 1,
            "question": "问" * 5000, "answer": "答" * 20000,
            "hits": big_hits})
        assert r.status_code == 201
        fid = r.json()["feedback_id"]
        detail = c.get(f"/api/v1/feedbacks/{fid}", headers=h).json()
        assert len(detail["answer"]) <= 8000
        assert len(detail["question"]) <= 4000
        assert len(detail["hits"]) == 20                    # 条数截断
        assert all("junk" not in hit for hit in detail["hits"])  # 非标量剔除

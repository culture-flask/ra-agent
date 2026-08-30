"""头脑风暴 API 测试：把 app.state.brainstorm_graph 换成假图（假 LLM ctx
编译），测 SSE 事件序列 / 属主校验 / 列表详情。"""

import json

from fastapi.testclient import TestClient

from app.main import app
from test_brainstorm_graph import _make_ctx, _run, build_brainstorm_graph


def _install_fake_graph():
    """替换 lifespan 装的真实子图为假图（不触发真实 LLM 调用）。"""
    ctx = _make_ctx()
    app.state.brainstorm_graph = _run(build_brainstorm_graph(ctx))


def test_stream_requires_auth():
    with TestClient(app) as c:
        r = c.post("/api/v1/brainstorm/stream",
                   json={"session_id": "bs-x", "topic": "t"})
        assert r.status_code == 401            # 无 token → 401


def test_stream_sse_event_sequence(auth_factory):
    with TestClient(app) as c:
        _install_fake_graph()
        with c.stream("POST", "/api/v1/brainstorm/stream",
                      headers=auth_factory(),
                      json={"session_id": "bs-api-1",
                            "topic": "如何提升RAG评测可靠性",
                            "max_rounds": 1}) as resp:
            assert resp.status_code == 200
            types = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    types.append(json.loads(line[6:]).get("type"))
    assert "bs_prepare" in types               # 五阶段事件都在流里
    assert "bs_agent_start" in types
    assert "bs_token" in types
    assert "bs_plan" in types
    assert types[-1] == "done"


def test_stop_and_ownership(auth_factory):
    with TestClient(app) as c:
        _install_fake_graph()
        h1, h2 = auth_factory("u1"), auth_factory("u2")
        # u2 不能停 u1 的会话
        r = c.post("/api/v1/brainstorm/stop", headers=h2,
                   json={"session_id": "bs-none"})
        assert r.status_code == 200            # 未登记的会话放行（同 /chat/stop）
        # 列表为空起步
        r = c.get("/api/v1/brainstorm/sessions", headers=h1)
        assert r.status_code == 200 and r.json()["sessions"] == []


def test_session_list_and_detail_after_run(auth_factory):
    """跑完一场：sessions 列表登记 done、详情可回放立场书/辩论/成稿。"""
    with TestClient(app) as c:
        _install_fake_graph()
        h = auth_factory("u1")
        with c.stream("POST", "/api/v1/brainstorm/stream",
                      headers=h,
                      json={"session_id": "bs-api-2",
                            "topic": "议题", "max_rounds": 1}) as resp:
            assert resp.status_code == 200
            for _ in resp.iter_lines():        # 消费完整 SSE 流
                pass
        r = c.get("/api/v1/brainstorm/sessions", headers=h)
        assert r.status_code == 200
        items = [s for s in r.json()["sessions"] if s["session_id"] == "bs-api-2"]
        assert items and items[0]["status"] == "done"
        assert items[0]["stats"]["tokens"] > 0 or items[0]["stats"]["positions"] > 0

        r = c.get("/api/v1/brainstorm/bs-api-2", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert len(body["positions"]) == 4            # 四份立场书
        assert body["final_proposal"]                 # 成稿可回放

        # 他人访问 → 403
        r = c.get("/api/v1/brainstorm/bs-api-2", headers=auth_factory("u2"))
        assert r.status_code == 403


def test_finish_session_records_budget_exhaustion(auth_factory):
    """预算熔断：status=done 但 stats 带截断标注（设计文档 §7.7）。"""
    from app.api.brainstorm import _finish_session
    from app.core.db import SessionLocal
    from app.models import BrainstormSession, User

    # 先登记一行（_finish_session 只更新 /stream 时已登记的会话）
    with SessionLocal() as db:
        if not db.get(User, "u1"):
            from app.core.security import hash_password
            db.add(User(id="u1", username="t-u1", password_hash=hash_password("x")))
        db.add(BrainstormSession(session_id="bs-stat-1", user_id="u1",
                                 topic="议题", status="running"))
        db.commit()
    _finish_session("u1", "bs-stat-1", {
        "transcript": [], "positions": [], "evidence_pool": [],
        "token_budget_used": 900_000, "final_proposal": "x",
        "budget_exhausted": True, "stopped": False})
    with SessionLocal() as db:
        row = db.get(BrainstormSession, "bs-stat-1")
        assert row is not None and row.status == "done"
        assert row.stats.get("budget_exhausted") is True


def test_stream_roles_subset_and_validation(auth_factory):
    """请求携带 roles 子集 → 只有子集角色参赛（§7.5 roles? 参数）；
    非法 id / 空列表 → 回退默认角色目录。"""
    from app.api.brainstorm import _resolve_roles
    from app.settings import Settings

    settings = Settings.load()
    # 非法 id 与重复 id 被过滤，温度被夹取，名称被截断
    roles = _resolve_roles([
        {"id": "critic", "temperature": 9, "name": "x" * 40,
         "config_id": "abc123"},
        {"id": "hacker"}, {"id": "critic"},
        {"id": "innovator", "temperature": "bad"},
    ], settings)
    assert [r["id"] for r in roles] == ["critic", "innovator"]
    critic = next(r for r in roles if r["id"] == "critic")
    assert critic.get("config_id") == "abc123"        # 凭证引用透传
    assert critic.get("model_id") is None             # 未传 model_id 不造键
    roles2 = _resolve_roles([
        {"id": "critic", "model_id": "  deepseek-v4-flash  "}], settings)
    assert roles2[0].get("model_id") == "deepseek-v4-flash"   # 同供应商换模型
    critic = next(r for r in roles if r["id"] == "critic")
    assert critic["temperature"] == 2.0 and len(critic["name"]) <= 20
    innovator = next(r for r in roles if r["id"] == "innovator")
    assert innovator["temperature"] == 0.9            # 解析失败回退 yaml 默认

    # None / 空 → 完整默认目录
    assert len(_resolve_roles(None, settings)) == 4
    assert len(_resolve_roles([], settings)) == 4

    # 端到端：只带 2 个角色 → transcript 只含子集角色
    with TestClient(app) as c:
        _install_fake_graph()
        h = auth_factory("u1")
        with c.stream("POST", "/api/v1/brainstorm/stream",
                      headers=h,
                      json={"session_id": "bs-api-roles",
                            "topic": "议题", "max_rounds": 1,
                            "roles": [{"id": "innovator"}, {"id": "critic"}]}) as resp:
            assert resp.status_code == 200
            types = [json.loads(l[6:]).get("type")
                     for l in resp.iter_lines() if l.startswith("data: ")]
        assert types[-1] == "done"
        r = c.get("/api/v1/brainstorm/bs-api-roles", headers=h)
        body = r.json()
        assert len(body["positions"]) == 2            # 只有两份立场书
        assert {p["agent_id"] for p in body["positions"]} == {"innovator", "critic"}
        assert all(t["agent_id"] in ("innovator", "critic")
                   for t in body["transcript"])


def test_stream_disconnect_marks_session_failed(auth_factory):
    """客户端中途断开（SSE 连接中断）→ 后台任务被取消，残留 running 行
    复位为 failed（CancelledError 分支，与启动自愈同语义）。"""
    import time

    class SlowGraph:
        """ainvoke 长跑不返回：模拟真实图在断开时尚未结束。"""

        async def adelete_thread(self, sid):
            pass

        async def ainvoke(self, initial, config):
            await asyncio.sleep(30)
            return {}

        async def aget_state(self, config):
            class _Snap:
                values = {}
            return _Snap()

    with TestClient(app) as c:
        h = auth_factory("u1")
        saved = app.state.brainstorm_graph
        app.state.brainstorm_graph = SlowGraph()
        try:
            with c.stream("POST", "/api/v1/brainstorm/stream", headers=h,
                          json={"session_id": "bs-disc-1", "topic": "t"}) as resp:
                assert resp.status_code == 200
            # 流已断开：轮询等待取消分支把 running 复位为 failed
            from app.core.db import SessionLocal
            from app.models import BrainstormSession
            deadline, status = time.time() + 8, None
            while time.time() < deadline:
                with SessionLocal() as db:
                    row = db.get(BrainstormSession, "bs-disc-1")
                    status = row.status if row else None
                    if status == "failed":
                        break
                time.sleep(0.2)
            assert status == "failed", f"会话状态未复位：{status}"
        finally:
            app.state.brainstorm_graph = saved


def test_ask_endpoint_routing_stream_and_ownership(auth_factory):
    """追问端点：点名角色路由正确（最早命中优先）、未点名 → 主持人；
    SSE 流 ask_start/token/done；属主校验 403、未知会话 404。"""
    import asyncio

    from app.main import app

    class FakeGraph:
        async def aget_state(self, config):
            class _Snap:
                values = {
                    "topic": "议题", "roles": [{"id": "innovator"}],
                    "positions": [{"agent_id": "innovator",
                                   "content": "我的主张", "failed": False}],
                    "transcript": [], "evidence_pool": [],
                    "moderator_notes": {"consensus": ["c"], "divergence": []},
                    "final_proposal": "# 研究方案",
                }
            return _Snap()

    class _Req:
        session_id = "bs-ask-1"
    req = _Req()

    def _setup():
        from app.core.db import SessionLocal
        from app.models import BrainstormSession, User
        from app.core.security import hash_password
        with SessionLocal() as db:
            for uid in ("u1", "u2"):
                if not db.get(User, uid):
                    db.add(User(id=uid, username=f"t-{uid}",
                                password_hash=hash_password("x")))
            if not db.get(BrainstormSession, "bs-ask-1"):
                db.add(BrainstormSession(session_id="bs-ask-1", user_id="u1",
                                         topic="议题", status="done"))
            db.commit()

    _setup()
    with TestClient(app) as c:
        saved = app.state.brainstorm_graph
        app.state.brainstorm_graph = FakeGraph()
        try:
            h1, h2 = auth_factory("u1"), auth_factory("u2")
            # 未知会话 404
            r = c.post("/api/v1/brainstorm/bs-none/ask", headers=h1,
                       json={"question": "创新者在想什么"})
            assert r.status_code == 404
            # 他人会话 403
            r = c.post("/api/v1/brainstorm/bs-ask-1/ask", headers=h2,
                       json={"question": "q"})
            assert r.status_code == 403
            # 点名创新者 → ask_start.agent == innovator
            with c.stream("POST", "/api/v1/brainstorm/bs-ask-1/ask",
                          headers=h1, json={"question": "创新者在想什么"}) as resp:
                assert resp.status_code == 200
                evs = [json.loads(l[6:]) for l in resp.iter_lines()
                       if l.startswith("data: ")]
            types = [e["type"] for e in evs]
            assert types[0] == "ask_start" and evs[0]["agent"] == "innovator"
            assert "token" in types and types[-1] == "done"
            # 未点名 → 主持人
            with c.stream("POST", "/api/v1/brainstorm/bs-ask-1/ask",
                          headers=h1, json={"question": "总结一下各方观点"}) as resp:
                evs = [json.loads(l[6:]) for l in resp.iter_lines()
                       if l.startswith("data: ")]
            assert evs[0]["agent"] == "moderator"
        finally:
            app.state.brainstorm_graph = saved

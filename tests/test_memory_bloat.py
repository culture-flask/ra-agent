"""记忆膨胀控制：分层归一化 / 主题压缩 / LRU 淘汰 / TTL / 选择性删除 API。"""

import asyncio
import json
from datetime import timedelta

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select, update

from app.core.db import SessionLocal
from app.graph.nodes import WorkflowContext, _normalize_memory, _maintain_memories
from app.main import app
from app.models import Memory, User
from app.services.memory_service import MemoryService

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


def _ensure_user(user_id: str):
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()


def _cleanup(user_id: str, keys: list[str] | None = None):
    with SessionLocal() as db:
        stmt = select(Memory).where(Memory.user_id == user_id)
        if keys is not None:
            stmt = stmt.where(Memory.key.in_(keys))
        for r in db.scalars(stmt):
            db.delete(r)
        db.commit()


# ---------- 归一化（LLM 标注兜底） ----------
def test_normalize_memory():
    # 时效词 + core → 强制降级 short
    m = _normalize_memory({"key": "current_paper", "value": "本周在写论文X",
                           "tier": "core", "topic": "项目"})
    assert m["tier"] == "short"
    # tier 白名单：非法值回退 core；topic 兜底取 key 首段
    m2 = _normalize_memory({"key": "research_topic", "value": "量子计算研究"})
    assert m2["tier"] == "core" and m2["topic"] == "research"
    # 合法标注原样保留
    m3 = _normalize_memory({"key": "pref_style", "value": "喜欢简洁回答",
                            "tier": "short", "topic": "偏好"})
    assert m3["tier"] == "short" and m3["topic"] == "偏好"


# ---------- 分层读写 / touch / TTL ----------
def test_tiered_get_all_touch_expire():
    uid = "mem-b1"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        ms.set(uid, "research_topic", {"v": "量子计算"}, tier="core", topic="研究方向")
        ms.set(uid, "current_task", {"v": "本周写论文"}, tier="short", topic="项目")
        # tier 过滤
        core = ms.get_all(uid, "core")
        assert list(core) == ["research_topic"]
        # touch 只刷指定 key
        ms.touch(uid, ["research_topic"])
        # short 过期：把 updated_at 拨回 15 天前
        with SessionLocal() as db:
            db.execute(update(Memory).where(
                Memory.user_id == uid, Memory.key == "current_task")
                .values(updated_at=Memory.updated_at - timedelta(days=15)))
            db.commit()
        assert ms.expire_short(uid) == 1
        assert ms.get_all(uid).get("current_task") is None   # 已清除
        assert ms.get_all(uid).get("research_topic")         # core 不受 TTL 影响
    finally:
        _cleanup(uid)


# ---------- 主题压缩 + LRU 管线（service 原语 + 节点编排） ----------
def test_topic_groups_and_apply_merge():
    uid = "mem-b2"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        ms.set(uid, "research_a", {"v": "研究量子纠错"}, tier="core", topic="研究方向")
        ms.set(uid, "research_b", {"v": "研究量子退火"}, tier="core", topic="研究方向")
        ms.set(uid, "pref_style", {"v": "喜欢简洁回答"}, tier="core", topic="偏好")
        groups = ms.topic_groups(uid)
        assert set(groups) == {"研究方向"}                    # 偏好组只有 1 条不成组
        freed = ms.apply_merge(uid, [{
            "topic": "研究方向", "key": "research_a", "value": "研究量子纠错与量子退火",
            "group_keys": [g["key"] for g in groups["研究方向"]]}])
        assert freed == 1                                     # 2 条 → 1 条
        memory = ms.get_all(uid)
        assert memory["research_a"]["v"] == "研究量子纠错与量子退火"
        assert "research_b" in memory["research_a"]["merged_from"]   # 合并留痕
        assert "research_b" not in memory
    finally:
        _cleanup(uid)


def test_evict_overflow_lru():
    uid = "mem-b3"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        # 51 条：50 core + 2 short → 淘汰 1 条最旧 short，core 与新 short 保留
        for i in range(49):
            ms.set(uid, f"core_{i:02d}", {"v": f"核心记忆{i}"}, tier="core")
        ms.set(uid, "short_old", {"v": "旧的短期记忆"}, tier="short")
        ms.set(uid, "short_new", {"v": "新的短期记忆"}, tier="short")
        with SessionLocal() as db:                            # short_old 拨回 10 天前
            db.execute(update(Memory).where(
                Memory.user_id == uid, Memory.key == "short_old")
                .values(last_used_at=Memory.last_used_at - timedelta(days=10)))
            db.commit()
        assert ms.count(uid) == 51
        evicted = ms.evict_overflow(uid)                      # 上限 50
        assert evicted == 1 and ms.count(uid) == 50
        assert ms.get_all(uid).get("short_old") is None       # 最旧 short 先淘汰
        assert ms.get_all(uid).get("short_new")               # core 未动，新 short 保留
    finally:
        _cleanup(uid)


def test_maintain_pipeline_with_fake_llm():
    """超限触发完整管线：假 LLM 合并同主题 → LRU 兜底，emit 不炸。"""
    uid = "mem-b4"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    for i in range(49):                                       # 49 条零散 core
        ms.set(uid, f"misc_{i:02d}", {"v": f"杂项{i}"}, tier="core", topic="杂项")
    ms.set(uid, "research_a", {"v": "研究量子纠错"}, tier="core", topic="研究方向")
    ms.set(uid, "research_b", {"v": "研究量子退火"}, tier="core", topic="研究方向")
    # 再写 1 条 → 52 条超限
    ms.set(uid, "current_task", {"v": "本周写论文"}, tier="short", topic="项目")

    class _MergeModel:
        async def ainvoke(self, messages):
            return AIMessage(content=json.dumps(
                {"merged": [{"topic": "研究方向", "key": "research_a",
                             "value": "研究量子纠错与量子退火"}]}, ensure_ascii=False))

    class _MergeLLMService:
        def get_chat_model(self, user_id, temperature=None):
            return _MergeModel()

    ctx = WorkflowContext(None, _MergeLLMService(), None, None, None, ms)
    try:
        stat = _run(_maintain_memories(ctx, uid))
        assert stat["compressed"] == 1                        # 研究方向 2→1
        assert ms.count(uid) <= 50                            # LRU 兜底到上限内
        assert ms.get_all(uid).get("research_a", {}).get("v") == "研究量子纠错与量子退火"
    finally:
        _cleanup(uid)


# ---------- API：调级 / 单删 / 批删 ----------
def test_memory_tier_and_delete_api(auth_factory):
    uid = "mem-b5"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    ms.set(uid, "research_topic", {"v": "量子计算"}, tier="core", topic="研究方向")
    ms.set(uid, "temp_task", {"v": "临时任务"}, tier="short", topic="项目")
    try:
        h = auth_factory(uid)          # P0-1：身份来自 token，视角即 uid 本人
        with TestClient(app) as c:
            # 列表带层级
            rows = c.get("/api/v1/memories", headers=h).json()
            assert {r["key"]: r["tier"] for r in rows} == {
                "research_topic": "core", "temp_task": "short"}
            # 非法 tier
            assert c.patch(f"/api/v1/memories/temp_task/tier",
                           headers=h,
                           json={"tier": "bad"}).status_code == 400
            # 调级 + 404
            assert c.patch(f"/api/v1/memories/temp_task/tier",
                           headers=h,
                           json={"tier": "core"}).status_code == 200
            assert c.patch(f"/api/v1/memories/ghost/tier",
                           headers=h,
                           json={"tier": "core"}).status_code == 404
            # 单删 + 批删
            assert c.delete(f"/api/v1/memories/temp_task",
                            headers=h).json()["deleted"] == ["temp_task"]
            assert c.delete(f"/api/v1/memories/temp_task",
                            headers=h).status_code == 404
            n = c.post("/api/v1/memories/delete", headers=h,
                       json={"keys": ["research_topic", "ghost"]}).json()["deleted_count"]
            assert n == 1
            assert ms.get_all(uid) == {}
    finally:
        _cleanup(uid)

"""记忆检索工具（search_memories）：short 7 天窗口 / 总量上限 / 分层标注 / 用户隔离。

背景：short 层记忆此前只写不用（不注入 prompt），现由 LLM 主动调用本工具取回。
离线纪律：真实 KBService/MemoryService + 空 MCP 目录（不拉外部子进程）。
"""

import asyncio
import json
import tempfile
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select, update

from app.core.db import SessionLocal
from app.core.tracing import Tracer
from app.mcp.adapter import MCPToolAdapter
from app.mcp.host import MCPHost
from app.models import Memory, User
from app.services.kb_service import KBService
from app.services.memory_service import MemoryService
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


def _make_adapter(memory_service):
    """真实适配器 + 空外部 MCP 目录（原生工具仍在）。"""
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    host = MCPHost({}, base_dir=Path(__file__).resolve().parent.parent)
    return MCPToolAdapter(host, Tracer(), kb_service=KBService(settings),
                          memory_service=memory_service)


def _ensure_user(user_id: str):
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()


def _cleanup(user_id: str):
    with SessionLocal() as db:
        for r in db.scalars(select(Memory).where(Memory.user_id == user_id)):
            db.delete(r)
        db.commit()


def _backdate(user_id: str, key: str, days: int):
    """把 updated_at 拨回 days 天前（模拟旧记忆，同 test_memory_bloat 手法）。"""
    with SessionLocal() as db:
        db.execute(update(Memory).where(
            Memory.user_id == user_id, Memory.key == key)
            .values(updated_at=Memory.updated_at - timedelta(days=days)))
        db.commit()


def _search(adapter, user_id: str, args: dict) -> dict:
    out = _run(adapter.call("search_memories", args, "t-mem-tool", user_id))
    return json.loads(out["output"])


# ---------- 工具目录 ----------
def test_search_memories_in_catalog():
    """工具出现在给 LLM 的目录里（否则模型无从调用）。"""
    adapter = _make_adapter(MemoryService())
    names = {s.get("name") or (s.get("function") or {}).get("name")
             for s in _run(adapter.schemas_for_llm())}
    assert "search_memories" in names


# ---------- short 的 7 天窗口 ----------
def test_short_window_seven_days():
    uid = "mem-tool-1"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        ms.set(uid, "task_old", {"v": "十天前在做旧项目"}, tier="short", topic="项目")
        ms.set(uid, "task_new", {"v": "最近在做新项目"}, tier="short", topic="项目")
        _backdate(uid, "task_old", 8)                  # 8 天前 → 出窗
        data = _search(_make_adapter(ms), uid, {"tier": "short"})
        assert [m["key"] for m in data["memories"]] == ["task_new"]
        assert "short" in data["note"]                 # 明确标注"临时信息"
    finally:
        _cleanup(uid)


def test_core_not_window_limited():
    """core 是稳定偏好：不受 7 天窗口限制（否则会无谓丢信息）。"""
    uid = "mem-tool-2"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        ms.set(uid, "core_old", {"v": "长期有效的稳定偏好"}, tier="core", topic="偏好")
        _backdate(uid, "core_old", 60)
        data = _search(_make_adapter(ms), uid, {"query": "稳定偏好"})
        assert [m["key"] for m in data["memories"]] == ["core_old"]
        assert data["memories"][0]["tier"] == "core"
    finally:
        _cleanup(uid)


# ---------- 总量上限 ----------
def test_limit_ten():
    uid = "mem-tool-3"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        for i in range(12):                            # 12 条命中 → 只回 10
            ms.set(uid, f"bulk_{i:02d}", {"v": f"批量记忆{i}"},
                   tier="core", topic="批量")
        data = _search(_make_adapter(ms), uid, {})     # 空 query = 最近更新的一批
        assert data["count"] == 10
        assert len(data["memories"]) == 10
    finally:
        _cleanup(uid)


# ---------- query 三路匹配 ----------
def test_query_matches_key_topic_value():
    uid = "mem-tool-4"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        ms.set(uid, "zzz_unique_key", {"v": "内容甲"}, tier="core", topic="主题甲")
        ms.set(uid, "topic_probe", {"v": "内容乙"}, tier="core", topic="独特研究方向")
        ms.set(uid, "value_probe", {"v": "关注贝尔不等式实验"}, tier="core", topic="主题丙")
        adapter = _make_adapter(ms)
        assert [m["key"] for m in _search(adapter, uid, {"query": "unique_key"})["memories"]] \
            == ["zzz_unique_key"]
        assert [m["key"] for m in _search(adapter, uid, {"query": "独特研究方向"})["memories"]] \
            == ["topic_probe"]
        assert [m["key"] for m in _search(adapter, uid, {"query": "贝尔不等式"})["memories"]] \
            == ["value_probe"]
    finally:
        _cleanup(uid)


# ---------- 用户隔离 / 超长截断 / 降级 ----------
def test_user_isolation():
    uid, other = "mem-tool-5", "mem-tool-5-b"
    _ensure_user(uid)
    _ensure_user(other)
    ms = MemoryService()
    _cleanup(uid)
    _cleanup(other)
    try:
        ms.set(uid, "private_note", {"v": "只属于 u1 的记忆"}, tier="core", topic="私密")
        data = _search(_make_adapter(ms), other, {"query": "私密"})
        assert data["count"] == 0                      # 查不到别人的记忆
    finally:
        _cleanup(uid)
        _cleanup(other)


def test_value_truncated():
    uid = "mem-tool-6"
    _ensure_user(uid)
    ms = MemoryService()
    _cleanup(uid)
    try:
        ms.set(uid, "long_note", {"v": "很长" * 300}, tier="core", topic="长文")
        data = _search(_make_adapter(ms), uid, {"query": "很长"})
        value = data["memories"][0]["value"]
        assert value.endswith("…(已截断)")
        assert len(value) == 500 + len("…(已截断)")
    finally:
        _cleanup(uid)


def test_missing_service_degrades():
    """无记忆服务（测试/降级部署）→ 结构化空结果，不抛错。"""
    data = _search(_make_adapter(None), "mem-tool-7", {"query": "任意"})
    assert data["count"] == 0 and data["memories"] == []
    assert "不可用" in data["note"]
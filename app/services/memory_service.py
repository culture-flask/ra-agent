"""长期记忆服务：分层（core 常驻 / short 降级）+ 容量管线（主题压缩 → LRU → TTL）。

memories 表（user_id/key/value/tier/topic/last_used_at/updated_at）。
- core：高价值记忆（研究方向、稳定偏好），每轮注入 prompt，常驻
- short：临时性事实（"本周在写某论文"），不注入 prompt，TTL 过期自动清除
- 膨胀控制：超 memory_max 触发「主题压缩（LLM 合并同类 topic）→ LRU 淘汰」
  （编排见 nodes._maintain_memories，本模块只提供同步原语，不掺 LLM 调用）
"""

import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.core.db import SessionLocal
from app.core.logging import get_logger
from app.models import Memory
from app.settings import settings

logger = get_logger("memory")

MEMORY_MAX = settings.memory_max                 # 每用户条数上限
SHORT_TTL_DAYS = settings.memory_short_ttl_days  # short 层过期天数


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryService:
    """记忆读写：全部按 user_id 命名空间隔离。

    同用户的读改写（合并/淘汰）用 threading.Lock 串行化——记忆写入走
    asyncio.to_thread，多线程并发下防止 maintain 读改写竞态。
    """

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, user_id: str) -> threading.Lock:
        with self._guard:
            if user_id not in self._locks:
                self._locks[user_id] = threading.Lock()
            return self._locks[user_id]

    # ---------- 读 ----------
    def get_all(self, user_id: str, tier: str | None = None) -> dict[str, dict]:
        """读取记忆：{key: value}。tier 过滤（load_memory_node 只取 core 注入）。"""
        with SessionLocal() as db:
            stmt = select(Memory).where(Memory.user_id == user_id)
            if tier:
                stmt = stmt.where(Memory.tier == tier)
            # 稳定序：touch() 每轮 UPDATE 后 PG 返回顺序可能漂移，
            # 定序保证注入 prompt 的记忆序列字节级稳定（前缀缓存友好）
            stmt = stmt.order_by(Memory.key)
            return {m.key: m.value for m in db.scalars(stmt)}

    def list_memory(self, user_id: str) -> list[dict]:
        """供 API 展示：含层级/主题/最近使用时间。"""
        with SessionLocal() as db:
            stmt = (select(Memory).where(Memory.user_id == user_id)
                    .order_by(Memory.updated_at.desc()))
            return [
                {"key": m.key, "value": m.value, "tier": m.tier, "topic": m.topic,
                 "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                 "last_used_at": m.last_used_at.isoformat() if m.last_used_at else None}
                for m in db.scalars(stmt)
            ]

    def count(self, user_id: str, tier: str | None = None) -> int:
        with SessionLocal() as db:
            stmt = select(Memory).where(Memory.user_id == user_id)
            if tier:
                stmt = stmt.where(Memory.tier == tier)
            return len(db.scalars(stmt).all())

    # ---------- 写 ----------
    def set(self, user_id: str, key: str, value: dict,
            tier: str = "core", topic: str = "") -> None:
        """写入/更新一条记忆（同 key 覆盖；tier/topic 一并落库）。"""
        now = _now()
        with self._lock(user_id), SessionLocal() as db:
            row = db.scalar(select(Memory).where(
                Memory.user_id == user_id, Memory.key == key))
            if row:
                row.value = value
                row.tier = tier
                row.topic = topic
                row.updated_at = now
                row.last_used_at = now
            else:
                db.add(Memory(user_id=user_id, key=key, value=value,
                              tier=tier, topic=topic,
                              updated_at=now, last_used_at=now))
            db.commit()

    def touch(self, user_id: str, keys: list[str]) -> None:
        """LRU：注入系统提示词 = 使用，批量刷新最近使用时间。"""
        if not keys:
            return
        with SessionLocal() as db:
            stmt = select(Memory).where(Memory.user_id == user_id,
                                        Memory.key.in_(keys))
            for m in db.scalars(stmt):
                m.last_used_at = _now()
            db.commit()

    def set_tier(self, user_id: str, key: str, tier: str) -> bool:
        """手动调级（置顶核心/降为短期）；晋升 core 时刷新使用时间。"""
        with SessionLocal() as db:
            row = db.scalar(select(Memory).where(
                Memory.user_id == user_id, Memory.key == key))
            if row is None:
                return False
            row.tier = tier
            if tier == "core":
                row.last_used_at = _now()
            db.commit()
            return True

    def delete_many(self, user_id: str, keys: list[str]) -> int:
        """选择性删除（单个或批量），返回实际删除条数。"""
        if not keys:
            return 0
        with SessionLocal() as db:
            rows = db.scalars(select(Memory).where(
                Memory.user_id == user_id, Memory.key.in_(keys))).all()
            for r in rows:
                db.delete(r)
            db.commit()
            return len(rows)

    # ---------- 容量管线原语 ----------
    def topic_groups(self, user_id: str) -> dict[str, list[dict]]:
        """同主题 ≥2 条的分组（主题压缩候选）：{topic: [{key, value}]}。

        topic 为空时用 key 首段兜底分组（research_xxx → research）。
        """
        groups: dict[str, list[dict]] = {}
        with SessionLocal() as db:
            rows = db.scalars(select(Memory).where(Memory.user_id == user_id))
            for m in rows:
                v = m.value.get("v", "") if isinstance(m.value, dict) else ""
                topic = m.topic or m.key.split("_")[0]
                groups.setdefault(topic, []).append({"key": m.key, "value": v})
        return {t: g for t, g in groups.items() if len(g) >= 2}

    def apply_merge(self, user_id: str, merged: list[dict]) -> int:
        """应用合并结果：每组 N 条 → 1 条（组内第一个 key 覆盖，其余删除）。

        merged: [{"topic", "key", "value"}]；返回腾出的条数。
        """
        freed = 0
        with self._lock(user_id), SessionLocal() as db:
            for item in merged:
                topic = str(item.get("topic") or "")[:64]
                group_keys = [str(k) for k in item.get("group_keys", [])]
                new_key = str(item.get("key") or (group_keys[0] if group_keys else ""))
                if not new_key:
                    continue
                old_keys = [k for k in group_keys if k != new_key]
                rows = db.scalars(select(Memory).where(
                    Memory.user_id == user_id,
                    Memory.key.in_(old_keys))).all()
                for r in rows:
                    db.delete(r)
                freed += len(rows)
                now = _now()
                row = db.scalar(select(Memory).where(
                    Memory.user_id == user_id, Memory.key == new_key))
                old_value = row.value if row else {}
                merged_from = list(old_value.get("merged_from", [])
                                   if isinstance(old_value, dict) else []) + old_keys
                if row:
                    row.value = {"v": item.get("value", ""), "merged_from": merged_from}
                    row.topic = topic
                    row.updated_at = now
                    row.last_used_at = now
                else:
                    db.add(Memory(user_id=user_id, key=new_key,
                                  value={"v": item.get("value", ""),
                                         "merged_from": merged_from},
                                  tier="core", topic=topic,
                                  updated_at=now, last_used_at=now))
            db.commit()
        return freed

    def evict_overflow(self, user_id: str, limit: int = MEMORY_MAX) -> int:
        """LRU 淘汰到 limit 以内：先 short（最旧优先），全 short 不够再 core 兜底。"""
        evicted = 0
        with self._lock(user_id), SessionLocal() as db:
            def _victim(tier: str) -> Memory | None:
                return db.scalars(select(Memory).where(
                    Memory.user_id == user_id, Memory.tier == tier)
                    .order_by(Memory.last_used_at.asc().nulls_first())
                    .limit(1)).first()

            total = len(db.scalars(select(Memory).where(
                Memory.user_id == user_id)).all())
            while total > limit:
                row = _victim("short") or _victim("core")
                if row is None:
                    break
                db.delete(row)
                db.flush()      # SessionLocal autoflush=False：先落库，否则下轮 SELECT 还能查到已删行
                total -= 1
                evicted += 1
            if evicted:
                db.commit()
        return evicted

    def expire_short(self, user_id: str,
                     ttl_days: int = SHORT_TTL_DAYS) -> int:
        """short 层 TTL 清理：updated_at 早于 cutoff 的删除（惰性触发）。"""
        cutoff = _now() - timedelta(days=ttl_days)
        with SessionLocal() as db:
            rows = db.scalars(select(Memory).where(
                Memory.user_id == user_id,
                Memory.tier == "short",
                Memory.updated_at < cutoff)).all()
            for r in rows:
                db.delete(r)
            if rows:
                db.commit()
            return len(rows)

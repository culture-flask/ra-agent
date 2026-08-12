"""长期记忆服务 ：用户级命名空间，跨会话读写。

memories 表（user_id/key/value/updated_at）启用。
key 是记忆条目名（如 research_topic），value 是任意 JSON 内容——
"键值 + JSON 值"模式，不用给每种记忆建表。
"""

from datetime import datetime, timezone

from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import Memory

MAX_MEMORIES_PER_USER = 50     # 每用户记忆条数上限（写入前审核的一部分）


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryService:
    """记忆读写：全部按 user_id 命名空间隔离。"""

    def get_all(self, user_id: str) -> dict[str, dict]:
        """读取该用户全部记忆：{key: value}。跨会话读取的入口。"""
        with SessionLocal() as db:
            stmt = select(Memory).where(Memory.user_id == user_id)
            return {m.key: m.value for m in db.scalars(stmt)}

    def set(self, user_id: str, key: str, value: dict) -> None:
        """写入/更新一条记忆（同 key 覆盖，保留最新）。"""
        with SessionLocal() as db:
            row = db.scalar(select(Memory).where(
                Memory.user_id == user_id, Memory.key == key))
            if row:
                row.value = value
                row.updated_at = _now()
            else:
                db.add(Memory(user_id=user_id, key=key, value=value,
                              updated_at=_now()))
            db.commit()

    def count(self, user_id: str) -> int:
        with SessionLocal() as db:
            return len(db.scalars(select(Memory).where(Memory.user_id == user_id)).all())

    def list(self, user_id: str) -> list[dict]:
        """供 API 展示：含更新时间。"""
        with SessionLocal() as db:
            stmt = select(Memory).where(Memory.user_id == user_id)
            return [
                {"key": m.key, "value": m.value,
                 "updated_at": m.updated_at.isoformat() if m.updated_at else None}
                for m in db.scalars(stmt)
            ]
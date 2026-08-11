"""ToolCallLog 追踪：Postgres 落库 + 层级 parent 追踪 + 查询。

tool_call_log 表（kind/name/args/output/error/parent_id/started_at/finished_at）
今天启用。每次 LLM/工具/检索调用都写一条，parent_id 形成调用树。
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.db import SessionLocal
from app.core.events import emit
from app.models import ToolCallLog


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Tracer:
    """调用追踪：start 返回 log_id，success/error 收尾。"""

    def start(self, kind: str, name: str, session_id: str, user_id: str,
              args: dict | None = None, parent_id: str | None = None) -> str:
        log_id = uuid.uuid4().hex[:16]
        with SessionLocal() as db:
            db.add(ToolCallLog(
                id=log_id, kind=kind, name=name, session_id=session_id,
                user_id=user_id, args=args or {}, parent_id=parent_id,
                started_at=_now(),
            ))
            db.commit()
        emit("trace_start", {"id": log_id, "kind": kind, "name": name})
        return log_id

    def success(self, log_id: str, output: str) -> None:
        with SessionLocal() as db:
            row = db.get(ToolCallLog, log_id)
            if row:
                row.output = output[:4000]
                row.finished_at = _now()
                db.commit()
        emit("trace_end", {"id": log_id, "ok": True, "output": output[:500]})

    def error(self, log_id: str, message: str) -> None:
        with SessionLocal() as db:
            row = db.get(ToolCallLog, log_id)
            if row:
                row.error = message[:2000]
                row.finished_at = _now()
                db.commit()
        emit("trace_end", {"id": log_id, "ok": False, "error": message[:500]})

    def list(self, user_id: str | None = None, session_id: str | None = None) -> list[dict]:
        with SessionLocal() as db:
            stmt = select(ToolCallLog).order_by(ToolCallLog.started_at.desc())
            if user_id:
                stmt = stmt.where(ToolCallLog.user_id == user_id)
            if session_id:
                stmt = stmt.where(ToolCallLog.session_id == session_id)
            rows = db.scalars(stmt.limit(200)).all()
            return [
                {
                    "id": r.id, "kind": r.kind, "name": r.name,
                    "session_id": r.session_id, "user_id": r.user_id,
                    "args": r.args, "output": r.output, "error": r.error,
                    "parent_id": r.parent_id,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                }
                for r in rows
            ]
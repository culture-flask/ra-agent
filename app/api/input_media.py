"""共享的多模态请求约定、附件校验与会话归属闸门。"""

import hashlib

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool
from sqlalchemy import text

from app.core.db import SessionLocal
from app.models import BrainstormSession, Conversation, utcnow


class MultimodalOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: StrictBool = False
    audio: StrictBool = False
    video: StrictBool = False


def ensure_session_owner(user_id: str, session_id: str) -> None:
    """普通聊天的辅助端点也须校验共用命名空间中的团队会话。"""
    with SessionLocal() as db:
        rows = (db.get(Conversation, session_id), db.get(BrainstormSession, session_id))
        if any(row is not None and row.user_id != user_id for row in rows):
            raise HTTPException(403, "not session owner")


async def ensure_checkpoint_owner(graph, config: dict, user_id: str) -> None:
    """SQL 登记丢失时，仍不能让其他用户接管尚存的 checkpoint。

    删除会话的 checkpoint 清理可能失败；旧版本也可能留下未登记状态。
    必须在 reserve_session 之前检查，避免先将他人孤儿状态登记给请求者。
    """
    snapshot = await graph.aget_state(config)
    values = (snapshot.values or {}) if snapshot else {}
    owner = values.get("user_id")
    if owner and owner != user_id:
        raise HTTPException(403, "not checkpoint owner")


def prepare_attachments(message: str, attachments: list[str], user_id: str,
                        options: MultimodalOptions, text_cache: dict,
                        media_service=None) -> tuple[str, list[dict]]:
    """返回文档拼接文本与持久媒体引用；新附件关闭/缺失/越权均显式拒绝。"""
    texts, refs = [], []
    for file_id in dict.fromkeys(attachments):
        cached = text_cache.get(file_id)
        if cached is not None:
            if cached.get("user_id") != user_id:
                raise HTTPException(404, "附件不存在、已过期或不属于当前用户")
            texts.append(f"[附件：{cached['filename']}]\n{cached['text']}")
            continue
        try:
            if media_service is None:
                raise FileNotFoundError()
            metadata = media_service.metadata(user_id, file_id)
            kind = metadata["kind"]
            if not options.model_dump().get(kind):
                label = {"image": "图片", "audio": "音频", "video": "视频"}.get(kind, kind)
                raise HTTPException(400, f"{label}输入已关闭，请启用后发送或移除附件")
            refs.extend(media_service.reference_blocks(user_id, file_id))
        except HTTPException:
            raise
        except (FileNotFoundError, PermissionError, ValueError, KeyError):
            raise HTTPException(404, "附件不存在、已过期或不属于当前用户") from None
    if texts:
        message += "\n<<<附件>>>\n" + "\n\n".join(texts)
    return message, refs


def reserve_session(user_id: str, session_id: str, *, team: str | None = None,
                    title: str = "", topic: str = "", roles: list | None = None) -> None:
    """先声明会话归属，再访问共享 checkpoint，避免媒体通过复用 ID 泄漏。

    同一 PostgreSQL 事务对 session_id 加 advisory lock，使跨表的新会话声明
    也串行执行。锁只覆盖登记事务，不包住长时间模型调用。
    """
    with SessionLocal() as db:
        if db.bind.dialect.name == "postgresql":
            key = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:8],
                                 "big", signed=True)
            db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
        conversation = db.get(Conversation, session_id)
        session = db.get(BrainstormSession, session_id)
        if any(row is not None and row.user_id != user_id
               for row in (conversation, session)):
            raise HTTPException(403, "not session owner")
        if team is None:
            if session is not None:
                raise HTTPException(409, "session_id already belongs to an agent team")
            if conversation is None:
                db.add(Conversation(session_id=session_id, user_id=user_id,
                                    title=(title or "新会话")[:128]))
            else:
                conversation.updated_at = utcnow()
        else:
            if conversation is not None or (session is not None and session.team != team):
                raise HTTPException(409, "session_id belongs to another workflow")
            if session is None:
                db.add(BrainstormSession(session_id=session_id, user_id=user_id,
                    team=team, topic=topic[:2000], roles=roles or [], status="running"))
            else:
                session.status = "running"
                session.topic = topic[:2000]
                session.roles = roles or []
                session.final_proposal = ""
                session.updated_at = utcnow()
        db.commit()

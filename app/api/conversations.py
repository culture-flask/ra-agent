"""会话跨设备同步：服务端会话列表 + 历史读取（LangGraph checkpoint）+ 删除。

此前会话 id 与消息全存浏览器 localStorage，换设备/清缓存即不可见。
现在每次发消息后端登记 conversations 表（见 chat._register_conversation），
列表与历史以服务端为准；消息本体仍从 checkpointer 读取（单一事实来源，
不重复存一份）。历史恢复是纯文本（steps/引用面板等富信息仅存于原设备）。
"""

import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select

from app.api.chat import _ATT_MARK
from app.core.db import SessionLocal
from app.core.logging import get_logger
from app.models import Conversation

logger = get_logger("conversations")

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def _get_row(session_id: str) -> Conversation | None:
    with SessionLocal() as db:
        return db.get(Conversation, session_id)


def _split_attachments(content) -> tuple[str, list[str]]:
    """拆出问题正文与附件名列表（历史还原为前端 chips 用）。

    _initial_state 把附件拼成「问题 + 标记 + 附件块」；无标记的老消息原样返回。
    """
    text = content if isinstance(content, str) else str(content or "")
    if _ATT_MARK not in text:
        return text, []
    question, _, block = text.partition(_ATT_MARK)
    names = re.findall(r"\[附件：(.+?)\]", block)
    return question.strip(), names


def _content_text(content) -> str:
    """消息 content 统一为纯文本（多模态 list 取各 text 部分拼接）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict))
    return str(content or "")


@router.get("")
async def list_conversations(user_id: str = "u1"):
    """该用户的会话列表（按最近活跃倒序），跨设备共享。"""
    def _list() -> list[dict]:
        with SessionLocal() as db:
            rows = db.scalars(select(Conversation).where(
                Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc()).limit(200)).all()
            return [{"session_id": r.session_id, "title": r.title,
                     "updated_at": r.updated_at.isoformat()}
                    for r in rows]
    return {"conversations": await run_in_threadpool(_list)}


@router.get("/{session_id}/messages")
async def conversation_messages(session_id: str, request: Request,
                                user_id: str = "u1"):
    """会话历史：从 checkpointer 读消息（user/assistant 交替的纯文本）。

    未登记的会话视为没聊过，返回空列表（前端当作新会话处理）。
    """
    row = await run_in_threadpool(_get_row, session_id)
    if row is None:
        return {"session_id": session_id, "messages": []}
    if row.user_id != user_id:
        raise HTTPException(status_code=403, detail="not conversation owner")
    graph = request.app.state.graph
    snap = await graph.aget_state({"configurable": {"thread_id": session_id}})
    values = (snap.values or {}) if snap else {}
    out: list[dict] = []
    for m in values.get("messages", []):
        t = getattr(m, "type", "")
        if t == "human":
            question, names = _split_attachments(getattr(m, "content", ""))
            msg = {"role": "user", "content": question}
            if names:
                msg["files"] = [{"name": n} for n in names]
            out.append(msg)
        elif t == "ai":
            text = _content_text(getattr(m, "content", ""))
            if text.strip():                     # 跳过纯工具调用（content 为空）的中间轮
                out.append({"role": "assistant", "content": text})
    return {"session_id": session_id, "messages": out}


class ConversationRenameRequest(BaseModel):
    """会话改名（用户自定义标题，方便后续找到对话）。"""
    title: str


@router.patch("/{session_id}")
async def rename_conversation(session_id: str, req: ConversationRenameRequest,
                              user_id: str = "u1"):
    """修改会话名称（仅本人）。改名后不被自动标题覆盖——
    发消息只刷新 updated_at，title 仅首次登记时生成（见 chat._register_conversation）。
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话名称不能为空")
    row = await run_in_threadpool(_get_row, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if row.user_id != user_id:
        raise HTTPException(status_code=403, detail="not conversation owner")

    def _rename():
        with SessionLocal() as db:
            obj = db.get(Conversation, session_id)
            obj.title = title[:128]
            db.commit()
    await run_in_threadpool(_rename)
    return {"session_id": session_id, "title": title[:128]}


@router.delete("/{session_id}")
async def delete_conversation(session_id: str, request: Request,
                              user_id: str = "u1"):
    """删除会话：登记行 + checkpoint 线程一并清理（仅本人可删）。"""
    row = await run_in_threadpool(_get_row, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if row.user_id != user_id:
        raise HTTPException(status_code=403, detail="not conversation owner")

    def _delete():
        with SessionLocal() as db:
            obj = db.get(Conversation, session_id)
            if obj is not None:
                db.delete(obj)
                db.commit()
    await run_in_threadpool(_delete)
    try:
        await request.app.state.graph.adelete_thread(session_id)
    except Exception as e:                        # checkpoint 删除失败不阻断（行已删，列表不可见）
        logger.warning("checkpoint delete failed for %s: %s", session_id, e)
    return {"deleted": session_id}

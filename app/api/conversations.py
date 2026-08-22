"""会话跨设备同步：服务端会话列表 + 历史读取（LangGraph checkpoint）+ 删除。

此前会话 id 与消息全存浏览器 localStorage，换设备/清缓存即不可见。
现在每次发消息后端登记 conversations 表（见 chat._register_conversation），
列表与历史以服务端为准；消息本体仍从 checkpointer 读取（单一事实来源，
不重复存一份）。历史接口只回纯文本骨架（role/content——steps/引用溯源
等富信息不入 checkpoint），前端拉取后会与本地富缓存做**尾部对齐合并不
降级**（P3-30）：同设备上执行过程面板/引用溯源不会因历史补全而丢失。

P0-1 鉴权改造：身份一律取自 Bearer token（core.deps.get_current_user），
客户端传入的 user_id 参数废除——此前任何人改个参数就能读/删他人会话。
"""

import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select

from app.api.chat import _ATT_MARK
from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.core.logging import get_logger
from app.models import Conversation, User

logger = get_logger("conversations")

# 路由级闸门：本组全部端点要求登录态
router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"],
                   dependencies=[Depends(get_current_user)])


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
async def list_conversations(user: User = Depends(get_current_user)):
    """当前用户的会话列表（按最近活跃倒序），跨设备共享。"""
    uid = user.id

    def _list() -> list[dict]:
        with SessionLocal() as db:
            rows = db.scalars(select(Conversation).where(
                Conversation.user_id == uid)
                .order_by(Conversation.updated_at.desc()).limit(200)).all()
            return [{"session_id": r.session_id, "title": r.title,
                     "updated_at": r.updated_at.isoformat()}
                    for r in rows]
    return {"conversations": await run_in_threadpool(_list)}


@router.get("/{session_id}/messages")
async def conversation_messages(session_id: str, request: Request,
                                user: User = Depends(get_current_user)):
    """会话历史：从 checkpointer 读消息（user/assistant 交替的纯文本）。

    未登记的会话视为没聊过，返回空列表（前端当作新会话处理）；
    已登记但不属于当前用户 → 403（不暴露他人会话存在性之外的任何内容）。
    """
    row = await run_in_threadpool(_get_row, session_id)
    if row is None:
        return {"session_id": session_id, "messages": []}
    if row.user_id != user.id:
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
                              user: User = Depends(get_current_user)):
    """修改会话名称（仅本人）。改名后不被自动标题覆盖——
    发消息只刷新 updated_at，title 仅首次登记时生成（见 chat._register_conversation）。
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话名称不能为空")
    row = await run_in_threadpool(_get_row, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if row.user_id != user.id:
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
                              user: User = Depends(get_current_user)):
    """删除会话：登记行 + checkpoint 线程一并清理（仅本人可删）。"""
    row = await run_in_threadpool(_get_row, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if row.user_id != user.id:
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

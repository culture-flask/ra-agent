"""用户反馈：点赞/点踩一条回答，为检索评测集供数（P3-19 反馈闭环）。

- 写入即资产：rating=1 的记录由 rag_test/build_golden_from_feedback.py
  半自动整理成金标准行，评测集随真实使用增长，不再只靠人工养 26 条；
- 身份取自 Bearer token（P0-1），hits 由前端引用面板的来源元数据冗余而来，
  服务端做尺寸约束（条数/字段长度），防超大 payload 与滥用。
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.models import Feedback, User

# 路由级闸门：本组全部端点要求登录态
router = APIRouter(prefix="/api/v1", tags=["feedbacks"],
                   dependencies=[Depends(get_current_user)])

_QUESTION_MAX = 4000        # 问题截断（与长问题评测分布匹配绰绰有余）
_ANSWER_MAX = 8000          # 回答截断（对齐附件解析的字符上限思路）
_HITS_MAX = 20              # 引用条数上限（前端面板单轮最多 total_k=50，
                            # 但金标准只需要文档级来源，20 条足够且防滥用）


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int                          # 仅接受 1 赞 / -1 踩（validator 收口）

    @field_validator("rating")
    @classmethod
    def _rating_valid(cls, v: int) -> int:
        # 不用 ge/le 区间：0 没有信息量，混进评测集是噪声，显式拒绝
        if v not in (-1, 1):
            raise ValueError("rating 必须是 1（赞）或 -1（踩）")
        return v
    question: str = ""
    answer: str = ""
    hits: list[dict] = []
    comment: str = Field(default="", max_length=1000)


@router.post("/feedbacks", status_code=201)
async def create_feedback(req: FeedbackRequest, request: Request,
                          user: User = Depends(get_current_user)):
    """记录当前用户对一条回答的反馈（question/answer 服务端截断）。"""
    def _save() -> dict:
        with SessionLocal() as db:
            row = Feedback(
                user_id=user.id, session_id=req.session_id[:36],
                rating=req.rating,
                question=req.question.strip()[:_QUESTION_MAX],
                answer=req.answer[:_ANSWER_MAX],
                hits=[{k: v for k, v in h.items() if isinstance(v, (str, int, float))}
                      for h in req.hits[:_HITS_MAX]],
                comment=req.comment.strip(),
            )
            db.add(row)
            db.commit()
            return {"id": row.id}
    out = await run_in_threadpool(_save)
    return {"feedback_id": out["id"], "rating": req.rating}


@router.get("/feedbacks")
async def list_feedbacks(request: Request,
                         user: User = Depends(get_current_user),
                         limit: int = 50):
    """当前用户自己的反馈列表（最近优先；给后续的个人历史/导出用）。"""
    from sqlalchemy import select

    limit = max(1, min(limit, 200))

    def _list() -> list[dict]:
        with SessionLocal() as db:
            rows = db.scalars(select(Feedback).where(
                Feedback.user_id == user.id)
                .order_by(Feedback.created_at.desc()).limit(limit)).all()
            return [{"id": r.id, "session_id": r.session_id,
                     "rating": r.rating,
                     "question": r.question[:80],
                     "created_at": r.created_at.isoformat()} for r in rows]
    return await run_in_threadpool(_list)


@router.get("/feedbacks/{feedback_id}")
async def get_feedback(feedback_id: str, request: Request,
                       user: User = Depends(get_current_user)):
    """单条详情（含 hits）：仅本人可见——他人 id 一律按不存在处理。"""
    def _get():
        with SessionLocal() as db:
            return db.get(Feedback, feedback_id)
    row = await run_in_threadpool(_get)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="feedback not found")
    return {
        "id": row.id, "user_id": row.user_id, "session_id": row.session_id,
        "rating": row.rating, "question": row.question, "answer": row.answer,
        "hits": row.hits, "comment": row.comment,
        "created_at": row.created_at.isoformat(),
    }

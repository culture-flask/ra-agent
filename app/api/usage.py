"""Token 用量汇总 API（P3-20）：按天 × 模型聚合当前用户的真实用量。

数据源是 llm_usage 表（generate_node 每轮 best-effort 落库），
供成本报表、模型对比与后续配额演进使用；仅本人可见。
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select

from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.models import LLMUsage, User, utcnow

router = APIRouter(prefix="/api/v1", tags=["usage"],
                   dependencies=[Depends(get_current_user)])


@router.get("/usage/summary")
async def usage_summary(request: Request,
                        user: User = Depends(get_current_user),
                        days: int = Query(default=30, ge=1, le=365)):
    """按天 × 模型聚合的用量报表：{days, days:[{date, models, totals}], grand}。"""
    since = utcnow() - timedelta(days=days)

    def _query() -> list[dict]:
        with SessionLocal() as db:
            rows = db.execute(
                select(
                    func.date(LLMUsage.created_at).label("d"),
                    LLMUsage.model,
                    func.sum(LLMUsage.input_tokens).label("inp"),
                    func.sum(LLMUsage.output_tokens).label("outp"),
                    func.sum(LLMUsage.total_tokens).label("tot"),
                    func.count().label("calls"),
                ).where(LLMUsage.user_id == user.id,
                        LLMUsage.created_at >= since)
                .group_by(func.date(LLMUsage.created_at), LLMUsage.model)
                .order_by(func.date(LLMUsage.created_at).desc())
            ).all()
            return [dict(r._mapping) for r in rows]

    raw = await run_in_threadpool(_query)

    days_out: dict[str, dict] = {}
    grand = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
    for row in raw:
        date = str(row["d"])
        day = days_out.setdefault(date, {
            "date": date, "models": [],
            "totals": {"input_tokens": 0, "output_tokens": 0,
                       "total_tokens": 0, "calls": 0}})
        item = {"model": row["model"],
                "input_tokens": int(row["inp"] or 0),
                "output_tokens": int(row["outp"] or 0),
                "total_tokens": int(row["tot"] or 0),
                "calls": int(row["calls"] or 0)}
        day["models"].append(item)
        for k in ("input_tokens", "output_tokens", "total_tokens", "calls"):
            day["totals"][k] += item[k]
            grand[k] += item[k]

    return {"days": days or 30,
            "items": sorted(days_out.values(), key=lambda x: x["date"], reverse=True),
            "grand": grand}

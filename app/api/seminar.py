"""格致会讲 API：SSE 流式跑研讨子图 + 停止 + 会话列表/详情。

与 brainstorm.py 同构（run_graph 后台任务 + event_gen drain）；
差异：team='seminar'、无 max_rounds（议程由学者数确定）、
详情读会讲特有字段（洞见池/构想卡/评分）。
"""

import asyncio
import contextlib
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.chat import _ATT_MARK, _CHAT_FILES
from app.core.cancel import clear_stop, request_stop
from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.core.events import clear_event_sink, set_event_sink
from app.core.logging import get_logger
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.agent_runtime import effective_cfg as _effective_cfg, role_cfg as _role_cfg
from app.graph.native_llm import native_round
from app.graph.seminar import (SCHOLAR_IDS, _cards_digest, _insight_digest,
                               _notes_digest, SCHOLAR_ORIENTATION)
from app.graph.workflow import aget_state_retry
from app.models import BrainstormSession, KnowledgeBase, User, utcnow

logger = get_logger("seminar")

router = APIRouter(prefix="/api/v1/seminar", tags=["seminar"],
                   dependencies=[Depends(get_current_user)])


class SeminarRequest(BaseModel):
    session_id: str = Field(min_length=1)     # 建议 sem_ 前缀
    topic: str = Field(min_length=1, max_length=8000)
    token_budget: int | None = Field(None, ge=10_000, le=100_000_000)
    roles: list[dict] | None = None          # 与会学者子集（id 限四学者）
    attachments: list[str] = []


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_roles(raw: list[dict] | None, settings) -> list[dict]:
    """角色解析：id 白名单（四学者）、去重、温度夹取、config_id/model_id
    透传——与 brainstorm._resolve_roles 同款逻辑，白名单换成 SCHOLAR_IDS。"""
    roles = getattr(settings, "seminar_roles", None) or []
    defaults = {r["id"]: r for r in roles if r.get("id") in SCHOLAR_IDS}
    if not raw:
        return list(defaults.values())
    out, seen = [], set()
    for r in raw:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", "")).strip()
        if rid not in defaults or rid in seen:
            continue
        seen.add(rid)
        try:
            temp = max(0.0, min(2.0,
                        float(r.get("temperature", defaults[rid]["temperature"]))))
        except (TypeError, ValueError):
            temp = defaults[rid]["temperature"]
        role = {"id": rid,
                "name": str(r.get("name") or defaults[rid]["name"])[:20],
                "temperature": temp}
        if r.get("config_id"):
            role["config_id"] = str(r["config_id"])[:36]
        if r.get("model_id"):
            role["model_id"] = str(r["model_id"]).strip()[:64]
        out.append(role)
    return out or list(defaults.values())


def _initial_state(req: SeminarRequest, user_id: str, settings,
                   roles: list[dict]) -> dict:
    topic = req.topic
    if req.attachments:
        blocks = [f"[附件：{att['filename']}]\n{att['text']}"
                  for fid in req.attachments
                  if (att := _CHAT_FILES.get(fid))]
        if blocks:
            topic = topic + _ATT_MARK + "\n\n".join(blocks)
    return {
        "user_id": user_id, "session_id": req.session_id, "topic": topic,
        "roles": roles,
        "token_budget": int(req.token_budget
                            or getattr(settings, "seminar_token_budget",
                                       20000000)),
        "agenda_index": 0, "token_budget_used": 0,
        "reading_notes": [], "evidence_pool": [], "qa_transcript": [],
        "insight_board": [], "open_questions": [], "idea_cards": [],
        "review_scores": [], "stopped": False,
    }


def _register_session(user_id: str, req: SeminarRequest, roles: list) -> None:
    with SessionLocal() as db:
        row = db.get(BrainstormSession, req.session_id)
        if row is None:
            db.add(BrainstormSession(
                session_id=req.session_id, user_id=user_id,
                topic=req.topic[:2000], status="running",
                roles=roles, team="seminar"))          # ← 差异点
        else:
            row.status = "running"
            row.final_proposal = ""
            row.updated_at = utcnow()
        db.commit()


def _finish_session(user_id: str, session_id: str, result: dict) -> None:
    stopped = bool(result.get("stopped"))
    stats = {"presents": len([e for e in result.get("qa_transcript") or []
                              if e.get("kind") == "present"]),
             "insights": len(result.get("insight_board") or []),
             "cards": len(result.get("card_registry") or []),
             "tokens": int(result.get("token_budget_used") or 0)}
    if result.get("budget_exhausted"):
        stats["budget_exhausted"] = True
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
        if row is not None and row.user_id == user_id:
            row.status = "stopped" if stopped else "done"
            row.final_proposal = str(result.get("final_proposal") or "")
            row.stats = stats
            db.commit()


def _fail_session(user_id: str, session_id: str, error: str) -> None:
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
        if row is not None and row.user_id == user_id:
            row.status = "failed"
            row.stats = {"error": error[:500]}
            db.commit()


SEM_OUTPUT_KB_NAME = "研讨式agent纪要"       # 会讲的知识沉淀库（自动创建；不会被普通对话检索）

# 沉淀库滚动窗口：每库最多保留的文档数（超过删最旧，防重复膨胀与召回稀释）
ARCHIVE_KEEP_DOCS = 10


def _cap_archive_docs(kb_service, kb_id: str, keep: int = ARCHIVE_KEEP_DOCS) -> int:
    """沉淀库滚动窗口：按文件名（前缀为日期，字典序即时间序）保留最新
    keep 份，删除更旧的。返回删除数。失败只记日志，不阻断沉淀。"""
    try:
        docs = kb_service.list_documents(kb_id)
        if len(docs) <= keep:
            return 0
        dated = sorted(docs, key=lambda d: d.get("filename") or "")
        stale = [d["doc_id"] for d in dated[:len(dated) - keep]]
        if stale:
            kb_service.delete_documents(kb_id, stale)
        return len(stale)
    except Exception as e:
        logger.warning("archive cap failed kb=%s: %s", kb_id, e)
        return 0


def _archive_report_to_kb(user_id: str, session_id: str,
                          topic: str, report: str) -> None:
    """会讲知识闭环：成稿入库（与争鸣社 _archive_proposal_to_kb 同构）。"""
    if not report.strip():
        return
    try:
        from app.main import app
        kb_service = app.state.kb_service
        with SessionLocal() as db:
            kb = db.scalar(select(KnowledgeBase).where(
                KnowledgeBase.name == SEM_OUTPUT_KB_NAME,
                KnowledgeBase.owner_user_id == user_id))
            if kb is None:
                kb = kb_service.create_kb(
                    SEM_OUTPUT_KB_NAME, "private", user_id,
                    description="研讨式多 agent 自动沉淀：历场研讨报告与构想组合"
                                 "（专用于研讨团队，不会被普通对话检索）",
                    kind="archive")
            kb_id = kb.kb_id
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{date}-会讲-{session_id[:12]}.md"   # 日期前缀：文件名字典序即时间序
        text = (f"# [{date}] 格致会讲报告（会话 {session_id[:12]}）\n"
                f"议题：{topic}\n\n{report}")
        kb_service.add_documents(kb_id, [text], filenames=[filename])
        _cap_archive_docs(kb_service, kb_id)
    except Exception as e:
        logger.warning("seminar archive to kb failed %s: %s", session_id, e)


@router.post("/stream")
async def seminar_stream(req: SeminarRequest, request: Request,
                         user: User = Depends(get_current_user)):
    """SSE 流式。事件流（图内 emit，sem_ 前缀）：
    sem_curate / sem_retrievals / sem_tool_start / sem_tool_end /
    sem_agent_start / sem_agent_end / sem_token / sem_reasoning /
    sem_present_end / sem_insight / sem_idea_card / sem_review /
    sem_ranking / sem_plan，最终 done。
    """
    uid = user.id
    graph = request.app.state.seminar_graph
    settings = request.app.state.settings
    sink: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": req.session_id},
              "recursion_limit": 200}

    async def run_graph():
        set_event_sink(sink)
        clear_stop(req.session_id)
        try:
            try:
                await graph.adelete_thread(req.session_id)
            except Exception:
                pass
            roles = _resolve_roles(req.roles, settings)
            await run_in_threadpool(_register_session, uid, req, roles)
            result = await graph.ainvoke(
                _initial_state(req, uid, settings, roles), config=config)
            await run_in_threadpool(_finish_session, uid, req.session_id, result)
            if not result.get("stopped"):
                await run_in_threadpool(_archive_report_to_kb, uid,
                                        req.session_id, req.topic,
                                        str(result.get("final_proposal") or ""))
            await sink.put({"type": "__done__"})
        except asyncio.CancelledError:     # 客户端断开 → running 复位 failed
            with contextlib.suppress(Exception):
                await asyncio.shield(run_in_threadpool(
                    _fail_session, uid, req.session_id, "client disconnected"))
            raise
        except Exception as e:
            logger.warning("seminar run failed: %s", e)
            await run_in_threadpool(_fail_session, uid, req.session_id, str(e))
            await sink.put({"type": "__error__", "error": str(e)})
        finally:
            clear_event_sink()

    async def event_gen():
        task = asyncio.create_task(run_graph())
        try:
            while True:
                ev = await sink.get()
                t = ev.get("type")
                if t == "__done__":
                    break
                if t == "__error__":
                    yield _sse({"type": "error", "error": ev.get("error", "")})
                    break
                yield _sse(ev)
            yield _sse({"type": "done"})
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_gen(), media_type="text/event-stream")


class StopRequest(BaseModel):
    session_id: str


@router.post("/stop")
async def seminar_stop(req: StopRequest, user: User = Depends(get_current_user)):
    with SessionLocal() as db:
        row = db.get(BrainstormSession, req.session_id)
        if row is not None and row.user_id != user.id:
            raise HTTPException(status_code=403, detail="not seminar owner")
    request_stop(req.session_id)
    return {"stopped": True, "session_id": req.session_id}


@router.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """当前用户的会讲列表（team='seminar' 过滤，与争鸣社分家）。"""
    uid = user.id

    def _list():
        with SessionLocal() as db:
            rows = db.scalars(select(BrainstormSession).where(
                BrainstormSession.user_id == uid,
                BrainstormSession.team == "seminar")     # ← 差异点
                .order_by(BrainstormSession.updated_at.desc()).limit(200)).all()
            return [{"session_id": r.session_id, "topic": r.topic[:100],
                     "status": r.status, "stats": r.stats,
                     "updated_at": r.updated_at.isoformat()} for r in rows]
    return {"sessions": await run_in_threadpool(_list)}


@router.get("/{session_id}")
async def session_detail(session_id: str, request: Request,
                         user: User = Depends(get_current_user)):
    """详情：笔记/洞见池/构想卡/评分/成稿（checkpoint 单一事实来源）。"""
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not seminar owner")

    graph = request.app.state.seminar_graph
    snap = await aget_state_retry(graph, {"configurable": {"thread_id": session_id}})
    values = (snap.values or {}) if snap else {}
    return {
        "session_id": session_id, "topic": row.topic, "status": row.status,
        "roles": row.roles, "stats": row.stats,
        "reading_notes": values.get("reading_notes", []),
        "qa_transcript": values.get("qa_transcript", []),
        "insight_board": values.get("insight_board", []),
        "open_questions": values.get("open_questions", []),
        "evidence_pool": values.get("evidence_pool", []),
        "card_registry": values.get("card_registry", []),
        "idea_ranking": values.get("idea_ranking", []),
        "final_proposal": values.get("final_proposal") or row.final_proposal,
    }


# ---------- 追问：会话结束后向点名学者（或主席）继续提问 ----------

class SeminarAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


# 触发词 → 角色别名表：问题文本中最早命中的学者被唤起；未命中 → 主席
SEM_QA_ALIASES = [
    ("historian", ["文献学家", "historian"]),
    ("theorist", ["理论家", "theorist"]),
    ("experimentalist", ["实验家", "experimentalist"]),
    ("visitor", ["访问学者", "visitor"]),
    ("rapporteur", ["执笔人", "执笔", "rapporteur"]),
    ("chair", ["主席", "主持人", "chair"]),
]


def _pick_agent(question: str, roles: list[dict]) -> str:
    """按触发词选唤起学者：问题中**最早出现**的身份优先（同一位置取更长
    别名）。未命中任何身份 → 主席（会讲的调度者是主席，不输出观点）。"""
    best: tuple[int, str] | None = None
    for agent_id, names in SEM_QA_ALIASES:
        for n in names:
            i = question.find(n)
            if i >= 0 and (best is None or i < best[0]
                           or (i == best[0] and len(n) > len(best[1]))):
                best = (i, agent_id)
    return best[1] if best else "chair"


@router.post("/{session_id}/ask")
async def seminar_ask(session_id: str, req: SeminarAskRequest,
                      request: Request,
                      user: User = Depends(get_current_user)):
    """会讲追问（SSE）：点名学者以其人设+绑定模型作答，未点名由主席作答。

    只读回放 checkpoint，不改变会话状态；追问不计入会讲预算。
    事件流：ask_start（agent/model）→ token → done / error。
    """
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not seminar owner")

    graph = request.app.state.seminar_graph
    ctx = request.app.state.workflow_ctx
    snap = await aget_state_retry(graph,
                                  {"configurable": {"thread_id": session_id}})
    values = dict((snap.values or {}) if snap else {})
    values.setdefault("user_id", user.id)   # _role_cfg 解析角色配置需要
    topic = values.get("topic") or row.topic
    roles = values.get("roles") or row.roles
    notes = values.get("reading_notes") or []
    cards = values.get("card_registry") or []
    ranking = values.get("idea_ranking") or []
    open_qs = values.get("open_questions") or []
    proposal = values.get("final_proposal") or row.final_proposal

    agent_id = _pick_agent(req.question, roles)
    is_scholar = agent_id in SCHOLAR_IDS
    role = next((r for r in roles if r.get("id") == agent_id), {}) \
        if is_scholar else {}
    if agent_id == "chair":
        cfg = _effective_cfg(ctx, user.id)
        system = ("你是本场格致会讲的主席，基于全场研讨客观回答用户追问；"
                  "引用观点请标明来自哪位学者，不清楚的如实说明。")
    elif agent_id == "rapporteur":
        cfg = _effective_cfg(ctx, user.id)
        system = ("你是本场格致会讲的执笔人，会讲报告由你执笔；"
                  "就报告内容与依据回答用户追问。")
    else:
        cfg = _role_cfg(ctx, values, agent_id)
        own = next((n.get("content", "") for n in notes
                    if n.get("agent_id") == agent_id), "（研读笔记缺失）")
        system = (f"你是本场格致会讲中的「{role.get('name', agent_id)}」。"
                  f"角色取向：{SCHOLAR_ORIENTATION[agent_id]}\n本场议题：{topic}\n\n"
                  f"你的研读笔记：\n{own[:1200]}\n\n"
                  "现在用户向你追问。以你的学者身份、基于你的研读与全场讨论"
                  "直接回答；与你的立场有出入时如实说明，不要扮演其他学者。")

    ranking_text = "\n".join(
        f"- {r.get('card_id')} 总分{r.get('total')}：{r.get('title')}"
        for r in ranking) or "（无）"
    system += (f"\n\n【全场参考】\n议题：{topic}\n\n学者研读笔记摘要：\n"
               f"{_notes_digest(values, limit=300)}\n\n洞见池：\n"
               f"{_insight_digest(values, 30)}\n\n构想卡：\n"
               f"{_cards_digest(cards, 200)}\n\n评审排序：\n{ranking_text}\n\n"
               f"开放问题：{'; '.join(q.get('q', '') for q in open_qs) or '（无）'}\n"
               f"最终报告（节选）：{proposal[:2000]}")
    temperature = float(role.get("temperature", 0.3))
    msgs = [SystemMessage(content=system),
            HumanMessage(content=req.question)]
    sid = session_id
    sink: asyncio.Queue = asyncio.Queue()

    async def run_ask():
        set_event_sink(sink)
        try:
            resp, _usage, stopped = await native_round(
                cfg, msgs, None, temperature, sid,
                token_event="token", reasoning_event="ask_reasoning")
            await sink.put({"type": "__done__"})
        except Exception as e:
            logger.warning("seminar ask failed: %s", e)
            await sink.put({"type": "__error__", "error": str(e)})
        finally:
            clear_event_sink()

    async def event_gen():
        task = asyncio.create_task(run_ask())
        try:
            yield _sse({"type": "ask_start", "agent": agent_id,
                        "model": getattr(cfg, "model_id", "") or ""})
            while True:
                ev = await sink.get()
                t = ev.get("type")
                if t == "__done__":
                    break
                if t == "__error__":
                    yield _sse({"type": "error", "error": ev.get("error", "")})
                    break
                yield _sse(ev)
            yield _sse({"type": "done"})
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_gen(), media_type="text/event-stream")

"""深度调研 API：SSE 流式跑调研子图 + 停止 + 会话列表/详情 + 会后追问。

与 seminar.py 同构（run_graph 后台任务 + event_gen drain）；
差异：team='deep_research'、事件前缀 dr_、新增 mode（完整/快速）与
recency（时效窗口）两个入参，详情读调研特有字段（大纲/章节/来源池/警告）。
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
from app.core.team_ctx import set_agent_team
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.agent_runtime import effective_cfg as _effective_cfg, role_cfg as _role_cfg
from app.graph.native_llm import native_round
from app.graph.deep_research import (DR_ROLE_IDS, make_write_frame_node,
                                     publish_node)
from app.graph.workflow import aget_state_retry
from app.models import BrainstormSession, KnowledgeBase, User, utcnow

logger = get_logger("deep_research")

router = APIRouter(prefix="/api/v1/deep-research", tags=["deep_research"],
                   dependencies=[Depends(get_current_user)])


class DeepResearchRequest(BaseModel):
    session_id: str = Field(min_length=1)     # 建议 dr_ 前缀
    topic: str = Field(min_length=1, max_length=8000)
    mode: str = "full"                        # full（含审稿闭环）| quick（跳过审稿）
    recency: str = "两年"                      # 时效窗口，注入 scout 提示词
    token_budget: int | None = Field(None, ge=10_000, le=100_000_000)
    roles: list[dict] | None = None
    attachments: list[str] = []


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_roles(raw: list[dict] | None, settings) -> list[dict]:
    """角色解析：id 白名单、去重、温度夹取、config_id/model_id 透传
    ——与 seminar._resolve_roles 同款逻辑，白名单换成 DR_ROLE_IDS。"""
    roles = getattr(settings, "deep_research_roles", None) or []
    defaults = {r["id"]: r for r in roles if r.get("id") in DR_ROLE_IDS}
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


def _initial_state(req: DeepResearchRequest, user_id: str, settings,
                   roles: list[dict]) -> dict:
    topic = req.topic
    if req.attachments:
        blocks = [f"[附件：{att['filename']}]\n{att['text']}"
                  for fid in req.attachments
                  if (att := _CHAT_FILES.get(fid))]
        if blocks:
            topic = topic + _ATT_MARK + "\n\n".join(blocks)
    mode = "quick" if str(req.mode).lower() == "quick" else "full"
    return {
        "user_id": user_id, "session_id": req.session_id, "topic": topic,
        "roles": roles, "mode": mode, "recency": str(req.recency)[:20],
        "token_budget": int(req.token_budget
                            or getattr(settings, "deep_research_token_budget",
                                       20000000)),
        "scouting_brief": "", "source_pool": [], "outline": {},
        "chapter_index": 0, "current_draft": "", "review_round": 0,
        "review_verdict": "", "review_feedback": "", "review_log": [],
        "chapters": [],
        "toc": "", "introduction": "", "conclusion": "", "references": [],
        "final_report": "", "warnings": [],
        "token_budget_used": 0, "stopped": False,
    }


def _register_session(user_id: str, req: DeepResearchRequest,
                      roles: list) -> None:
    with SessionLocal() as db:
        row = db.get(BrainstormSession, req.session_id)
        if row is None:
            db.add(BrainstormSession(
                session_id=req.session_id, user_id=user_id,
                topic=req.topic[:2000], status="running",
                roles=roles, team="deep_research"))       # ← 差异点
        else:
            row.status = "running"
            row.final_proposal = ""
            row.updated_at = utcnow()
        db.commit()


def _finish_session(user_id: str, session_id: str, result: dict) -> None:
    stopped = bool(result.get("stopped"))
    chapters = result.get("chapters") or []
    stats = {"chapters": len(chapters),
             "sources": len(result.get("source_pool") or []),
             "tokens": int(result.get("token_budget_used") or 0),
             "warnings": len(result.get("warnings") or []),
             "review_rounds": sum(int(c.get("rounds") or 0) for c in chapters)}
    if result.get("budget_exhausted"):
        stats["budget_exhausted"] = True
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
        if row is not None and row.user_id == user_id:
            row.status = "stopped" if stopped else "done"
            row.final_proposal = str(result.get("final_report") or "")
            row.stats = stats
            db.commit()


def _fail_session(user_id: str, session_id: str, error: str) -> None:
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
        if row is not None and row.user_id == user_id:
            row.status = "failed"
            row.stats = {"error": error[:500]}
            db.commit()


DR_OUTPUT_KB_NAME = "深度调研agent纪要"    # 本团队的知识沉淀库（自动创建）


def _archive_report_to_kb(user_id: str, session_id: str,
                          topic: str, report: str) -> None:
    """知识闭环：报告入库（与会讲同构），滚动保留最近若干份。"""
    if not report.strip():
        return
    try:
        from app.main import app
        kb_service = app.state.kb_service
        with SessionLocal() as db:
            kb = db.scalar(select(KnowledgeBase).where(
                KnowledgeBase.name == DR_OUTPUT_KB_NAME,
                KnowledgeBase.owner_user_id == user_id))
            if kb is None:
                kb = kb_service.create_kb(
                    DR_OUTPUT_KB_NAME, "private", user_id,
                    description="深度调研多 agent 自动沉淀：历场研究报告"
                                 "（专用于调研团队，不会被普通对话检索）",
                    kind="archive")
            kb_id = kb.kb_id
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{date}-调研-{session_id[:12]}.md"
        text = (f"# [{date}] 深度调研报告（会话 {session_id[:12]}）\n"
                f"课题：{topic}\n\n{report}")
        kb_service.add_documents(kb_id, [text], filenames=[filename])
        kb_service.cap_archive_docs(kb_id)
    except Exception as e:
        logger.warning("deep_research archive to kb failed %s: %s",
                       session_id, e)


@router.post("/stream")
async def dr_stream(req: DeepResearchRequest, request: Request,
                    user: User = Depends(get_current_user)):
    """SSE 流式。事件流（图内 emit，dr_ 前缀）：
    dr_scout / dr_plan / dr_sources / dr_agent_start / dr_agent_end /
    dr_token / dr_reasoning / dr_tool_start / dr_tool_end / dr_internal /
    dr_review / dr_chapter_done / dr_frame / dr_report，最终 done。
    """
    set_agent_team("deep_research")   # 团队身份：只检索普通库 + 本队沉淀库
    uid = user.id
    graph = request.app.state.deep_research_graph
    settings = request.app.state.settings
    sink: asyncio.Queue = asyncio.Queue()
    # 章节循环 × 审稿循环，递归深度高于前两支团队，上限相应放宽
    config = {"configurable": {"thread_id": req.session_id},
              "recursion_limit": 400}

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
                                        str(result.get("final_report") or ""))
            await sink.put({"type": "__done__"})
        except asyncio.CancelledError:     # 客户端断开 → running 复位 failed
            with contextlib.suppress(Exception):
                await asyncio.shield(run_in_threadpool(
                    _fail_session, uid, req.session_id, "client disconnected"))
            raise
        except Exception as e:
            logger.warning("deep_research run failed: %s", e)
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
async def dr_stop(req: StopRequest, user: User = Depends(get_current_user)):
    with SessionLocal() as db:
        row = db.get(BrainstormSession, req.session_id)
        if row is not None and row.user_id != user.id:
            raise HTTPException(status_code=403, detail="not session owner")
    request_stop(req.session_id)
    return {"stopped": True, "session_id": req.session_id}


@router.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """当前用户的调研列表（team='deep_research' 过滤，与另两支分家）。"""
    uid = user.id

    def _list():
        with SessionLocal() as db:
            rows = db.scalars(select(BrainstormSession).where(
                BrainstormSession.user_id == uid,
                BrainstormSession.team == "deep_research")     # ← 差异点
                .order_by(BrainstormSession.updated_at.desc()).limit(200)).all()
            return [{"session_id": r.session_id, "topic": r.topic[:100],
                     "status": r.status, "stats": r.stats,
                     "updated_at": r.updated_at.isoformat()} for r in rows]
    return {"sessions": await run_in_threadpool(_list)}


@router.get("/{session_id}")
async def session_detail(session_id: str, request: Request,
                         user: User = Depends(get_current_user)):
    """详情：大纲/章节/来源池/警告/引言结论/成稿（checkpoint 单一事实来源）。"""
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not session owner")

    graph = request.app.state.deep_research_graph
    snap = await aget_state_retry(graph, {"configurable": {"thread_id": session_id}})
    values = (snap.values or {}) if snap else {}
    return {
        "session_id": session_id, "topic": row.topic, "status": row.status,
        "roles": row.roles, "stats": row.stats, "mode": values.get("mode", "full"),
        "scouting_brief": values.get("scouting_brief", ""),
        "outline": values.get("outline") or {},
        "chapters": sorted(values.get("chapters") or [],
                           key=lambda c: c.get("index") or 0),
        "review_log": sorted(values.get("review_log") or [],
                             key=lambda r: (r.get("chapter") or 0,
                                            r.get("round") or 0)),
        "source_pool": values.get("source_pool", []),
        "warnings": values.get("warnings", []),
        "toc": values.get("toc", ""),
        "introduction": values.get("introduction", ""),
        "conclusion": values.get("conclusion", ""),
        "references": values.get("references", []),
        "final_report": values.get("final_report") or row.final_proposal,
        "token_budget_used": int(values.get("token_budget_used") or 0),
    }


# ---------- 追问：会话结束后向点名角色（或主编）继续提问 ----------

class DRAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


def _pick_agent(question: str) -> str:
    """按触发词选唤起角色：问题中**最早出现**的身份优先（同一位置取更长
    别名）。未命中任何身份 → 主编。"""
    from app.graph.prompts.deep_research import DR_QA_ALIASES
    best: tuple[int, str] | None = None
    for agent_id, names in DR_QA_ALIASES:
        for n in names:
            i = question.find(n)
            if i >= 0 and (best is None or i < best[0]
                           or (i == best[0] and len(n) > len(best[1]))):
                best = (i, agent_id)
    return best[1] if best else "chair"


@router.post("/{session_id}/ask")
async def dr_ask(session_id: str, req: DRAskRequest,
                 request: Request, user: User = Depends(get_current_user)):
    """会后追问（SSE）：点名角色作答，未点名由主编作答。

    只读回放 checkpoint，不改变会话状态；追问不计入本场预算。
    事件流：ask_start（agent/model）→ token → done / error。
    """
    set_agent_team("deep_research")   # 追问与主流程同一团队身份
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not session owner")

    graph = request.app.state.deep_research_graph
    ctx = request.app.state.workflow_ctx
    snap = await aget_state_retry(graph,
                                  {"configurable": {"thread_id": session_id}})
    values = dict((snap.values or {}) if snap else {})
    values.setdefault("user_id", user.id)   # _role_cfg 解析角色配置需要
    topic = values.get("topic") or row.topic
    roles = values.get("roles") or row.roles
    chapters = sorted(values.get("chapters") or [],
                      key=lambda c: c.get("index") or 0)
    report = values.get("final_report") or row.final_proposal

    agent_id = _pick_agent(req.question)
    from app.graph.prompts.deep_research import CHAIR_QA_SYSTEM, ROLE_ORIENTATION
    if agent_id == "chair":
        cfg = _effective_cfg(ctx, user.id)
        system = CHAIR_QA_SYSTEM
        role_name = "主编"
    else:
        cfg = _role_cfg(ctx, values, agent_id)
        role = next((r for r in roles if r.get("id") == agent_id), {})
        role_name = role.get("name", agent_id)
        system = (f"你是本场深度调研中的「{role_name}」。"
                  f"角色职责：{ROLE_ORIENTATION.get(agent_id, '')}\n"
                  f"本场课题：{topic}\n\n"
                  "现在用户向你追问。以你的角色身份、基于本场实际产出的内容"
                  "直接回答；与你的职责无关的如实说明，不要扮演其他角色。")

    chapters_text = "\n\n".join(
        f"## 第{c.get('index')}章 {c.get('title', '')}\n"
        f"{str(c.get('content', ''))[:1200]}" for c in chapters) or "（无章节）"
    system += (f"\n\n【全场参考】\n课题：{topic}\n\n"
               f"初步调研摘要：\n{str(values.get('scouting_brief') or '（无）')[:1500]}\n\n"
               f"各章正文：\n{chapters_text[:6000]}\n\n"
               f"结论：{str(values.get('conclusion') or '（无）')[:1500]}\n\n"
               f"成稿（节选）：{str(report)[:2000]}")
    temperature = 0.3
    msgs = [SystemMessage(content=system), HumanMessage(content=req.question)]
    sink: asyncio.Queue = asyncio.Queue()

    async def run_ask():
        set_event_sink(sink)
        try:
            await native_round(cfg, msgs, None, temperature, session_id,
                               token_event="token",
                               reasoning_event="ask_reasoning")
            await sink.put({"type": "__done__"})
        except Exception as e:
            logger.warning("deep_research ask failed: %s", e)
            await sink.put({"type": "__error__", "error": str(e)})
        finally:
            clear_event_sink()

    async def event_gen():
        task = asyncio.create_task(run_ask())
        try:
            yield _sse({"type": "ask_start", "agent": agent_id,
                        "name": role_name,
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


# ---------- 仅重跑执笔人（与 brainstorm/seminar rerun-writer 同构） ----------
#
# 溯源社的成稿 = write_frame（引言/结论/目录，LLM）+ publish（装配，纯代码）。
# 从 checkpoint 恢复章节/审稿记录/来源池，只重跑这两步；as_node 记为 publish
# （成稿链最后节点），详情回放即见新报告。

def _mark_running(session_id: str) -> None:
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
        if row is not None:
            row.status = "running"
            row.updated_at = utcnow()
            db.commit()


@router.post("/rerun-writer")
async def deep_research_rerun_writer(req: StopRequest, request: Request,
                                     user: User = Depends(get_current_user)):
    """SSE：只重跑报告撰写人 + 发布员。事件流与 /stream 相同（dr_ 前缀）。"""
    set_agent_team("deep_research")
    uid = user.id
    sid = req.session_id
    with SessionLocal() as db:
        row = db.get(BrainstormSession, sid)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != uid:
        raise HTTPException(status_code=403, detail="not session owner")
    if row.status == "running":
        raise HTTPException(status_code=409, detail="场次进行中，稍后再试")
    topic = row.topic

    graph = request.app.state.deep_research_graph
    config = {"configurable": {"thread_id": sid}}
    snap = await aget_state_retry(graph, config)
    values = dict((snap.values or {}) if snap else {})
    if not (values.get("chapters") or values.get("outline")):
        raise HTTPException(status_code=404, detail=
            "该场次没有可用的中间状态（未完成大纲/章节），请整场重跑")

    await run_in_threadpool(_mark_running, sid)
    sink: asyncio.Queue = asyncio.Queue()

    async def run_writer():
        set_event_sink(sink)
        clear_stop(sid)
        try:
            state = {**values, "stopped": False}
            ctx = request.app.state.workflow_ctx
            update_frame = await make_write_frame_node(ctx)(state)
            merged = {**values, **update_frame}
            update_pub = await publish_node(merged)      # 纯代码装配（引用去重）
            merged.update(update_pub)
            await run_in_threadpool(_finish_session, uid, sid, merged)
            if not merged.get("stopped"):
                await run_in_threadpool(_archive_report_to_kb, uid, sid,
                                        topic,
                                        str(merged.get("final_report") or ""))
            try:
                await graph.aupdate_state(
                    config, {**update_frame, **update_pub}, as_node="publish")
            except Exception as e:
                logger.warning("deep_research rerun-writer update_state: %s", e)
            await sink.put({"type": "__done__"})
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.shield(run_in_threadpool(
                    _fail_session, uid, sid, "client disconnected"))
            raise
        except Exception as e:
            logger.warning("deep_research rerun-writer failed: %s", e)
            await run_in_threadpool(_fail_session, uid, sid, str(e))
            await sink.put({"type": "__error__", "error": str(e)})
        finally:
            clear_event_sink()

    async def event_gen():
        task = asyncio.create_task(run_writer())
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

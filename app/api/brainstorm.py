"""头脑风暴 API：SSE 流式跑多 agent 子图 + 停止 + 会话列表/详情。

SSE 骨架与 /chat/stream 同构：run_graph 后台任务挂事件队列，
event_gen 持续 drain 推给前端。所有图内事件（bs_*）与 token 同一流出。
"""

import asyncio
import contextlib
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.chat import _ATT_MARK, _CHAT_FILES
from app.core.cancel import clear_stop, request_stop
from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.core.events import clear_event_sink, set_event_sink
from app.core.logging import get_logger
from app.graph.brainstorm import (DEBATER_IDS, ROLE_ORIENTATION,
                                  _evidence_digest, _positions_digest,
                                  _recent_transcript, _role_cfg,
                                  _effective_cfg)
from app.graph.native_llm import native_round
from app.models import BrainstormSession, KnowledgeBase, User, utcnow

logger = get_logger("brainstorm")

router = APIRouter(prefix="/api/v1/brainstorm", tags=["brainstorm"],
                   dependencies=[Depends(get_current_user)])


class BrainstormRequest(BaseModel):
    session_id: str = Field(min_length=1)     # 建议前端用 bs_ 前缀的 uuid
    topic: str = Field(min_length=1, max_length=8000)
    max_rounds: int | None = None            # None = 全局默认
    token_budget: int | None = Field(None, ge=10_000, le=100_000_000)  # 本场预算；None = 全局默认
    roles: list[dict] | None = None          # 可选：参赛角色子集（id 限四辩手，缺省=全部默认角色）
    attachments: list[str] = []               # /chat/files 上传的 file_id


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _default_roles(settings) -> list[dict]:
    """角色目录：settings 默认（可被 yaml 调整），过滤非法 id。"""
    roles = getattr(settings, "brainstorm_roles", None) or []
    return [{"id": r["id"], "name": r.get("name", r["id"]),
             "temperature": float(r.get("temperature", 0.5))}
            for r in roles if r.get("id") in DEBATER_IDS]


def _resolve_roles(raw: list[dict] | None, settings) -> list[dict]:
    """解析请求携带的角色配置：id 白名单校验（只认四辩手）、去重、
    温度夹取 0~2、名称截断。config_id 引用该用户已保存的 LLM 配置
    （每角色独立模型，合法性由图内 _role_cfg 解析时校验属主）。
    空/全非法 → 回退默认目录（角色可关不可造）。"""
    defaults = {r["id"]: r for r in _default_roles(settings)}
    if not raw:
        return list(defaults.values())
    out: list[dict] = []
    seen: set[str] = set()
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


def _initial_state(req: BrainstormRequest, user_id: str, settings,
                   roles: list[dict]) -> dict:
    """组装图初始状态：议题（附件全文拼入）+ 角色目录 + 控制字段。
    附件复用主链路 /chat/files 的进程内暂存（_CHAT_FILES）。"""
    topic = req.topic
    if req.attachments:
        blocks = []
        for fid in req.attachments:
            att = _CHAT_FILES.get(fid)
            if att:
                blocks.append(f"[附件：{att['filename']}]\n{att['text']}")
        if blocks:
            topic = topic + _ATT_MARK + "\n\n".join(blocks)
    return {
        "user_id": user_id,
        "session_id": req.session_id,
        "topic": topic,
        "roles": roles,
        "max_rounds": req.max_rounds
        or int(getattr(settings, "brainstorm_max_rounds", 3)),
        "token_budget": int(req.token_budget
                            or getattr(settings, "brainstorm_token_budget",
                                       20000000)),
        "turn_count": 0,
        "token_budget_used": 0,
        "positions": [],
        "evidence_pool": [],
        "transcript": [],
        "stopped": False,
    }


# ---------- 会话登记（与 chat._register_conversation 同构） ----------

def _register_session(user_id: str, req: BrainstormRequest, roles: list) -> None:
    with SessionLocal() as db:
        row = db.get(BrainstormSession, req.session_id)
        if row is None:
            db.add(BrainstormSession(
                session_id=req.session_id, user_id=user_id,
                topic=req.topic[:2000], status="running", roles=roles))
        else:                                  # 重跑同 session：复位状态
            row.status = "running"
            row.final_proposal = ""
            row.updated_at = utcnow()
        db.commit()


def _finish_session(user_id: str, session_id: str, result: dict) -> None:
    """收尾：status（done/stopped）+ 成稿 + stats 落库（best-effort）。"""
    stopped = bool(result.get("stopped"))
    stats = {"turns": len(result.get("transcript") or []),
             "positions": len(result.get("positions") or []),
             "evidence": len(result.get("evidence_pool") or []),
             "tokens": int(result.get("token_budget_used") or 0)}
    if result.get("budget_exhausted"):
        stats["budget_exhausted"] = True       # 硬熔断截断标注（设计文档 §7.7）
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


# ---------- 会话知识闭环（API 层机制，非 LLM 工具） ----------

BS_OUTPUT_KB_NAME = "头脑风暴成果"       # 用户专属沉淀库（自动创建）


def _archive_proposal_to_kb(user_id: str, session_id: str,
                             topic: str, proposal: str) -> None:
    """成稿入库：检索→（无则建）「头脑风暴成果」库 → add_documents。

    best-effort：失败只记日志（飞轮断一环不影响本次会话交付）。
    private scope：成果只对本人可见。入库是分块+嵌入的慢操作，故只在
    收尾（SSE 已推完 done 之前的一次线程池调用）执行且绝不阻塞图运行。
    """
    if not proposal.strip():
        return
    try:
        from app.main import app            # 拿 lifespan 装配的 KBService
        kb_service = app.state.kb_service
        with SessionLocal() as db:
            kb = db.scalar(select(KnowledgeBase).where(
                KnowledgeBase.name == BS_OUTPUT_KB_NAME,
                KnowledgeBase.owner_user_id == user_id))
            if kb is None:
                kb = kb_service.create_kb(
                    BS_OUTPUT_KB_NAME, "private", user_id,
                    description="头脑风暴自动沉淀：历场科研方案成稿")
            kb_id = kb.kb_id
        kb_service.add_documents(
            kb_id, [f"# {topic}\n\n{proposal}"])   # 标题入文，便于检索命中
    except Exception as e:
        logger.warning("brainstorm archive to kb failed %s: %s", session_id, e)


# ---------- SSE 流式主端点 ----------

@router.post("/stream")
async def brainstorm_stream(req: BrainstormRequest, request: Request,
                            user: User = Depends(get_current_user)):
    """SSE 流式：子图在后台任务里跑，端点持续 drain 事件队列。

    事件流（图内 emit，统一 bs_ 前缀与主链路隔离）：
    bs_prepare / bs_retrievals / bs_tool_start / bs_tool_end /
    bs_agent_start / bs_agent_end / bs_token / bs_reasoning /
    bs_moderator / bs_plan，最终 done。
    """
    uid = user.id
    graph = request.app.state.brainstorm_graph
    settings = request.app.state.settings
    sink: asyncio.Queue = asyncio.Queue()
    config = {"configurable": {"thread_id": req.session_id},
              "recursion_limit": 200}          # 辩论循环步数多，必须放大

    async def run_graph():
        set_event_sink(sink)
        clear_stop(req.session_id)
        try:
            # 重跑同 session：清旧 checkpoint，保证状态干净
            try:
                await graph.adelete_thread(req.session_id)
            except Exception:
                pass
            roles = _resolve_roles(req.roles, settings)
            await run_in_threadpool(_register_session, uid, req, roles)
            initial = _initial_state(req, uid, settings, roles)
            result = await graph.ainvoke(initial, config=config)
            await run_in_threadpool(_finish_session, uid, req.session_id, result)
            if not result.get("stopped"):    # 停止的半成品不入库（飞轮只沉淀完成品）
                await run_in_threadpool(_archive_proposal_to_kb, uid,
                                        req.session_id, req.topic,
                                        str(result.get("final_proposal") or ""))
            await sink.put({"type": "__done__"})
        except asyncio.CancelledError:
            # 客户端中途断开（SSE 连接中断）→ 本任务被取消：残留的 running 行
            # 复位 failed（与启动自愈同语义），否则列表永远"进行中"。
            # CancelledError 是 BaseException，上面的 except Exception 接不住；
            # shield 保证复位落库本身不被连环取消打断。
            with contextlib.suppress(Exception):
                await asyncio.shield(run_in_threadpool(
                    _fail_session, uid, req.session_id, "client disconnected"))
            raise
        except Exception as e:
            logger.warning("brainstorm run failed: %s", e)
            await run_in_threadpool(_fail_session, uid, req.session_id, str(e))
            await sink.put({"type": "__error__", "error": str(e)})
        finally:
            clear_event_sink()

    async def event_gen():
        task = asyncio.create_task(run_graph())
        try:
            while True:
                ev = await sink.get()            # 来一个发一个，实时
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


# ---------- 停止 / 列表 / 详情 ----------

class StopRequest(BaseModel):
    session_id: str


@router.post("/stop")
async def brainstorm_stop(req: StopRequest, user: User = Depends(get_current_user)):
    """终止自己的头脑风暴：只设标记，当前 agent 在下一个 token 边界退出，
    图跳 synthesis 以已有材料汇编成稿（部分成果保留）。"""
    with SessionLocal() as db:
        row = db.get(BrainstormSession, req.session_id)
        if row is not None and row.user_id != user.id:
            raise HTTPException(status_code=403, detail="not brainstorm owner")
    request_stop(req.session_id)
    return {"stopped": True, "session_id": req.session_id}


@router.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """当前用户的头脑风暴列表（最近活跃倒序）。"""
    uid = user.id

    def _list():
        with SessionLocal() as db:
            rows = db.scalars(select(BrainstormSession).where(
                BrainstormSession.user_id == uid)
                .order_by(BrainstormSession.updated_at.desc()).limit(200)).all()
            return [{"session_id": r.session_id, "topic": r.topic[:100],
                     "status": r.status, "stats": r.stats,
                     "updated_at": r.updated_at.isoformat()} for r in rows]
    return {"sessions": await run_in_threadpool(_list)}


@router.get("/{session_id}")
async def session_detail(session_id: str, request: Request,
                         user: User = Depends(get_current_user)):
    """详情：元数据 + 立场书/辩论记录/成稿（从 checkpoint 读，
    单一事实来源，不在表里重复存）。"""
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not brainstorm owner")

    graph = request.app.state.brainstorm_graph
    snap = await graph.aget_state({"configurable": {"thread_id": session_id}})
    values = (snap.values or {}) if snap else {}
    return {
        "session_id": session_id, "topic": row.topic, "status": row.status,
        "roles": row.roles, "stats": row.stats,
        "positions": values.get("positions", []),
        "transcript": values.get("transcript", []),
        "evidence_pool": values.get("evidence_pool", []),
        "final_proposal": values.get("final_proposal") or row.final_proposal,
    }


# ---------- 追问：会话结束后向点名角色（或主持人）继续提问 ----------

class BrainstormAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


# 触发词 → 角色别名表：问题文本中最早命中的角色被唤起；未命中 → 主持人
BS_QA_ALIASES = [
    ("innovator", ["创新者", "innovator"]),
    ("critic", ["批评者", "评论家", "critic"]),
    ("methodologist", ["方法论专家", "方法论", "methodologist"]),
    ("practitioner", ["实践者", "practitioner"]),
    ("writer", ["撰稿人", "撰稿", "writer"]),
    ("moderator", ["主持人", "moderator"]),
]


def _pick_agent(question: str, roles: list[dict]) -> str:
    """按触发词选唤起角色：问题中**最早出现**的身份优先（同一位置取更长别名）。
    未命中任何身份 → 主持人。"""
    best: tuple[int, str] | None = None
    for agent_id, names in BS_QA_ALIASES:
        for n in names:
            i = question.find(n)
            if i >= 0 and (best is None or i < best[0]
                           or (i == best[0] and len(n) > len(best[1]))):
                best = (i, agent_id)
    return best[1] if best else "moderator"


@router.post("/{session_id}/ask")
async def brainstorm_ask(session_id: str, req: BrainstormAskRequest,
                         request: Request,
                         user: User = Depends(get_current_user)):
    """会话追问（SSE）：点名角色以其人设+绑定模型作答，未点名由主持人作答。

    只读回放 checkpoint，不改变会话状态；追问不计入头脑风暴预算。
    事件流：ask_start（agent/model）→ token → done / error。
    """
    with SessionLocal() as db:
        row = db.get(BrainstormSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="not brainstorm owner")

    graph = request.app.state.brainstorm_graph
    ctx = request.app.state.workflow_ctx
    snap = await graph.aget_state({"configurable": {"thread_id": session_id}})
    values = dict((snap.values or {}) if snap else {})
    values.setdefault("user_id", user.id)   # _role_cfg 解析角色配置需要
    topic = values.get("topic") or row.topic
    positions = values.get("positions") or []
    notes = values.get("moderator_notes") or {}
    roles = values.get("roles") or row.roles
    proposal = values.get("final_proposal") or row.final_proposal

    agent_id = _pick_agent(req.question, roles)
    is_debater = agent_id in ("innovator", "critic",
                              "methodologist", "practitioner")
    role = next((r for r in roles if r.get("id") == agent_id), {}) \
        if is_debater else {}
    if agent_id == "moderator":
        cfg = _effective_cfg(ctx, user.id)
        system = ("你是本场头脑风暴的主持人，基于全场讨论客观回答用户追问；"
                  "引用观点请标明来自哪个角色，不清楚的如实说明。")
    elif agent_id == "writer":
        cfg = _effective_cfg(ctx, user.id)
        system = ("你是本场头脑风暴的撰稿人，最终方案由你执笔；"
                  "就方案内容与依据回答用户追问。")
    else:
        cfg = _role_cfg(ctx, values, agent_id)
        orientation = ROLE_ORIENTATION.get(agent_id, "")
        own = next((p.get("content", "") for p in positions
                    if p.get("agent_id") == agent_id), "（立场书缺失）")
        system = (f"你是本场头脑风暴中的「{role.get('name', agent_id)}」。"
                  f"角色取向：{orientation}\n本场议题：{topic}\n\n"
                  f"你的立场书：\n{own[:1200]}\n\n"
                  "现在用户向你追问。以你的角色身份、基于本场讨论与上述材料"
                  "直接回答；与你的立场有出入时如实说明分歧，不要扮演其他角色。")
    system += (f"\n\n【全场参考】\n议题：{topic}\n\n各角色立场书摘要：\n"
               f"{_positions_digest(values)}\n\n辩论记录（最近）：\n"
               f"{_recent_transcript(values, 20)}\n\n"
               f"共识：{json.dumps(notes.get('consensus', []), ensure_ascii=False)}\n"
               f"未决分歧：{json.dumps(notes.get('divergence', []), ensure_ascii=False)}\n"
               f"最终方案（节选）：{proposal[:2000]}")
    if agent_id == "moderator":
        system += "\n辩论记录（最近）与共识/分歧如上，回答请综合全场。"
    temperature = float(role.get("temperature", 0.3))
    msgs = [SystemMessage(content=system),
            HumanMessage(content=req.question)]
    sid = session_id
    uid = user.id
    sink: asyncio.Queue = asyncio.Queue()

    async def run_ask():
        set_event_sink(sink)
        try:
            resp, _usage, stopped = await native_round(
                cfg, msgs, None, temperature, sid,
                token_event="token", reasoning_event="ask_reasoning")
            await sink.put({"type": "__done__"})
        except Exception as e:
            logger.warning("brainstorm ask failed: %s", e)
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

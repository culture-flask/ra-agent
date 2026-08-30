"""头脑风暴子图：多 Agent 相互对话、集思广益产出科研方案。

拓扑（对应设计文档 §3/§5）：
  START → prepare ──(Send fan-out)──▶ research_{innovator|critic|methodologist|practitioner}
                                        │（四路并行，各自检索+调工具+写立场书）
                                        ▼
              ┌──────────────────── moderator（主持人调度）◀────┐
              │                        │                        │
              │  next_speaker=""       │ next_speaker="critic"   │
              ▼                        ▼                        │
           synthesis ──▶ END      agent_{role}（辩手发言）───────┘

纪律（与主链路一致）：工具失败→无工具辩论；单 agent 失败→缺席声明继续；
预算耗尽/用户停止→立即进 synthesis。绝不让 brainstorm 变 500。
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from app.core.cancel import clear_stop, is_stopped
from app.core.events import emit
from app.core.logging import get_logger
from app.graph.bs_state import BrainstormState
from app.graph.nodes import WorkflowContext, _loads_fuzzy
from app.graph.agent_runtime import (
    agent_speak as _agent_speak,
    effective_cfg as _effective_cfg,
    llm_text as _llm_text,
    role_cfg as _role_cfg,
    short_reason as _short_reason,
)
from app.graph.prompts.brainstorm import (
    DEBATE_SPEAK_SYSTEM,
    DEBATER_RESEARCH_SYSTEM,
    MODERATOR_PROMPT,
    PREPARE_PROMPT,
    ROLE_ORIENTATION,
    WRITER_PROMPT,
)

logger = get_logger("brainstorm")

DEBATER_IDS = ("innovator", "critic", "methodologist", "practitioner")


# ---------- 路由函数（纯函数，读 state 决定下一节点） ----------

def _debater_ids(state: BrainstormState) -> list[str]:
    """本场实际参赛的辩手 id（用户可关掉某角色）。"""
    ids = [r.get("id") for r in (state.get("roles") or [])
           if r.get("id") in DEBATER_IDS]
    return ids or list(DEBATER_IDS)


def route_research(state: BrainstormState) -> list[Send]:
    """prepare 之后：给每个辩手发一份**当前完整 state 的副本**，
    四路并行调研（Send = 动态并行 fan-out 的 LangGraph 原语）。"""
    return [Send(f"research_{rid}", state) for rid in _debater_ids(state)]


def route_after_moderator(state: BrainstormState) -> str:
    """moderator 之后：结束条件任一满足 → synthesis；否则去被点名的 agent。

    结束条件：用户停止 / next_speaker 为空或 finish（轮数上限、预算熔断、
    主持人判定收敛都会把 next_speaker 置空，见 moderator_node）。
    """
    if state.get("stopped"):
        return "synthesis"
    if state.get("next_speaker") in (None, "", "finish"):
        return "synthesis"
    return f"agent_{state['next_speaker']}"


async def build_brainstorm_graph(ctx: WorkflowContext):
    """编译头脑风暴子图。checkpointer 与主图各自持有独立连接
    （AsyncPostgresSaver 绑定事件循环，不能共享实例）。"""
    from app.graph.workflow import _build_checkpointer   # 复用主图的建 saver 逻辑

    builder = StateGraph(BrainstormState)

    async def _prepare(s): return await prepare_node(ctx, s)
    async def _moderator(s): return await moderator_node(ctx, s)
    async def _synthesis(s): return await synthesis_node(ctx, s)

    builder.add_node("prepare", _prepare)
    builder.add_node("moderator", _moderator)
    builder.add_node("synthesis", _synthesis)
    for rid in DEBATER_IDS:
        # 工厂在构建期绑定 ctx：LangGraph 调节点只传 state 一个参数，
        # 节点函数不能是 (ctx, state) 双参签名（与 _prepare 闭包同理）
        builder.add_node(f"research_{rid}", make_research_node(ctx, rid))
        builder.add_node(f"agent_{rid}", make_agent_node(ctx, rid))

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges(
        "prepare", route_research,
        {f"research_{rid}": f"research_{rid}" for rid in DEBATER_IDS})
    for rid in DEBATER_IDS:
        builder.add_edge(f"research_{rid}", "moderator")   # 并行汇合点
        builder.add_edge(f"agent_{rid}", "moderator")     # 辩论循环回主持人

    targets = {"synthesis": "synthesis"}
    for rid in DEBATER_IDS:
        targets[f"agent_{rid}"] = f"agent_{rid}"
    builder.add_conditional_edges("moderator", route_after_moderator, targets)

    builder.add_edge("synthesis", END)

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)


# ---------- 公共帮助：配置获取 / 控制类调用 / 流式发言 ----------

# ---------- Phase 0：议题解析 ----------

async def prepare_node(ctx: WorkflowContext, state: BrainstormState) -> dict:
    """主持人解析议题：重述 + 四角色差异化调研指令。

    解析失败（LLM 异常 / JSON 不合法）→ 降级为通用调研指令 + 角色取向兜底，
    绝不阻断头脑风暴（与主图 supervisor 的降级纪律一致）。
    """
    emit("bs_prepare", {"topic": state["topic"][:100]})
    restatement = state["topic"]
    directives: dict = {}
    used = 0
    try:
        text, used = await _llm_text(
            ctx, state, PREPARE_PROMPT.format(topic=state["topic"]),
            "请解析议题并输出 JSON。", temperature=0.2)
        data = _loads_fuzzy(text) or {}
        if data.get("topic_restatement"):
            restatement = str(data["topic_restatement"])
        if isinstance(data.get("research_directives"), dict):
            directives = data["research_directives"]
    except Exception as e:
        logger.warning("brainstorm prepare failed, fallback: %s", e)
    emit("bs_prepare", {"topic_restatement": restatement,
                        "directive_count": len(directives)})
    return {"topic_restatement": restatement,
            "research_directives": directives,
            "token_budget_used": used}


# ---------- Phase 1：并行独立调研（观点独立性关键） ----------

def make_research_node(ctx: WorkflowContext, role_id: str):
    """工厂：四个 research 节点共享实现，差异只来自 role_id 注入的
    指令/取向/温度。ctx 在图构建期绑定进闭包（LangGraph 调节点只传 state）。
    失败降级为"缺席声明"空立场书，绝不让单个 agent 挂掉整场头脑风暴。"""

    async def research_node(state: BrainstormState) -> dict:
        cfg = _role_cfg(ctx, state, role_id)   # 角色实际使用的模型
        emit("bs_agent_start", {"agent": role_id, "phase": "research",
                                "model": getattr(cfg, "model_id", "") or ""})
        role = next((r for r in state.get("roles", []) if r.get("id") == role_id),
                    {"id": role_id, "name": role_id, "temperature": 0.5})
        directive = (state.get("research_directives") or {}).get(role_id) \
            or "围绕议题做通用调研"
        max_chars = int(getattr(ctx.settings, "brainstorm_position_max_chars", 500))
        system = DEBATER_RESEARCH_SYSTEM.format(
            name=role.get("name", role_id),
            orientation=ROLE_ORIENTATION[role_id],
            directive=directive, max_chars=max_chars)

        # 1) 知识库直接检索：对每个可见库用「议题」搜 top3（混合模式），
        #    结果进共享证据池 + 拼进调研 prompt（不依赖模型自觉调工具）
        kb_lines: list[str] = []
        evidence: list[dict] = []
        try:
            kbs = await asyncio.to_thread(ctx.kb_service.list_queryable_kbs,
                                          state["user_id"])
            for kb in kbs:
                try:
                    hits = await asyncio.to_thread(
                        ctx.kb_service.search, kb.kb_id, state["topic"], k=3,
                        user_id=state["user_id"], mode="hybrid")
                except Exception as e:          # 单库隔离：坏库跳过
                    logger.warning("bs research kb search failed kb=%s: %s",
                                   kb.name, e)
                    continue
                for h in hits:
                    text = str(h.get("text", ""))[:800]
                    # 来源与主链路同款回退：Chroma 命中的 source 在 metadata 里
                    meta = h.get("metadata") or {}
                    src = h.get("source") or meta.get("source") or "未知来源"
                    if meta.get("page"):
                        src += f" 第{meta['page']}页"
                    kb_lines.append(f"[{kb.name} / {src}] {text}")
                    evidence.append({
                        "id": f"{role_id}-{len(evidence)}",
                        "found_by": role_id, "kb": kb.name,
                        "source": src, "digest": text[:120]})
            if evidence:
                emit("bs_retrievals", {"agent": role_id, "results": [
                    {"kb_name": e["kb"], "source": e["source"],
                     "text": e["digest"]} for e in evidence]})
        except Exception as e:
            logger.warning("bs research list kbs failed: %s", e)

        # 2) 工具子循环（联网搜索 / 学术检索 / 取原文……）
        tools = []
        if ctx.mcp_adapter is not None:
            tools = await ctx.mcp_adapter.schemas_for_llm() or []
        human = (f"议题：{state['topic']}\n\n"
                 + ("".join(f"[知识库检索结果] {l}\n" for l in kb_lines)
                    if kb_lines else "（知识库无相关命中）"))
        loop_max = int(getattr(ctx.settings, "brainstorm_research_tool_loop_max", 6))
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, role_id, system, human,
                float(role.get("temperature", 0.5)), tools, loop_max, cfg=cfg)
        except Exception as e:
            # 失败原因透出（模型不被端点支持/限流/网络……），界面与日志都可查
            logger.warning("bs research %s failed: %s", role_id, e)
            fail_reason = _short_reason(e)
            content, used, stopped = "", 0, False

        # 停止 ≠ 失败：用户主动停止时角色是"调研中止"，不能标成缺席失败
        if content.strip():
            body = content[:max_chars]
        elif stopped:
            body = "（已停止，本角色调研中止）"
        else:
            body = (f"（调研失败：{fail_reason}）" if fail_reason
                    else "(调研失败，本角色缺席)")
        position = {"agent_id": role_id, "agent_name": role.get("name", role_id),
                    "model": getattr(cfg, "model_id", "") or "",
                    "content": body,
                    "failed": not content.strip() and not stopped}
        emit("bs_agent_end", {"agent": role_id, "phase": "research",
                              "failed": position["failed"],
                              **({"reason": fail_reason} if fail_reason else {}),
                              **({"aborted": True}
                                 if stopped and not content.strip() else {})})
        # 只返回增量：positions/evidence_pool/token_budget_used 全是 add reducer。
        # ⚠️ 不能返回 stopped：它是单值 channel，四路并行写直接 InvalidUpdateError；
        # 停止语义由 is_stopped(sid) 注册表承载（moderator/synthesis 各自查表）。
        return {"positions": [position], "evidence_pool": evidence,
                "token_budget_used": used}

    return research_node


# ---------- Phase 2：主持人调度 ----------

async def moderator_node(ctx: WorkflowContext, state: BrainstormState) -> dict:
    """决定下一个发言者 + 判定收敛。终止优先级（高的先判）：
    用户停止 > token 预算熔断 > 轮数上限 > 主持人判定收敛。

    moderator 是辩论循环里唯一写 turn_count 的节点（单写者，默认覆盖即可）。
    主持人输出不合法（点到不存在的人）→ round-robin 顺序兜底。
    """
    sid = state["session_id"]
    if is_stopped(sid):                          # ① 用户停止：直接收束
        return {"stopped": True, "next_speaker": ""}

    ids = _debater_ids(state)
    turn = int(state.get("turn_count") or 0)
    max_turns = int(state.get("max_rounds") or 3) * len(ids)
    if turn >= max_turns:                        # ② 轮数上限
        emit("bs_moderator", {"reason": "max_rounds", "turns": turn})
        return {"next_speaker": ""}
    budget = int(getattr(ctx.settings, "brainstorm_token_budget", 800000))
    if int(state.get("token_budget_used") or 0) > budget:   # ③ 预算熔断
        emit("bs_moderator", {"reason": "budget_exhausted", "turns": turn})
        return {"next_speaker": ""}

    # ④ 主持人 LLM 判定
    notes = state.get("moderator_notes") or {}
    window = int(getattr(ctx.settings, "brainstorm_transcript_window", 6))
    prompt = MODERATOR_PROMPT.format(
        topic=state["topic"],
        positions=_positions_digest(state),
        recent=_recent_transcript(state, window),
        consensus=json.dumps(notes.get("consensus", []), ensure_ascii=False),
        divergence=json.dumps(notes.get("divergence", []), ensure_ascii=False))
    next_speaker, focus, used = "", "", 0
    try:
        text, used = await _llm_text(ctx, state, prompt,
                                     "请调度并输出 JSON。", temperature=0.2)
        data = _loads_fuzzy(text) or {}
        cand = str(data.get("next_speaker", "")).strip().lower()
        # finish 与收敛判定同门槛：至少辩满一轮才允许提前收场——否则主持人
        # 开场看到立场书互相一致就直接 finish，辩论零发言（真实事故复现）
        if cand in ids or (cand == "finish" and turn >= len(ids)):
            next_speaker = cand
        focus = str(data.get("focus", ""))[:200]
        notes = {"consensus": data.get("consensus") or [],
                 "divergence": data.get("divergence") or [],
                 "consensus_level": int(data.get("consensus_level") or 0),
                 "focus": focus}
    except Exception as e:
        logger.warning("bs moderator failed, round-robin: %s", e)

    if not next_speaker:                         # 降级：round-robin 顺序点名
        next_speaker = ids[turn % len(ids)]

    # 收敛判定：至少辩满一轮 + 主持人共识度 ≥7 且无分歧 → 提前结束
    level = int(notes.get("consensus_level") or 0)
    if turn >= len(ids) and level >= 7 and not notes.get("divergence"):
        emit("bs_moderator", {"reason": "converged", "consensus_level": level})
        return {"next_speaker": "", "moderator_notes": notes,
                "token_budget_used": used}

    emit("bs_moderator", {"next_speaker": next_speaker, "focus": focus,
                          "turn": turn + 1, "consensus_level": level,
                          "consensus": notes.get("consensus", []),
                          "divergence": notes.get("divergence", [])})
    return {"next_speaker": next_speaker, "moderator_notes": notes,
            "turn_count": turn + 1, "token_budget_used": used}


# ---------- Phase 2：辩手发言 ----------

def make_agent_node(ctx: WorkflowContext, role_id: str):
    """工厂：四个 agent 节点共享实现，ctx 在图构建期绑定进闭包。
    发言上下文 = 角色取向 + 全部立场书
    摘要 + 最近 K 轮发言原文 + 共享证据池摘要（滚动窗口控窗口占用，
    更早内容由 moderator_notes 的共识/分歧承载——等价滚动摘要）。"""

    async def agent_node(state: BrainstormState) -> dict:
        if is_stopped(state["session_id"]):
            return {"stopped": True, "next_speaker": ""}
        role = next((r for r in state.get("roles", []) if r.get("id") == role_id),
                    {"id": role_id, "name": role_id, "temperature": 0.5})
        turn = int(state.get("turn_count") or 0)
        focus = (state.get("moderator_notes") or {}).get("focus", "自由发言")
        cfg = _role_cfg(ctx, state, role_id)
        emit("bs_agent_start", {"agent": role_id, "phase": "debate",
                                "turn": turn, "focus": focus,
                                "model": getattr(cfg, "model_id", "") or ""})

        # 工具只在辩论前几轮开放（防无限调研不辩论）
        tools = []
        if turn <= int(getattr(ctx.settings, "brainstorm_debate_tool_rounds", 1)) \
                and ctx.mcp_adapter is not None:
            tools = await ctx.mcp_adapter.schemas_for_llm() or []

        window = int(getattr(ctx.settings, "brainstorm_transcript_window", 6))
        system = DEBATE_SPEAK_SYSTEM.format(
            name=role.get("name", role_id),
            orientation=ROLE_ORIENTATION[role_id], focus=focus)
        human = (f"议题：{state['topic']}\n\n"
                 f"【各方立场书】\n{_positions_digest(state)}\n\n"
                 f"【最近的发言】\n{_recent_transcript(state, window)}\n\n"
                 f"【共享证据池】\n{_evidence_digest(state)}\n\n"
                 f"请发言（记得用 @[角色名] 回应他人）。")
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, role_id, system, human,
                float(role.get("temperature", 0.5)), tools, tool_loop_max=2,
                cfg=cfg)
        except Exception as e:
            logger.warning("bs debate %s failed: %s", role_id, e)
            fail_reason = _short_reason(e)
            content = f"（本轮发言失败：{fail_reason}）"
            used, stopped = 0, False

        entry = {"turn": turn, "agent_id": role_id,
                 "agent_name": role.get("name", role_id),
                 "model": getattr(cfg, "model_id", "") or "",
                 "content": content[:1000]}
        emit("bs_agent_end", {"agent": role_id, "phase": "debate",
                              **({"reason": fail_reason} if fail_reason else {})})
        return {"transcript": [entry], "token_budget_used": used,
                "stopped": stopped, "next_speaker": ""}

    return agent_node


# ---------- 上下文摘要（控窗口占用的三把刀） ----------

def _positions_full(state: BrainstormState) -> str:
    """立场书全文（撰稿人专用：立场书本就 ≤500 字/份，全文注入成本可忽略，
    却能让撰稿人拿到完整论证而非 300 字摘要）。"""
    parts = []
    for p in state.get("positions") or []:
        flag = "（调研失败）" if p.get("failed") else ""
        parts.append(f"### {p.get('agent_name')} {flag}\n{p.get('content', '')}")
    return "\n\n".join(parts) or "（无）"


def _positions_digest(state: BrainstormState) -> str:
    """立场书摘要：每份截断 300 字（完整版只在 writer 成稿时用）。"""
    parts = []
    for p in state.get("positions") or []:
        flag = "（调研失败）" if p.get("failed") else ""
        parts.append(f"### {p.get('agent_name')} {flag}\n"
                     f"{str(p.get('content', ''))[:300]}")
    return "\n\n".join(parts) or "（无）"


def _recent_transcript(state: BrainstormState, window: int) -> str:
    """最近 window 条发言原文（更早的靠 moderator_notes 承载）。"""
    items = (state.get("transcript") or [])[-window:]
    if not items:
        return "（辩论尚未开始，这是第一轮）"
    return "\n\n".join(f"[{t.get('agent_name')}] {t.get('content', '')[:400]}"
                       for t in items)


def _evidence_digest(state: BrainstormState, limit: int = 15) -> str:
    """证据池摘要：每条 digest ≤120 字，超出取前 limit 条。"""
    items = (state.get("evidence_pool") or [])[:limit]
    if not items:
        return "（无证据）"
    return "\n".join(f"- [{e.get('found_by')}发现 / {e.get('kb')} / "
                     f"{e.get('source')}] {e.get('digest', '')}" for e in items)


# ---------- Phase 4：撰稿成稿 ----------

def _fallback_proposal(state: BrainstormState) -> str:
    """降级成稿：writer 失败/用户停止时，直接汇编立场书 + 分歧清单。
    结构化中间产物也好过空手而归。"""
    notes = state.get("moderator_notes") or {}
    parts = [f"# 研究方案（汇编版）：{state['topic']}",
             "", "> 注：撰稿人生成失败或中途停止，以下为各方立场与分歧的"
             "结构化汇编。", "", "## 各方立场书"]
    for p in state.get("positions") or []:
        parts.append(f"### {p.get('agent_name')}\n{p.get('content', '')}")
    if notes.get("consensus"):
        parts.append("## 已达成共识\n" + "\n".join(
            f"- {c}" for c in notes["consensus"]))
    if notes.get("divergence"):
        parts.append("## 未决分歧\n" + "\n".join(
            f"- {d}" for d in notes["divergence"]))
    return "\n\n".join(parts)


async def synthesis_node(ctx: WorkflowContext, state: BrainstormState) -> dict:
    """撰稿人成稿。用户停止 → 不再调 LLM，直接汇编降级稿（部分成果保留）。

    token 预算熔断只停止辩论（moderator 不再点名），撰稿人**不受预算约束、
    必须执行**——撰稿成本相对调研+辩论可忽略，若被熔断跳过，前面讨论的
    全部投入就浪费了。预算超限仅在 stats 标注截断（budget_exhausted）。
    writer 失败 → _fallback_proposal（status 标 failed 由 API 层落库）。"""
    sid = state["session_id"]
    stopped = is_stopped(sid)
    budget = int(state.get("token_budget")
                 or getattr(ctx.settings, "brainstorm_token_budget", 20000000))
    over_budget = int(state.get("token_budget_used") or 0) > budget
    if stopped:                             # 仅用户停止跳过撰稿
        clear_stop(sid)
        content = _fallback_proposal(state)
        emit("bs_plan", {"content": content, "fallback": True,
                         "reason": "stopped"})
        return {"final_proposal": content, "stopped": True,
                **({"budget_exhausted": True} if over_budget else {})}

    wcfg = _effective_cfg(ctx, state["user_id"])
    emit("bs_agent_start", {"agent": "writer", "phase": "synthesis",
                            "model": getattr(wcfg, "model_id", "") or ""})
    notes = state.get("moderator_notes") or {}
    # 辩论全文给 writer：每条截断 600 字、最多 20 条（分段摘要的简化版）
    transcript = "\n\n".join(
        f"[{t.get('agent_name')}] {t.get('content', '')[:600]}"
        for t in (state.get("transcript") or [])[-20:]) or "（无辩论记录）"
    system = WRITER_PROMPT.format(
        topic=state["topic"],
        positions=_positions_full(state),
        transcript=transcript,
        consensus=json.dumps(notes.get("consensus", []), ensure_ascii=False),
        divergence=json.dumps(notes.get("divergence", []), ensure_ascii=False),
        evidence=_evidence_digest(state, limit=30))
    # 撰稿人专属工具子集：保存文档/导出引文（save_document/export_bibtex
    # 是 writer 的"手"，检索类工具对成稿无益反而引入跑题风险）。
    # schemas_for_llm 是混合格式：MCP 工具裸 {"name":..}，原生工具包
    # {"type":"function","function":{"name":..}}——两种形态都要取到 name。
    def _schema_name(t: dict) -> str:
        return t.get("name") or (t.get("function") or {}).get("name") or ""

    doc_tools = []
    if ctx.mcp_adapter is not None:
        doc_tools = [t for t in (await ctx.mcp_adapter.schemas_for_llm() or [])
                     if _schema_name(t) in ("save_document", "export_bibtex")]
    try:
        content, used, _ = await _agent_speak(
            ctx, state, "writer", system,
            "请把全场讨论收敛为一份唯一、明确、可执行的研究方案（不是观点综述）；"
            "完成后如需保存草稿或导出引文可调用文档工具。", 0.4, tools=doc_tools,
            tool_loop_max=3, cfg=wcfg)
        if not content.strip():
            raise RuntimeError("empty proposal")
    except Exception as e:
        logger.warning("bs writer failed, fallback digest: %s", e)
        content, used = _fallback_proposal(state), 0
    emit("bs_plan", {"content": content})
    out = {"final_proposal": content, "token_budget_used": used}
    if over_budget:
        out["budget_exhausted"] = True      # 辩论被熔断截断的标注（撰稿已照常执行）
    return out

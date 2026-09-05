"""格致会讲子图：学术研讨式多 agent 协作，产出研究构想组合。

拓扑（对应设计文档 §2/§4）：
  START → curate ──(Send×4)──▶ read_{hist|theo|exp|visit}   四路并行研读
                                   │（汇合）
                                   ▼
             ┌──────────── present（议程循环：agenda_index 推进）◀──┐
             │                     │                                │
             │                     ▼                                │
             │                qa（全员承接式问答 + 主席采集洞见）────┘
             │ 议程完成（Send×4）
             ▼
        propose_{rid} ─▶ merge（主席定稿）─(Send×4)─▶ improve_{rid}
                                   │（汇合）
                                   ▼
                              panel（公布参评卡）─(Send×4)─▶ score_{rid}
                                   │（汇合）
                                   ▼
                          aggregate（纯代码聚合）──▶ rapporteur ──▶ END

与争鸣社的根本差异：无 LLM 调度——循环由 agenda_index 驱动，
终止条件（停止/预算/议程完）在 route_after_qa 的代码里判定。

纪律（与争鸣社一致）：工具失败→无工具研讨；单学者失败→缺席声明继续；
预算熔断只截断后续阶段、rapporteur 不受预算阻断；并行节点绝不写单值 channel。
"""

import asyncio
import json
import re

from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from app.core.cancel import clear_stop, is_stopped
from app.core.events import emit
from app.core.logging import get_logger
from app.graph.agent_runtime import (
    agent_speak as _agent_speak,
    effective_cfg as _effective_cfg,
    llm_text as _llm_text,
    role_cfg as _role_cfg,
    short_reason as _short_reason,
)
from app.graph.nodes import WorkflowContext, _loads_fuzzy
from app.graph.prompts.seminar import (
    ANSWER_SYSTEM, ASK_SYSTEM, CHAIR_INSIGHT_PROMPT, CURATE_PROMPT,
    IMPROVE_SYSTEM, MERGE_PROMPT, NOTE_TEMPLATES, PRESENT_SYSTEM,
    PROPOSE_SYSTEM, RAPPORTEUR_PROMPT, READING_SYSTEM, SCHOLAR_ORIENTATION,
    SCORE_SYSTEM,
)
from app.graph.sem_state import SeminarState

logger = get_logger("seminar")

SCHOLAR_IDS = ("historian", "theorist", "experimentalist", "visitor")


# ---------- 路由函数（纯代码，无 LLM——本团队的核心特性） ----------

def _scholar_ids(state: SeminarState) -> list[str]:
    """本场实际与会的学者 id（用户可关掉某角色）。"""
    ids = [r.get("id") for r in (state.get("roles") or [])
           if r.get("id") in SCHOLAR_IDS]
    return ids or list(SCHOLAR_IDS)


def route_read(state: SeminarState) -> list[Send]:
    """curate 之后：四路并行研读。"""
    return [Send(f"read_{rid}", state) for rid in _scholar_ids(state)]


def route_after_qa(state: SeminarState):
    """qa 之后的三岔路口（对照争鸣社 moderator 的四级终止，这里无收敛判定）：
    用户停止/预算熔断 → rapporteur（汇编降级稿）；
    议程未完 → present（下一位学者）；
    议程已完 → Send fan-out 进构想工作坊。
    """
    if state.get("stopped"):
        return "rapporteur"
    budget = int(state.get("token_budget") or 0)   # 缺省大值=不熔断（API 层已注入）
    if budget and int(state.get("token_budget_used") or 0) > budget:
        return "rapporteur"
    if state.get("agenda_index", 0) >= len(_scholar_ids(state)):
        return [Send(f"propose_{rid}", state) for rid in _scholar_ids(state)]
    return "present"


def route_after_merge(state: SeminarState):
    """merge 之后：无卡可改进（全部提案失败）→ 直接评不了，去 rapporteur。"""
    if not state.get("card_registry"):
        return "rapporteur"
    return [Send(f"improve_{rid}", state) for rid in _scholar_ids(state)]


def route_after_panel(state: SeminarState):
    """panel 之后：无卡参评 → 直接成稿。"""
    if not state.get("card_registry"):
        return "rapporteur"
    return [Send(f"score_{rid}", state) for rid in _scholar_ids(state)]


async def build_seminar_graph(ctx: WorkflowContext):
    """编译会讲子图（独立 checkpointer，与主图/争鸣社各自持有连接）。"""
    from app.graph.workflow import _build_checkpointer

    builder = StateGraph(SeminarState)

    builder.add_node("curate", make_curate_node(ctx))
    builder.add_node("present", make_present_node(ctx))
    builder.add_node("qa", make_qa_node(ctx))
    builder.add_node("merge", make_merge_node(ctx))
    builder.add_node("panel", make_panel_node(ctx))
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("rapporteur", make_rapporteur_node(ctx))
    for rid in SCHOLAR_IDS:
        # 工厂在构建期闭包绑定 ctx
        builder.add_node(f"read_{rid}", make_read_node(ctx, rid))
        builder.add_node(f"propose_{rid}", make_propose_node(ctx, rid))
        builder.add_node(f"improve_{rid}", make_improve_node(ctx, rid))
        builder.add_node(f"score_{rid}", make_score_node(ctx, rid))

    builder.add_edge(START, "curate")
    builder.add_conditional_edges(
        "curate", route_read,
        {f"read_{rid}": f"read_{rid}" for rid in SCHOLAR_IDS})
    for rid in SCHOLAR_IDS:
        builder.add_edge(f"read_{rid}", "present")      # 四路汇合进议程

    builder.add_edge("present", "qa")                   # 议程循环体
    qa_targets = {"present": "present", "rapporteur": "rapporteur"}
    for rid in SCHOLAR_IDS:
        qa_targets[f"propose_{rid}"] = f"propose_{rid}"
    builder.add_conditional_edges("qa", route_after_qa, qa_targets)

    for rid in SCHOLAR_IDS:
        builder.add_edge(f"propose_{rid}", "merge")
    merge_targets = {"rapporteur": "rapporteur"}
    for rid in SCHOLAR_IDS:
        merge_targets[f"improve_{rid}"] = f"improve_{rid}"
    builder.add_conditional_edges("merge", route_after_merge, merge_targets)

    for rid in SCHOLAR_IDS:
        builder.add_edge(f"improve_{rid}", "panel")
    panel_targets = {"rapporteur": "rapporteur"}
    for rid in SCHOLAR_IDS:
        panel_targets[f"score_{rid}"] = f"score_{rid}"
    builder.add_conditional_edges("panel", route_after_panel, panel_targets)

    for rid in SCHOLAR_IDS:
        builder.add_edge(f"score_{rid}", "aggregate")
    builder.add_edge("aggregate", "rapporteur")
    builder.add_edge("rapporteur", END)

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)


# ---------- 上下文摘要（控窗口占用，争鸣社三把刀同款） ----------

def _loads_list(text: str) -> list | None:
    """容错解析 JSON 数组（propose/improve/score 的输出形态）。

    ⚠️ 不能用 _loads_fuzzy——它是 dict-only（isinstance dict else None），
    会把 JSON 数组静默丢成 None（实测：构想卡与评分全部消失）。
    """
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None


def _notes_digest(state: SeminarState, exclude: str = "",
                  full_for: str = "", limit: int = 300) -> str:
    """研读笔记摘要：full_for 的笔记全文，其余各截 limit 字。"""
    parts = []
    for n in state.get("reading_notes") or []:
        rid = n.get("agent_id", "")
        if rid == exclude:
            continue
        flag = "（研读失败）" if n.get("failed") else ""
        body = (n.get("content", "") if rid == full_for
                else n.get("content", "")[:limit])
        parts.append(f"### {n.get('agent_name')} {flag}\n{body}")
    return "\n\n".join(parts) or "（无）"


def _insight_digest(state: SeminarState, window: int = 20) -> str:
    """洞见池摘要（只增不减的领域认知资产，注入各阶段上下文）。"""
    items = (state.get("insight_board") or [])[-window:]
    if not items:
        return "（暂无）"
    return "\n".join(f"- [{i.get('kind')}] {i.get('statement')}"
                     f"（{i.get('origin')}）" for i in items)


def _cards_digest(cards: list[dict], limit: int = 500) -> str:
    """构想卡摘要：每卡各字段截 limit 字。

    存库上限（panel/merge/improve 解析时截断）：标题 120、灵感 300、
    假设 500、风险 300、**验证思路 5500**——limit < 5500 时验证思路
    展示不全，消费方按需选择：评审 2000（可行性打分要见验证深度）、
    执笔 5500（全文）。"""
    if not cards:
        return "（无）"
    out = []
    for c in cards:
        builds = f"（改进自 {c['builds_on']}）" if c.get("builds_on") else ""
        out.append(f"[{c.get('card_id', '?')}] {c.get('author', '?')}{builds}："
                   f"{c.get('title', '')}\n"
                   f"假设：{str(c.get('hypothesis', ''))[:limit]}\n"
                   f"验证：{str(c.get('validation_sketch', ''))[:limit]}\n"
                   f"风险：{str(c.get('risk', ''))[:limit]}")
    return "\n\n".join(out)


def _role_map(state: SeminarState) -> dict:
    """role_id -> 角色信息（名称/温度），缺省兜底。"""
    out = {}
    for r in state.get("roles") or []:
        if r.get("id") in SCHOLAR_IDS:
            out[r["id"]] = r
    for rid in SCHOLAR_IDS:
        out.setdefault(rid, {"id": rid, "name": rid, "temperature": 0.5})
    return out


# ---------- Phase 0：开题 ----------

def make_curate_node(ctx: WorkflowContext):
    async def curate_node(state: SeminarState) -> dict:
        """主席规划全场：领域定位 + 邻近领域选定 + 双重差异化任务单。

        解析失败 → 通用任务单 + 角色默认取向。
        """
        emit("sem_curate", {"topic": state["topic"][:100]})
        positioning, adjacent = "", ""
        assignments: dict = {}
        directives: dict = {}
        used = 0
        try:
            text, used = await _llm_text(
                ctx, state,
                CURATE_PROMPT.format(topic=state["topic"]),
                "请规划议程并输出 JSON。", temperature=0.2, prefix="sem_")
            data = _loads_fuzzy(text) or {}
            positioning = str(data.get("field_positioning", ""))[:500]
            adjacent = str(data.get("adjacent_field", ""))[:300]
            if isinstance(data.get("reading_assignments"), dict):
                assignments = data["reading_assignments"]
            if isinstance(data.get("ideation_directives"), dict):
                directives = data["ideation_directives"]
        except Exception as e:
            logger.warning("seminar curate failed, fallback: %s", e)
        emit("sem_curate", {"field_positioning": positioning,
                            "adjacent_field": adjacent,
                            "assignment_count": len(assignments)})
        return {"field_positioning": positioning, "adjacent_field": adjacent,
                "reading_assignments": assignments,
                "ideation_directives": directives,
                "token_budget_used": used}
    return curate_node


# ---------- Phase 1：并行独立研读 ----------

def make_read_node(ctx: WorkflowContext, role_id: str):
    async def read_node(state: SeminarState) -> dict:
        cfg = _role_cfg(ctx, state, role_id)
        emit("sem_agent_start", {"agent": role_id, "phase": "read",
                                 "model": getattr(cfg, "model_id", "") or ""})
        rmap = _role_map(state)
        role = rmap[role_id]
        assignment = (state.get("reading_assignments") or {}).get(role_id) \
            or "围绕议题做通用研读"
        # 访问学者：主席选定的邻近领域写进任务单（跨领域创新的锚点）
        if role_id == "visitor" and state.get("adjacent_field"):
            assignment = f"主席建议的源领域：{state['adjacent_field']}\n{assignment}"
        max_chars = int(getattr(ctx.settings, "seminar_notes_max_chars", 600))
        system = READING_SYSTEM.format(
            name=role["name"], role_id=role_id,
            orientation=SCHOLAR_ORIENTATION[role_id],
            assignment=assignment, max_chars=max_chars,
            note_template=NOTE_TEMPLATES[role_id])

        # 1) 知识库直检（同争鸣社 research：不依赖模型自觉调工具）
        kb_lines: list[str] = []
        evidence: list[dict] = []
        try:
            kbs = await asyncio.to_thread(ctx.kb_service.list_queryable_kbs,
                                          state["user_id"], team="seminar")
            for kb in kbs:
                try:
                    hits = await asyncio.to_thread(
                        ctx.kb_service.search, kb.kb_id, state["topic"], k=15,
                        user_id=state["user_id"], mode="hybrid")
                except Exception as e:              # 单库隔离
                    logger.warning("sem read kb search failed kb=%s: %s",
                                   kb.name, e)
                    continue
                for h in hits:
                    text = str(h.get("text", ""))
                    meta = h.get("metadata") or {}
                    src = h.get("source") or meta.get("source") or "未知来源"
                    if meta.get("page"):
                        src += f" 第{meta['page']}页"
                    kb_lines.append(f"[{kb.name} / {src}] {text}")
                    evidence.append({"id": f"{role_id}-{len(evidence)}",
                                     "found_by": role_id, "kb": kb.name,
                                     "source": src, "digest": text[:400]})
            if evidence:
                emit("sem_retrievals", {"agent": role_id, "results": [
                    {"kb_name": e["kb"], "source": e["source"],
                     "text": e["digest"]} for e in evidence]})
        except Exception as e:
            logger.warning("sem read list kbs failed: %s", e)

        # 2) 工具子循环（联网/学术检索/引文追溯/知识库主动检索……）
        tools = []
        if ctx.mcp_adapter is not None:
            tools = await ctx.mcp_adapter.schemas_for_llm() or []
        human = (f"议题：{state['topic']}\n\n"
                 + ("".join(f"[知识库检索结果] {l}\n" for l in kb_lines)
                    if kb_lines else "（知识库无相关命中）"))
        loop_max = int(getattr(ctx.settings, "seminar_reading_tool_loop_max", 6))
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, role_id, system, human,
                float(role.get("temperature", 0.5)), tools, loop_max,
                cfg=cfg, prefix="sem_")
        except Exception as e:
            logger.warning("sem read %s failed: %s", role_id, e)
            fail_reason = _short_reason(e)
            content, used, stopped = "", 0, False

        # 停止 ≠ 失败
        if content.strip():
            body = content[:max_chars]
        elif stopped:
            body = "（已停止，本学者研读中止）"
        else:
            body = (f"（研读失败：{fail_reason}）" if fail_reason
                    else "（研读失败，本学者缺席）")
        note = {"agent_id": role_id, "agent_name": role["name"],
                "model": getattr(cfg, "model_id", "") or "",
                "content": body,
                "failed": not content.strip() and not stopped}
        emit("sem_agent_end", {"agent": role_id, "phase": "read",
                               "failed": note["failed"],
                               **({"reason": fail_reason}
                                  if fail_reason else {})})
        return {"reading_notes": [note], "evidence_pool": evidence,
                "token_budget_used": used}
    return read_node


# ---------- Phase 2：议程循环（汇报 → 全员问答 → 洞见采集） ----------

def make_present_node(ctx: WorkflowContext):
    async def present_node(state: SeminarState) -> dict:
        """当前议程学者汇报。agenda_index 单写者递增（本节点）。"""
        if is_stopped(state["session_id"]):
            return {"stopped": True}
        scholars = _scholar_ids(state)
        idx = int(state.get("agenda_index") or 0)
        if idx >= len(scholars):                 # 防御：议程已完不该进来
            return {}
        rid = scholars[idx]
        rmap = _role_map(state)
        role = rmap[rid]
        cfg = _role_cfg(ctx, state, rid)
        emit("sem_agent_start", {"agent": rid, "phase": "present",
                                 "index": idx,
                                 "model": getattr(cfg, "model_id", "") or ""})
        own = next((n for n in state.get("reading_notes") or []
                    if n.get("agent_id") == rid), {})
        system = PRESENT_SYSTEM.format(
            name=role["name"], role_id=rid,
            orientation=SCHOLAR_ORIENTATION[rid])
        human = (f"议题：{state['topic']}\n\n"
                 f"【你的研读笔记】\n{own.get('content', '（研读失败）')}\n\n"
                 f"【其他学者的笔记摘要（供你了解全场进度）】\n"
                 f"{_notes_digest(state, full_for=rid)}\n\n"
                 "请开始你的汇报。")
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, rid, system, human,
                float(role.get("temperature", 0.5)), None, 1,
                cfg=cfg, prefix="sem_")          # 汇报用已研读的材料，不再调工具
        except Exception as e:
            logger.warning("sem present %s failed: %s", rid, e)
            fail_reason = _short_reason(e)
            content = f"（汇报失败：{fail_reason}）"
            used, stopped = 0, False
        entry = {"kind": "present", "agent_id": rid,
                 "agent_name": role["name"],
                 "model": getattr(cfg, "model_id", "") or "",
                 "content": content[:1500]}
        emit("sem_present_end", {"agent": rid})
        return {"qa_transcript": [entry], "agenda_index": idx + 1,
                "token_budget_used": used,
                **({"stopped": True} if stopped else {})}
    return present_node


def make_qa_node(ctx: WorkflowContext):
    async def qa_node(state: SeminarState) -> dict:
        """全员承接式问答 + 主席采集洞见。

        节点内部循环（非图级循环）：askers 逐个提问、汇报人逐一回应——
        每组问答就是一次多 agent 直接相互对话，SSE 逐 token 可见。
        """
        sid = state["session_id"]
        if is_stopped(sid):
            return {"stopped": True}
        scholars = _scholar_ids(state)
        idx = int(state.get("agenda_index") or 0) - 1   # 刚汇报的那位
        if idx < 0 or idx >= len(scholars):
            return {}
        presenter = scholars[idx]
        rmap = _role_map(state)
        prole = rmap[presenter]
        entries: list[dict] = []
        used_total = 0
        presentation = next((e.get("content", "") for e in
                             reversed(state.get("qa_transcript") or [])
                             if e.get("kind") == "present"
                             and e.get("agent_id") == presenter), "")

        for asker in scholars:
            if asker == presenter:
                continue
            if is_stopped(sid):
                return {"qa_transcript": entries, "stopped": True}
            arole = rmap[asker]
            acfg = _role_cfg(ctx, state, asker)
            emit("sem_agent_start", {"agent": asker, "phase": "qa",
                                     "target": presenter,
                                     "model": getattr(acfg, "model_id", "") or ""})
            own_note = next((n.get("content", "") for n in
                             state.get("reading_notes") or []
                             if n.get("agent_id") == asker), "")
            ask_sys = ASK_SYSTEM.format(
                name=arole["name"], role_id=asker,
                orientation=SCHOLAR_ORIENTATION[asker],
                presenter=prole["name"])
            ask_human = (f"{prole['name']}的汇报：\n{presentation[:1500]}\n\n"
                         f"你的研读笔记：\n{own_note}\n\n"
                         f"已提出的洞见池：\n{_insight_digest(state, 20)}\n\n"
                         "请提出你的问题。")
            try:
                question, used, stopped = await _agent_speak(
                    ctx, state, asker, ask_sys, ask_human,
                    float(arole.get("temperature", 0.5)), None, 1,
                    cfg=acfg, prefix="sem_")
            except Exception as e:
                logger.warning("sem ask %s failed: %s", asker, e)
                question, used, stopped = (
                    f"（提问失败：{_short_reason(e)}）", 0, False)
            used_total += used
            entries.append({"kind": "ask", "agent_id": asker,
                            "agent_name": arole["name"],
                            "target": presenter, "content": question[:500]})
            if stopped:
                return {"qa_transcript": entries, "stopped": True,
                        "token_budget_used": used_total}

            # 汇报人回应（新开气泡：每组问答一次多 agent 直接对话，逐 token 可见）
            pcfg = _role_cfg(ctx, state, presenter)
            emit("sem_agent_start", {"agent": presenter, "phase": "qa",
                                     "target": asker,
                                     "model": getattr(pcfg, "model_id", "") or ""})
            ans_sys = ANSWER_SYSTEM.format(
                name=prole["name"], role_id=presenter,
                orientation=SCHOLAR_ORIENTATION[presenter],
                asker=arole["name"])
            ans_human = (f"{arole['name']}的问题：\n{question}\n\n"
                         f"你的研读笔记：\n"
                         f"{next((n.get('content', '') for n in state.get('reading_notes') or [] if n.get('agent_id') == presenter), '')[:1500]}\n\n"
                         "请回答。")
            try:
                answer, used, stopped = await _agent_speak(
                    ctx, state, presenter, ans_sys, ans_human,
                    float(prole.get("temperature", 0.5)), None, 1,
                    cfg=pcfg, prefix="sem_")
            except Exception as e:
                logger.warning("sem answer %s failed: %s", presenter, e)
                answer, used, stopped = (
                    f"（回应失败：{_short_reason(e)}）", 0, False)
            used_total += used
            entries.append({"kind": "answer", "agent_id": presenter,
                            "agent_name": prole["name"],
                            "target": asker, "content": answer[:800]})
            if stopped:
                return {"qa_transcript": entries, "stopped": True,
                        "token_budget_used": used_total}

        # 主席采集洞见（一次控制类调用；失败则洞见池不增长，问答记录仍在）
        insights: list[dict] = []
        open_qs: list[dict] = []
        try:
            qa_pairs = "\n\n".join(f"[{e['agent_name']}→{e.get('target', '')}]"
                                   f" {e['content']}" for e in entries) or "（无）"
            text, used = await _llm_text(
                ctx, state,
                CHAIR_INSIGHT_PROMPT.format(presenter=prole["name"],
                                            presentation=presentation[:1500],
                                            qa_pairs=qa_pairs[:20000]),
                "请提取洞见并输出 JSON。", temperature=0.2, prefix="sem_")
            used_total += used
            data = _loads_fuzzy(text) or {}
            if isinstance(data.get("insights"), list):
                insights = [{"kind": str(i.get("kind", "gap")),
                             "statement": str(i.get("statement", ""))[:200],
                             "origin": str(i.get("origin", ""))[:60]}
                            for i in data["insights"][:8]]
            if isinstance(data.get("open_questions"), list):
                open_qs = [{"q": str(q)[:150]}
                           for q in data["open_questions"][:5]]
        except Exception as e:
            logger.warning("sem insight extraction failed: %s", e)
        if insights or open_qs:
            emit("sem_insight", {"insights": insights,
                                 "open_questions": open_qs})
        return {"qa_transcript": entries, "insight_board": insights,
                "open_questions": open_qs,
                "token_budget_used": used_total}
    return qa_node


# ---------- Phase 3：构想工作坊 ----------

def make_propose_node(ctx: WorkflowContext, role_id: str):
    async def propose_node(state: SeminarState) -> dict:
        """各学者基于洞见池+自己的笔记提出构想卡（JSON 数组输出）。"""
        # 并行 fan-out 节点（编排纪律）：绝不写单值 channel（stopped）——
        # 四路同时返回会 InvalidUpdateError。停止语义走 is_stopped 注册表，
        # 本节点只负责"少产出/不产出"，收尾由 rapporteur 统一处理。
        if is_stopped(state["session_id"]):
            return {}
        rmap = _role_map(state)
        role = rmap[role_id]
        cfg = _role_cfg(ctx, state, role_id)
        emit("sem_agent_start", {"agent": role_id, "phase": "ideate",
                                 "model": getattr(cfg, "model_id", "") or ""})
        directive = (state.get("ideation_directives") or {}).get(role_id) \
            or SCHOLAR_ORIENTATION[role_id]
        own = next((n.get("content", "") for n in state.get("reading_notes") or []
                    if n.get("agent_id") == role_id), "")
        max_cards = int(getattr(ctx.settings, "seminar_cards_per_scholar", 2))
        system = PROPOSE_SYSTEM.format(
            name=role["name"], role_id=role_id,
            orientation=SCHOLAR_ORIENTATION[role_id], directive=directive,
            max_cards=max_cards)
        human = (f"议题：{state['topic']}\n\n"
                 f"【你的研读笔记】\n{own}\n\n"
                 f"【全场洞见池】\n{_insight_digest(state)}\n\n"
                 "请提出你的研究构想。")
        try:
            text, used, stopped = await _agent_speak(
                ctx, state, role_id, system, human,
                float(role.get("temperature", 0.5)), None, 1,
                cfg=cfg, prefix="sem_")
        except Exception as e:
            logger.warning("sem propose %s failed: %s", role_id, e)
            text, used, stopped = "", 0, False
        cards = []
        data = _loads_list(text) if text.strip() else None
        if isinstance(data, list):
            cards = [{"author": role_id,
                      "title": str(c.get("title", ""))[:120],
                      "inspiration": str(c.get("inspiration", ""))[:300],
                      "hypothesis": str(c.get("hypothesis", ""))[:500],
                      "validation_sketch": str(c.get("validation_sketch", ""))[:5500],
                      "risk": str(c.get("risk", ""))[:300]}
                     for c in data[:max_cards] if isinstance(c, dict)
                     and c.get("title")]
        # 并行节点不写单值 channel（stopped）：停止由注册表承载，
        # 本阶段的"少产出"由路由与 rapporteur 兜底
        return {"idea_cards": cards, "token_budget_used": used}
    return propose_node


def make_merge_node(ctx: WorkflowContext):
    async def merge_node(state: SeminarState) -> dict:
        """主席定稿：去重 + 统一编号。失败 → 代码兜底（按到达序编号）。"""
        raw = [c for c in state.get("idea_cards") or [] if not c.get("builds_on")]
        cards: list[dict] = []
        used = 0
        try:
            cards_json = json.dumps(
                [{"i": i, **c} for i, c in enumerate(raw)], ensure_ascii=False)
            text, used = await _llm_text(
                ctx, state, MERGE_PROMPT.format(cards=cards_json[:150000]),
                "请合并定稿并输出 JSON。", temperature=0.2, prefix="sem_")
            data = _loads_fuzzy(text) or {}
            if isinstance(data.get("merged_cards"), list):
                for c in data["merged_cards"]:
                    if not isinstance(c, dict) or not c.get("title"):
                        continue
                    cards.append({"card_id": str(c.get("card_id", ""))[:8],
                                  "author": str(c.get("author", "?"))[:20],
                                  "title": str(c.get("title", ""))[:120],
                                  "inspiration": str(c.get("inspiration", ""))[:300],
                                  "hypothesis": str(c.get("hypothesis", ""))[:500],
                                  "validation_sketch":
                                      str(c.get("validation_sketch", ""))[:5500],
                                  "risk": str(c.get("risk", ""))[:300]})
        except Exception as e:
            logger.warning("seminar merge failed, code fallback: %s", e)
        if not cards:                          # 代码兜底：按到达序编号
            cards = [{"card_id": f"C{i+1}", **c} for i, c in enumerate(raw)]
        for c in cards:
            emit("sem_idea_card", {"card": c})
        return {"card_registry": cards, "token_budget_used": used}
    return merge_node


def make_improve_node(ctx: WorkflowContext, role_id: str):
    async def improve_node(state: SeminarState) -> dict:
        """建构轮：各学者必须改进至少一张**他人的**卡（builds_on 谱系）。"""
        # 同 propose：并行节点不写单值 channel（stopped）
        if is_stopped(state["session_id"]):
            return {}
        rmap = _role_map(state)
        role = rmap[role_id]
        cfg = _role_cfg(ctx, state, role_id)
        emit("sem_agent_start", {"agent": role_id, "phase": "improve",
                                 "model": getattr(cfg, "model_id", "") or ""})
        others = [c for c in state.get("card_registry") or []
                  if c.get("author") != role_id]
        if not others:                         # 没有他人的卡可改进
            return {}
        system = IMPROVE_SYSTEM.format(
            name=role["name"], role_id=role_id,
            orientation=SCHOLAR_ORIENTATION[role_id],
            cards=_cards_digest(others, limit=500),
            insights=_insight_digest(state, 15))
        human = "请提出你对他人构想的改进版。"
        try:
            text, used, stopped = await _agent_speak(
                ctx, state, role_id, system, human,
                float(role.get("temperature", 0.5)), None, 1,
                cfg=cfg, prefix="sem_")
        except Exception as e:
            logger.warning("sem improve %s failed: %s", role_id, e)
            text, used, stopped = "", 0, False
        cards = []
        data = _loads_list(text) if text.strip() else None
        if isinstance(data, list):
            known = {c.get("card_id") for c in state.get("card_registry") or []}
            for c in data[:3]:
                if not isinstance(c, dict):
                    continue
                builds = str(c.get("builds_on", ""))[:8]
                if builds not in known:        # 谱系校验：只能改进已定稿的卡
                    continue
                cards.append({"builds_on": builds, "author": role_id,
                              "title": str(c.get("title", ""))[:120],
                              "improvement": str(c.get("improvement", ""))[:300],
                              "hypothesis": str(c.get("hypothesis", ""))[:500],
                              "validation_sketch":
                                  str(c.get("validation_sketch", ""))[:5500],
                              "risk": str(c.get("risk", ""))[:300]})
        # 同 propose：并行节点不写单值 channel（stopped）
        return {"idea_cards": cards, "token_budget_used": used}
    return improve_node


def make_panel_node(ctx: WorkflowContext):
    async def panel_node(state: SeminarState) -> dict:
        """主席公布参评卡：原卡 + 改进卡合并定稿（merge 同款，兜底代码编号）。"""
        originals = state.get("card_registry") or []
        improved = [c for c in state.get("idea_cards") or [] if c.get("builds_on")]
        raw = originals + improved
        cards: list[dict] = []
        used = 0
        try:
            cards_json = json.dumps(raw, ensure_ascii=False, default=str)
            text, used = await _llm_text(
                ctx, state, MERGE_PROMPT.format(cards=cards_json[:150000]),
                "请合并定稿全部卡片并输出 JSON。", temperature=0.2,
                prefix="sem_")
            data = _loads_fuzzy(text) or {}
            if isinstance(data.get("merged_cards"), list):
                for c in data["merged_cards"]:
                    if isinstance(c, dict) and c.get("title"):
                        cards.append({"card_id": str(c.get("card_id", ""))[:8],
                                      "author": str(c.get("author", "?"))[:20],
                                      "builds_on": c.get("builds_on"),
                                      "title": str(c.get("title", ""))[:120],
                                      "improvement":
                                          str(c.get("improvement") or "")[:300],
                                      "inspiration":
                                          str(c.get("inspiration", ""))[:300],
                                      "hypothesis": str(c.get("hypothesis", ""))[:500],
                                      "validation_sketch":
                                          str(c.get("validation_sketch", ""))[:5500],
                                      "risk": str(c.get("risk", ""))[:300]})
        except Exception as e:
            logger.warning("seminar panel failed, code fallback: %s", e)
        if not cards:
            cards = [{"card_id": f"C{i+1}", **c} for i, c in enumerate(raw)]
        for c in cards:
            emit("sem_idea_card", {"card": c, "final": True})
        return {"card_registry": cards, "token_budget_used": used}
    return panel_node


# ---------- Phase 4：评议会 ----------

def make_score_node(ctx: WorkflowContext, role_id: str):
    async def score_node(state: SeminarState) -> dict:
        """各学者对全部卡片独立打分（并行 fan-out，互不可见）。"""
        # 同 propose：并行节点不写单值 channel（stopped）
        if is_stopped(state["session_id"]):
            return {}
        rmap = _role_map(state)
        role = rmap[role_id]
        cfg = _role_cfg(ctx, state, role_id)
        emit("sem_agent_start", {"agent": role_id, "phase": "review",
                                 "model": getattr(cfg, "model_id", "") or ""})
        cards = state.get("card_registry") or []
        system = SCORE_SYSTEM.format(
            name=role["name"], role_id=role_id,
            orientation=SCHOLAR_ORIENTATION[role_id],
            cards=_cards_digest(cards, limit=2000))
        human = "请独立评审全部构想卡。"
        try:
            text, used, stopped = await _agent_speak(
                ctx, state, role_id, system, human,
                float(role.get("temperature", 0.4)), None, 1,
                cfg=cfg, prefix="sem_")
        except Exception as e:
            logger.warning("sem score %s failed: %s", role_id, e)
            text, used, stopped = "", 0, False
        scores = []
        data = _loads_list(text) if text.strip() else None
        if isinstance(data, list):
            known = {c.get("card_id") for c in cards}
            for s in data:
                if not isinstance(s, dict) or s.get("card_id") not in known:
                    continue

                def _dim(k):                   # 夹取 0-10 整数
                    try:
                        return max(0, min(10, int(s.get(k, 0))))
                    except (TypeError, ValueError):
                        return 0

                scores.append({"card_id": s.get("card_id"),
                               "reviewer": role_id,
                               "novelty": _dim("novelty"),
                               "significance": _dim("significance"),
                               "feasibility": _dim("feasibility"),
                               "risk_control": _dim("risk_control"),
                               "comment": str(s.get("comment", ""))[:300]})
        emit("sem_review", {"agent": role_id, "scores": scores})
        # 同 propose：并行节点不写单值 channel（stopped）
        return {"review_scores": scores, "token_budget_used": used}
    return score_node


# 四维权重：创新/意义/可行为主，风险可控为辅
_SCORE_WEIGHTS = {"novelty": 0.3, "significance": 0.3,
                  "feasibility": 0.3, "risk_control": 0.1}


def _stdev(xs: list[float]) -> float:
    """总体标准差（评审分歧度）。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5


def aggregate_node(state: SeminarState) -> dict:
    """纯代码聚合（零 LLM）：均分、加权总分、分歧度（标准差）、排序。

    分歧度不抹平——高分歧 = 高风险高潜力，在产出中如实标注。
    评分数 < 2 的卡不参与排序（均值无统计意义），仅罗列。
    """
    scores = state.get("review_scores") or []
    by_card: dict[str, list[dict]] = {}
    for s in scores:
        by_card.setdefault(s.get("card_id", "?"), []).append(s)
    ranking = []
    for card in state.get("card_registry") or []:
        cid = card.get("card_id", "?")
        revs = by_card.get(cid, [])
        if not revs:
            continue
        dims = {}
        for k in _SCORE_WEIGHTS:
            vals = [r[k] for r in revs]
            dims[k] = round(sum(vals) / len(vals), 1)
        total = round(sum(dims[k] * w for k, w in _SCORE_WEIGHTS.items()), 2)
        ranking.append({
            "card_id": cid, "author": card.get("author"),
            "builds_on": card.get("builds_on"), "title": card.get("title"),
            "reviewers": len(revs), "dims": dims, "total": total,
            "divergence": round(_stdev(
                [sum(r[k] * w for k, w in _SCORE_WEIGHTS.items())
                 for r in revs]), 2),
            "comments": [r.get("comment", "") for r in revs if r.get("comment")]})
    ranked = sorted([r for r in ranking if r["reviewers"] >= 2],
                    key=lambda r: r["total"], reverse=True)
    unranked = [r for r in ranking if r["reviewers"] < 2]
    final = ranked + unranked
    if final:
        emit("sem_ranking", {"ranking": final})
    return {"idea_ranking": final}


# ---------- Phase 5：执笔成稿 ----------

def _fallback_report(state: SeminarState) -> str:
    """降级成稿：执笔人失败/停止时，结构化汇编全部中间产物。"""
    parts = [f"# 会讲报告（汇编版）：{state['topic']}", "",
             "> 注：执笔人生成失败或中途停止，以下为会讲中间产物的结构化汇编。"]
    if state.get("reading_notes"):
        parts.append("## 学者研读笔记")
        for n in state["reading_notes"]:
            parts.append(f"### {n.get('agent_name')}\n{n.get('content', '')}")
    if state.get("insight_board"):
        parts.append("## 洞见池\n" + _insight_digest(state, 50))
    if state.get("card_registry"):
        parts.append("## 构想卡\n" + _cards_digest(state["card_registry"], 500))
    if state.get("idea_ranking"):
        rows = "\n".join(f"- {r['card_id']}（{r.get('author')}）总分 {r['total']}"
                         f" 分歧度 {r['divergence']}：{r['title']}"
                         for r in state["idea_ranking"])
        parts.append(f"## 评审排序\n{rows}")
    return "\n\n".join(parts)


def make_rapporteur_node(ctx: WorkflowContext):
    async def rapporteur_node(state: SeminarState) -> dict:
        """执笔人成稿。纪律与争鸣社 synthesis 一致：
        用户停止 → 汇编降级稿；预算熔断不阻断执笔（截断只影响前面阶段）。"""
        sid = state["session_id"]
        stopped = is_stopped(sid)
        budget = int(state.get("token_budget")
                     or getattr(ctx.settings, "seminar_token_budget",
                                20000000))
        over_budget = int(state.get("token_budget_used") or 0) > budget
        if stopped:
            clear_stop(sid)
            content = _fallback_report(state)
            emit("sem_plan", {"content": content, "fallback": True,
                              "reason": "stopped"})
            return {"final_proposal": content, "stopped": True,
                    **({"budget_exhausted": True} if over_budget else {})}

        cfg = _role_cfg(ctx, state, "rapporteur") \
            if "rapporteur" in _role_map(state) else None
        if cfg is None:
            cfg = _effective_cfg(ctx, state["user_id"])
        emit("sem_agent_start", {"agent": "rapporteur", "phase": "synthesis",
                                 "model": getattr(cfg, "model_id", "") or ""})
        notes = _notes_digest(state, limit=400)
        ranking = state.get("idea_ranking") or []
        ranking_text = "\n".join(
            f"- {r['card_id']} 总分{r['total']} 分歧度{r['divergence']} "
            f"各维{r['dims']} 评语：{'; '.join(r.get('comments', [])[:3])}"
            for r in ranking) or "（无评审）"
        open_qs = "\n".join(q.get("q", "") for q in
                            state.get("open_questions") or []) or "（无）"
        system = RAPPORTEUR_PROMPT.format(
            topic=state["topic"], notes=notes,
            insights=_insight_digest(state, 50),
            cards=_cards_digest(state.get("card_registry") or [], 5500),
            ranking=ranking_text, open_questions=open_qs)
        # 执笔人文档工具子集（产出资产，不再调研）
        def _schema_name(t: dict) -> str:
            return t.get("name") or (t.get("function") or {}).get("name") or ""
        doc_tools = []
        if ctx.mcp_adapter is not None:
            doc_tools = [t for t in
                         (await ctx.mcp_adapter.schemas_for_llm() or [])
                         if _schema_name(t) in ("save_document", "export_bibtex")]
        try:
            content, used, _ = await _agent_speak(
                ctx, state, "rapporteur", system,
                "请按固定骨架撰写会讲报告；完成后可保存文档/导出引文。",
                0.4, doc_tools, 3, cfg=cfg, prefix="sem_")
            if not content.strip():
                raise RuntimeError("empty report")
        except Exception as e:
            logger.warning("sem rapporteur failed, fallback: %s", e)
            content, used = _fallback_report(state), 0
        emit("sem_plan", {"content": content})
        out = {"final_proposal": content, "token_budget_used": used}
        if over_budget:
            out["budget_exhausted"] = True
        return out
    return rapporteur_node

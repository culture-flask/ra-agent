"""深度调研子图：第三支多 agent 团队，产出带多源引用的研究报告。

## 拓扑（与前两支团队的关键差异）

  START → scout ──▶ plan ──▶ research ──▶ merge_sources ──▶ review
                                ▲                              │
                                │                    REVISE ───┤
                                │                              │
                                │         revise ◀─────────────┘
                                │            └──▶ review（轮次+1）
                                │
                                └──── archive ◀── PASS / 第3轮熔断
                                         │
                       还有章节？─────────┤
                                         ▼
                                   write_frame ──▶ publish ──▶ END

争鸣社是「LLM 主持人调度循环」、会讲是「agenda_index 代码循环 + Send 并行」，
本团队是**双层嵌套循环**：外层章循环（chapter_index）× 内层审稿轮次循环
（review_round）。两个计数器都由代码单写者维护，不交给模型。

## 三条纪律（继承前两支，并新增一条）

1. 全串行、无 fan-out —— 章节必须逐章推进，后章复用前章的来源与结论；
   因此单值 channel 由单写者覆盖是安全的（与会讲的并行纪律不冲突）。
2. 工具失败 → 无工具继续；单角色失败 → 缺席声明继续（不中断全场）。
3. 预算熔断只截断**后续章节**，write_frame / publish 不受阻断——
   否则前面几章的调研成果会因为最后一次超预算而全部丢失。
4. 【新增】**第 3 轮强制通过在代码层兜底**：路由函数里判 review_round >= 3
   就直接放行。提示词里也写了同一条，但模型未必守约，代码必须兜住。
"""

import asyncio
import json
import re
from datetime import datetime, timezone

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
from app.graph.prompts.deep_research import (
    PLAN_SYSTEM, RESEARCH_SYSTEM, REVISE_SYSTEM, REVIEW_SYSTEM,
    SCOUT_SYSTEM, WRITE_FRAME_SYSTEM, WRITE_FRAME_PLAIN,
)
from app.graph.dr_state import DeepResearchState

logger = get_logger("deep_research")

DR_ROLE_IDS = ("planner", "researcher", "reviewer", "reviser", "writer")

MAX_REVIEW_ROUNDS = 3          # 审稿轮次上限（代码熔断，与提示词同值）
MIN_SOURCES_PER_CHAPTER = 5    # 每章来源数下限（低于此值记警告，不阻断）
SOURCE_POOL_MAX = 60           # 来源池容量上限（防上下文无限膨胀）
DEFAULT_BUDGET = 20_000_000

# 引用链接抽取：正文里所有 [标题](http...) 就是本章实际用到的来源
_LINK_RE = re.compile(r"\[([^\]]{1,120})\]\((https?://[^\s)]{5,500})\)")


# ---------- 工具函数 ----------

def _extract_links(text: str) -> list[dict]:
    """从 Markdown 正文抽取引用链接（按 URL 去重，保序）。"""
    out: list[dict] = []
    seen: set[str] = set()
    for title, url in _LINK_RE.findall(text or ""):
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": title.strip(), "url": url, "digest": ""})
    return out


def _over_budget(state: DeepResearchState) -> bool:
    budget = int(state.get("token_budget") or DEFAULT_BUDGET)
    if not budget:
        return False
    return int(state.get("token_budget_used") or 0) > budget


def _role_map(state: DeepResearchState) -> dict:
    """role_id -> 角色信息（名称/温度），缺省兜底。"""
    out: dict = {}
    for r in state.get("roles") or []:
        if r.get("id") in DR_ROLE_IDS:
            out[r["id"]] = r
    for rid in DR_ROLE_IDS:
        out.setdefault(rid, {"id": rid, "name": rid, "temperature": 0.3})
    return out


def _sections(state: DeepResearchState) -> list[dict]:
    return list((state.get("outline") or {}).get("sections") or [])


def _source_pool_digest(state: DeepResearchState, window: int = 30) -> str:
    pool = (state.get("source_pool") or [])[-window:]
    if not pool:
        return "（暂无，请自行检索）"
    return "\n".join(f"- [{s.get('title', '未命名')}]({s.get('url', '')})"
                     for s in pool)


def _done_digest(state: DeepResearchState, limit: int = 120) -> str:
    """已完成章节摘要（≤limit 字/章）：后章靠它避免重复论述。"""
    chapters = sorted(state.get("chapters") or [],
                      key=lambda c: c.get("index") or 0)
    if not chapters:
        return "（无，你是第一章）"
    return "\n".join(f"- 第{c.get('index')}章 {c.get('title', '')}："
                     f"{str(c.get('content', ''))[:limit]}"
                     for c in chapters)


def _stop_guard(state: DeepResearchState) -> dict | None:
    """节点入口统一停止检查：命中即返回增量，不干活。"""
    if is_stopped(state["session_id"]):
        return {"stopped": True}
    return None


# ---------- Phase 1：初调 ----------

def make_scout_node(ctx: WorkflowContext):
    async def scout_node(state: DeepResearchState) -> dict:
        """初调：带工具检索，产出研究摘要（纯 Markdown + 内联引用）。

        与会讲的 curate 不同——curate 只是解析议题（llm_text 无工具），
        初调必须真的去检索，否则整份报告的来源池是空的。
        """
        g = _stop_guard(state)
        if g:
            return g
        rid = "researcher"
        role = _role_map(state)[rid]
        cfg = _role_cfg(ctx, state, rid)
        emit("dr_agent_start", {"agent": rid, "phase": "scout",
                                "model": getattr(cfg, "model_id", "") or ""})
        tools = []
        if ctx.mcp_adapter is not None:
            tools = await ctx.mcp_adapter.schemas_for_llm() or []
        recency = str(state.get("recency") or "两年")
        system = SCOUT_SYSTEM.format(topic=state["topic"], recency=recency,
                                     min_sources=MIN_SOURCES_PER_CHAPTER)
        loop_max = int(getattr(ctx.settings, "deep_research_scout_tool_loop_max", 8))
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, rid, system, "请开始初步调研，输出研究摘要。",
                float(role.get("temperature", 0.4)), tools, loop_max,
                cfg=cfg, prefix="dr_")
        except Exception as e:
            logger.warning("dr scout failed: %s", e)
            fail_reason = _short_reason(e)
            content, used, stopped = "", 0, False
            if is_stopped(state["session_id"]):
                return {"stopped": True}

        links = _extract_links(content)
        if content.strip() and len(links) < MIN_SOURCES_PER_CHAPTER:
            warn = (f"初调来源不足：仅检索到 {len(links)} 条"
                    f"（建议 ≥{MIN_SOURCES_PER_CHAPTER} 条）")
            logger.warning(warn)
        emit("dr_scout", {"chars": len(content), "sources": len(links),
                          **({"failed": True, "reason": fail_reason}
                             if fail_reason else {})})
        emit("dr_agent_end", {"agent": rid, "phase": "scout",
                              "failed": not content.strip()})
        return {"scouting_brief": content, "source_pool": links,
                "token_budget_used": used,
                **({"stopped": True} if stopped else {})}
    return scout_node


# ---------- Phase 2：大纲规划 ----------

def make_plan_node(ctx: WorkflowContext):
    async def plan_node(state: DeepResearchState) -> dict:
        """规划大纲（控制类调用，不流式透传，JSON 输出）。

        研究编辑也要有气泡：agent_start/token/end 三件套与审稿同款——
        否则这个角色在直播与回放里都"不存在"（真实 bug：全程只见
        研究员/审稿人/撰写人，研究编辑凭空消失）。"""
        g = _stop_guard(state)
        if g:
            return g
        emit("dr_plan", {"topic": state["topic"][:100]})
        rid = "planner"
        role = _role_map(state)[rid]
        cfg = _role_cfg(ctx, state, rid)
        emit("dr_agent_start", {"agent": rid, "phase": "plan",
                                "model": getattr(cfg, "model_id", "") or ""})
        quick = str(state.get("mode") or "full") == "quick"
        min_s = 3 if quick else int(getattr(ctx.settings,
                                            "deep_research_min_sections", 3))
        max_s = 3 if quick else int(getattr(ctx.settings,
                                            "deep_research_max_sections", 5))
        outline: dict = {}
        used = 0
        try:
            text, used = await _llm_text(
                ctx, state,
                PLAN_SYSTEM.format(topic=state["topic"],
                                   brief=str(state.get("scouting_brief") or "")[:4000],
                                   min_sections=min_s, max_sections=max_s),
                "请规划大纲并输出 JSON。", temperature=0.2, prefix="dr_")
            data = _loads_fuzzy(text) or {}
            secs = data.get("sections")
            if isinstance(secs, list):
                clean = []
                for i, s in enumerate(secs[:max_s]):
                    if not isinstance(s, dict) or not s.get("title"):
                        continue
                    kps = s.get("key_points") or []
                    clean.append({
                        "index": i + 1,
                        "title": str(s["title"])[:80],
                        "key_points": [str(k)[:200] for k in kps[:5]]
                        if isinstance(kps, list) else [],
                        "rationale": str(s.get("rationale", ""))[:200]})
                if clean:
                    outline = {"title": str(data.get("title")
                                            or state["topic"])[:120],
                               "sections": clean}
        except Exception as e:
            logger.warning("dr plan failed, fallback: %s", e)
        if not outline:                       # 代码兜底：单章也能跑完全流程
            outline = {"title": state["topic"][:120],
                       "sections": [{"index": 1, "title": "综合论述",
                                     "key_points": [], "rationale": "规划失败兜底"}]}
        emit("dr_plan", {"title": outline.get("title", ""),
                         "sections": [s["title"] for s in outline["sections"]]})
        # 大纲进研究编辑气泡（Markdown，与回放渲染同口径）
        outline_md = (f"### 报告大纲：《{outline.get('title', '')}》\n\n"
                      + "\n".join(
                          f"{s['index']}. **{s['title']}**"
                          + (f" —— {s['rationale']}" if s.get("rationale") else "")
                          for s in outline["sections"]))
        emit("dr_token", {"agent": rid, "content": outline_md})
        emit("dr_agent_end", {"agent": rid, "phase": "plan"})
        return {"outline": outline, "token_budget_used": used}
    return plan_node


# ---------- Phase 3.1：分章深研 ----------

def make_research_node(ctx: WorkflowContext):
    async def research_node(state: DeepResearchState) -> dict:
        """写当前章节草稿（带工具，纯 Markdown 输出 + 内联引用）。"""
        g = _stop_guard(state)
        if g:
            return g
        idx = int(state.get("chapter_index") or 0)
        secs = _sections(state)
        if idx >= len(secs):
            return {}                          # 防御：章节已完不该进来
        sec = secs[idx]
        rid = "researcher"
        role = _role_map(state)[rid]
        cfg = _role_cfg(ctx, state, rid)
        emit("dr_agent_start", {"agent": rid, "phase": "research",
                                "chapter": idx + 1,
                                "model": getattr(cfg, "model_id", "") or ""})
        tools = []
        if ctx.mcp_adapter is not None:
            tools = await ctx.mcp_adapter.schemas_for_llm() or []
        system = RESEARCH_SYSTEM.format(
            report_title=str((state.get("outline") or {}).get("title")
                             or state["topic"])[:120],
            chapter_no=idx + 1, total_chapters=len(secs),
            chapter_title=sec.get("title", ""),
            key_points="\n".join(f"- {k}" for k in sec.get("key_points") or [])
                       or "（未指定，请按章节标题自行界定）",
            topic=state["topic"],
            brief=str(state.get("scouting_brief") or "")[:3000],
            source_pool=_source_pool_digest(state),
            done_digest=_done_digest(state),
            min_words=int(getattr(ctx.settings, "deep_research_chapter_min_words", 800)),
            max_words=int(getattr(ctx.settings, "deep_research_chapter_max_words", 1500)),
            min_sources=MIN_SOURCES_PER_CHAPTER)
        loop_max = int(getattr(ctx.settings, "deep_research_tool_loop_max", 10))
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, rid, system, "请撰写本章正文。",
                float(role.get("temperature", 0.5)), tools, loop_max,
                cfg=cfg, prefix="dr_")
        except Exception as e:
            logger.warning("dr research ch%d failed: %s", idx + 1, e)
            fail_reason = _short_reason(e)
            content, used, stopped = "", 0, False
            if is_stopped(state["session_id"]):
                return {"stopped": True}
        if not content.strip() and fail_reason:
            content = f"（本章调研失败：{fail_reason}）"
        emit("dr_agent_end", {"agent": rid, "phase": "research",
                              "chapter": idx + 1,
                              "sources": len(_extract_links(content)),
                              "failed": bool(fail_reason)})
        # review_round 归零交给本节点（单写者），避免上一章的轮次残留
        return {"current_draft": content, "review_round": 0,
                "review_verdict": "", "review_feedback": "",
                "token_budget_used": used,
                **({"stopped": True} if stopped else {})}
    return research_node


# ---------- Phase 3.1b：来源池合并（纯代码） ----------

def merge_sources_node(state: DeepResearchState) -> dict:
    """从当前草稿抽取引用 → 按 URL 去重 → 整池写回。

    ⚠️ source_pool 是**单写者覆盖**而非 operator.add：add 会让同一来源
    随章节反复累积重复条目，注入上下文时白吃窗口。
    """
    by_url: dict[str, dict] = {}
    for s in state.get("source_pool") or []:
        u = str(s.get("url") or "").strip()
        if u:
            by_url[u] = s
    added = 0
    for s in _extract_links(state.get("current_draft") or ""):
        if s["url"] not in by_url:
            by_url[s["url"]] = s
            added += 1
    pool = list(by_url.values())[:SOURCE_POOL_MAX]
    if added:
        emit("dr_sources", {"added": added, "total": len(pool)})
    return {"source_pool": pool}


# ---------- Phase 3.2：审稿 ----------

def make_review_node(ctx: WorkflowContext):
    async def review_node(state: DeepResearchState) -> dict:
        """六维审稿（控制类调用，无工具——审稿人不能自己去"补证据"）。

        输出判定 PASS 或 REVISE JSON。解析失败 → 放行（fail-open）：
        已有代码层 3 轮熔断兜底，不会因此死循环。
        """
        g = _stop_guard(state)
        if g:
            return g
        idx = int(state.get("chapter_index") or 0)
        secs = _sections(state)
        sec = secs[idx] if idx < len(secs) else {}
        rnd = int(state.get("review_round") or 0) + 1     # 本轮序号（1-based）
        rid = "reviewer"
        role = _role_map(state)[rid]
        cfg = _role_cfg(ctx, state, rid)
        emit("dr_agent_start", {"agent": rid, "phase": "review",
                                "chapter": idx + 1, "round": rnd,
                                "model": getattr(cfg, "model_id", "") or ""})
        system = REVIEW_SYSTEM.format(
            chapter_title=sec.get("title", ""),
            key_points="\n".join(f"- {k}" for k in sec.get("key_points") or [])
                       or "（未指定）",
            draft=str(state.get("current_draft") or "")[:8000],
            round=rnd, max_rounds=MAX_REVIEW_ROUNDS,
            min_sources=MIN_SOURCES_PER_CHAPTER)
        verdict, feedback, used = "PASS", "", 0
        display = ""
        try:
            text, used = await _llm_text(
                ctx, state, system, "请审查本章草稿并输出判定。",
                temperature=float(role.get("temperature", 0.2)), prefix="dr_")
            text = (text or "").strip()
            if text.upper().strip("\"'` \n") == "PASS":
                verdict = "PASS"
                display = (f"**✅ 第 {rnd} 轮审稿通过（PASS）**\n\n"
                           "本章草稿达到「可信」标准，无需修订，继续下一章。")
            else:
                data = _loads_fuzzy(text) or {}
                v = str(data.get("verdict", "")).upper()
                if v == "REVISE" or data.get("must_fix"):
                    verdict = "REVISE"
                    lines = [f"必须修改项 {i}：{m}" for i, m in
                             enumerate(data.get("must_fix") or [], 1)]
                    lines += [f"建议改进项 {i}：{s}" for i, s in
                              enumerate(data.get("suggestions") or [], 1)]
                    if data.get("verdict_reason"):
                        lines.append(f"\n审稿结论：{data['verdict_reason']}")
                    feedback = "\n".join(lines)
                    display = f"**🔁 第 {rnd} 轮审稿退回（REVISE）**\n\n{feedback}"
                else:
                    if not text:
                        logger.warning("dr review empty output, fail-open")
                    verdict = "PASS"
                    display = "（审稿人未输出有效判定，按 fail-open 规则放行）"
        except Exception as e:
            logger.warning("dr review ch%d failed, fail-open: %s", idx + 1, e)
            verdict = "PASS"
            display = (f"（审稿调用失败：{_short_reason(e)}，"
                       "按 fail-open 规则放行）")
        # 审稿走控制类调用（不流式）——不补一条展示事件的话，前端审稿
        # 气泡从 agent_start 到 agent_end 全程空白，判定与退回意见用户
        # 都看不到（真实 bug）。判定是给前端的一句话摘要，原始输出
        # （PASS/JSON）不必透传。
        if display:
            emit("dr_token", {"agent": rid, "content": display})
        emit("dr_review", {"chapter": idx + 1, "round": rnd,
                           "verdict": verdict})
        emit("dr_agent_end", {"agent": rid, "phase": "review",
                              "chapter": idx + 1, "round": rnd})
        # 审稿流水（只增）：单值的 review_verdict/feedback 每章被覆盖，
        # 详情回放靠这条记录还原每章每轮的审稿结论（否则刷新后审稿人消失）
        return {"review_verdict": verdict, "review_feedback": feedback,
                "review_log": [{"chapter": idx + 1, "round": rnd,
                                "verdict": verdict, "content": display}],
                "token_budget_used": used}
    return review_node


# ---------- Phase 3.3：修订 ----------

def make_revise_node(ctx: WorkflowContext):
    async def revise_node(state: DeepResearchState) -> dict:
        """依审稿意见修订（带工具：需要真的去检索补充引用）。

        轮次递增由本节点写（单写者）。审稿未批评的部分靠提示词约束不动。
        """
        g = _stop_guard(state)
        if g:
            return g
        idx = int(state.get("chapter_index") or 0)
        secs = _sections(state)
        sec = secs[idx] if idx < len(secs) else {}
        rid = "reviser"
        role = _role_map(state)[rid]
        cfg = _role_cfg(ctx, state, rid)
        rnd = int(state.get("review_round") or 0) + 1
        emit("dr_agent_start", {"agent": rid, "phase": "revise",
                                "chapter": idx + 1, "round": rnd,
                                "model": getattr(cfg, "model_id", "") or ""})
        tools = []
        if ctx.mcp_adapter is not None:
            tools = await ctx.mcp_adapter.schemas_for_llm() or []
        system = REVISE_SYSTEM.format(
            chapter_title=sec.get("title", ""),
            draft=str(state.get("current_draft") or "")[:8000],
            feedback=str(state.get("review_feedback") or "")[:3000],
            source_pool=_source_pool_digest(state))
        loop_max = int(getattr(ctx.settings, "deep_research_revise_tool_loop_max", 5))
        fail_reason = ""
        try:
            content, used, stopped = await _agent_speak(
                ctx, state, rid, system, "请输出修订后的完整章节正文。",
                float(role.get("temperature", 0.3)), tools, loop_max,
                cfg=cfg, prefix="dr_")
        except Exception as e:
            logger.warning("dr revise ch%d r%d failed: %s", idx + 1, rnd, e)
            fail_reason = _short_reason(e)
            content, used, stopped = "", 0, False
            if is_stopped(state["session_id"]):
                return {"stopped": True}
        if not content.strip():
            # 修订失败：保留原稿（不能把已通过的部分弄丢），但轮次照常递增
            content = str(state.get("current_draft") or "")
            if fail_reason:
                content += f"\n\n> ⚠️ 本轮修订失败（{fail_reason}），沿用上一版正文。"
        emit("dr_agent_end", {"agent": rid, "phase": "revise",
                              "chapter": idx + 1, "round": rnd,
                              "failed": bool(fail_reason)})
        return {"current_draft": content, "review_round": rnd,
                "token_budget_used": used,
                **({"stopped": True} if stopped else {})}
    return revise_node


# ---------- Phase 3.4：归档本章、推进章号 ----------

def archive_chapter_node(state: DeepResearchState) -> dict:
    """纯代码：把 current_draft 归档进 chapters，重置循环变量，推进章号。

    chapter_index 的唯一写者就是本节点（会讲的 agenda_index 同款纪律）。
    """
    idx = int(state.get("chapter_index") or 0)
    secs = _sections(state)
    sec = secs[idx] if idx < len(secs) else {}
    draft = str(state.get("current_draft") or "").strip()
    out: dict = {"chapter_index": idx + 1, "current_draft": "",
                 "review_verdict": "", "review_feedback": "",
                 "review_round": 0}
    if not draft:
        emit("dr_chapter_done", {"chapter": idx + 1, "skipped": True})
        return out
    links = _extract_links(draft)
    warnings: list[dict] = []
    if state.get("review_verdict") == "REVISE":
        # 走到归档却仍是 REVISE = 第 3 轮熔断强制放行
        warnings.append({"chapter": idx + 1, "kind": "force_pass",
                         "detail": f"经 {MAX_REVIEW_ROUNDS} 轮审核仍未获通过，"
                                   f"已强制放行，遗留问题见本节"})
    if len(links) < MIN_SOURCES_PER_CHAPTER:
        warnings.append({"chapter": idx + 1, "kind": "thin_sources",
                         "detail": f"本章仅 {len(links)} 个来源"
                                   f"（建议 ≥{MIN_SOURCES_PER_CHAPTER}）"})
    chapter = {"index": idx + 1, "title": sec.get("title") or f"第{idx+1}章",
               "content": draft, "rounds": int(state.get("review_round") or 0) + 1,
               "sources": links}
    emit("dr_chapter_done", {"chapter": idx + 1,
                             "title": chapter["title"],
                             "sources": len(links),
                             "chars": len(draft)})
    out["chapters"] = [chapter]
    if warnings:
        out["warnings"] = warnings
    return out


# ---------- 路由函数（纯代码，无 LLM） ----------

def route_after_scout(state: DeepResearchState) -> str:
    return "publish" if state.get("stopped") else "plan"


def route_after_plan(state: DeepResearchState) -> str:
    return "publish" if state.get("stopped") else "research"


def route_after_merge(state: DeepResearchState) -> str:
    """merge_sources 之后：停止 → 直接成稿；快速模式 → 跳过审稿。"""
    if state.get("stopped"):
        return "publish"
    if str(state.get("mode") or "full") == "quick":
        return "archive"
    return "review"


def route_after_review(state: DeepResearchState) -> str:
    """审稿之后的三岔口 —— **第 3 轮熔断在这里由代码兜底**。

    提示词里也写了「第 3 轮必须 PASS」，但模型未必守约；这里再判一次：
    review_round 已达上限就直接归档，绝不让系统进入第 4 轮。
    """
    if state.get("stopped"):
        return "archive"
    if state.get("review_verdict") != "REVISE":
        return "archive"
    if int(state.get("review_round") or 0) >= MAX_REVIEW_ROUNDS:
        logger.warning("dr ch%d hit round cap %d, force archive",
                       int(state.get("chapter_index") or 0) + 1,
                       MAX_REVIEW_ROUNDS)
        return "archive"
    return "revise"


def route_after_archive(state: DeepResearchState) -> str:
    """归档之后：还有章节？→ 回到 research；否则 → write_frame。

    预算耗尽只截断**后续章节**，不阻断 write_frame/publish —— 否则前面
    几章的调研成果会因为最后一次超预算而全部丢失（会讲 rapporteur 同纪律）。
    """
    if state.get("stopped"):
        return "publish"
    if int(state.get("chapter_index") or 0) >= len(_sections(state)):
        return "write_frame"
    if _over_budget(state):
        logger.warning("dr budget exhausted, skip remaining chapters")
        return "write_frame"
    return "research"


# ---------- Phase 4：报告框架 ----------

def _parse_frame_sections(text: str) -> tuple[str, str, str]:
    """定界符格式解析（===TOC=== / ===INTRO=== / ===CONCL===）。

    JSON 契约的兜底：弱模型输出含长 Markdown 的 JSON 时经常不转义字符串
    内的换行——json.loads 必然失败，_loads_fuzzy 也救不回（实测
    sensenova-flash-lite）；三段定界符对转义免疫。"""
    if not text:
        return "", "", ""
    parts = re.split(r"^\s*={3}\s*(TOC|INTRO|CONCL)\s*={3}\s*$", text, flags=re.M)
    out: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):     # [前文, 标记1, 段1, 标记2, 段2, ...]
        out[parts[i]] = parts[i + 1].strip()
    return out.get("TOC", ""), out.get("INTRO", ""), out.get("CONCL", "")


def make_write_frame_node(ctx: WorkflowContext):
    async def write_frame_node(state: DeepResearchState) -> dict:
        """引言 / 结论 / 目录（控制类调用）。

        两段式契约：先 JSON（WRITE_FRAME_SYSTEM）；解析结果为空（空完成
        或弱模型 JSON 转义非法）→ 换定界符格式（WRITE_FRAME_PLAIN）重试
        ——对转义免疫，两类根因一次覆盖。仍失败才交由 publish 降级汇编。
        参考文献**不交给模型**——由 publish 从 source_pool 纯代码生成，
        保证去重与可追溯。
        """
        g = _stop_guard(state)
        if g:
            return g
        rid = "writer"
        role = _role_map(state)[rid]
        cfg = _role_cfg(ctx, state, rid)
        emit("dr_agent_start", {"agent": rid, "phase": "frame",
                                "model": getattr(cfg, "model_id", "") or ""})
        chapters = sorted(state.get("chapters") or [],
                          key=lambda c: c.get("index") or 0)
        chapters_text = "\n\n".join(
            f"## 第{c.get('index')}章 {c.get('title', '')}\n"
            f"{str(c.get('content', ''))[:2500]}" for c in chapters) or "（无章节）"
        frame_kw = dict(topic=state["topic"],
                        report_title=str((state.get("outline") or {}).get("title")
                                         or state["topic"])[:120],
                        chapters=chapters_text[:12000])
        toc, intro, concl, used = "", "", "", 0
        raw_text = ""
        last_err = ""
        # 不流式但用户在场：重试倒计时与失败原因透出到撰写人气泡
        retry_kw = dict(emit_retry=True, emit_extra={"agent": rid})
        try:
            text, used = await _llm_text(
                ctx, state, WRITE_FRAME_SYSTEM.format(**frame_kw),
                "请撰写引言/结论/目录并输出 JSON。",
                temperature=float(role.get("temperature", 0.4)), prefix="dr_",
                **retry_kw)
            raw_text = (text or "").strip()
            data = _loads_fuzzy(raw_text) or {}
            toc = str(data.get("toc", ""))[:2000]
            intro = str(data.get("introduction", ""))[:3000]
            concl = str(data.get("conclusion", ""))[:3000]
        except Exception as e:
            logger.warning("dr write_frame failed: %s", e)
            last_err = str(e)
        if not (toc or intro or concl):
            # 第一段契约失败：留诊断证据（原始输出头部），换定界符格式重试
            logger.warning("dr write_frame json attempt empty (raw head: %s)，"
                           "retry with plain-delimiter contract",
                           raw_text[:300] or "（空完成）")
            try:
                text, used2 = await _llm_text(
                    ctx, state, WRITE_FRAME_PLAIN.format(**frame_kw),
                    "请按 ===TOC===/===INTRO===/===CONCL=== 三段格式输出。",
                    temperature=float(role.get("temperature", 0.4)),
                    prefix="dr_", **retry_kw)
                used += used2
                toc, intro, concl = _parse_frame_sections(text or "")
            except Exception as e:
                logger.warning("dr write_frame plain retry failed: %s", e)
                last_err = str(e)
        if not (toc or intro or concl):
            logger.warning("dr write_frame both contracts empty（报告将降级汇编），"
                           "raw head: %s", raw_text[:300] or "（空）")
            # 重试耗尽仍无产出：失败原因即时透出（发布员会降级汇编，但
            # 撰写人气泡不能只留空白，用户要能看到为什么降级）
            emit("dr_agent_fail", {"agent": rid, "error":
                _short_reason(Exception(last_err)) if last_err
                else "两段输出契约均为空（网关空完成或格式不可解析）"})
        # 与审稿同理：控制类调用不产生流式 token，不补展示事件的话前端
        # 撰写人气泡全程空白。这里把目录/引言/结论的实际产出回放进气泡
        # （与详情回放路径 put("writer","frame",…) 的内容口径一致）。
        frame_md = "\n\n".join(
            f"## {header}\n\n{body}" for header, body in
            (("目录", toc), ("引言", intro), ("结论", concl)) if body)
        if frame_md:
            emit("dr_token", {"agent": rid, "content": frame_md})
        emit("dr_frame", {"toc": toc[:200], "intro": len(intro),
                          "conclusion": len(concl)})
        emit("dr_agent_end", {"agent": rid, "phase": "frame"})
        return {"toc": toc, "introduction": intro, "conclusion": concl,
                "token_budget_used": used}
    return write_frame_node


# ---------- Phase 5：成稿（纯代码） ----------

def _fallback_report(state: DeepResearchState) -> str:
    """降级成稿：未走 write_frame（停止/预算熔断）时，结构化汇编已有章节。"""
    parts = [f"# {str((state.get('outline') or {}).get('title') or state.get('topic') or '深度研究报告')}",
             "", "> 注：本场未正常完成，以下为已完成章节的结构化汇编。"]
    for c in sorted(state.get("chapters") or [], key=lambda x: x.get("index") or 0):
        parts.append(f"## {c.get('index')}. {c.get('title', '')}")
        parts.append(str(c.get("content", "")))
    return "\n\n".join(parts) or "（本场未产出任何章节内容）"


def publish_node(state: DeepResearchState) -> dict:
    """纯代码整合与格式化（零 LLM，不改动任何研究内容）。

    ⚠️ 为什么发布员不用模型：整合/编号/去重/排序恰恰是 LLM 最不可靠的环节
    （会顺手"润色"掉原文表述）。内容层面的活儿前面五个角色已经干完，
    这里只做确定性装配。若确实需要模型润色，请在 write_frame 里做，
    不要放在最后一步。
    """
    outline = state.get("outline") or {}
    title = str(outline.get("title") or state.get("topic") or "深度研究报告")
    chapters = sorted(state.get("chapters") or [],
                      key=lambda c: c.get("index") or 0)
    over = _over_budget(state)
    stopped = bool(state.get("stopped"))
    has_frame = bool(state.get("toc") or state.get("introduction")
                     or state.get("conclusion"))

    if has_frame:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        parts = [f"# {title}", "", f"**日期**：{date}",
                 f"**课题**：{str(state.get('topic') or '')[:300]}",
                 f"**执行模式**：{'快速（未经审稿）' if str(state.get('mode') or 'full') == 'quick' else '完整（含审稿闭环）'}",
                 "---", ""]
        if state.get("toc"):
            parts += ["## 目录", "", str(state["toc"]), "", "---", ""]
        if state.get("introduction"):
            parts += ["## 引言", "", str(state["introduction"]), "", "---", ""]
        for c in chapters:
            parts += [f"## {c.get('index')}. {c.get('title', '')}", "",
                      str(c.get("content", "")), "", "---", ""]
        if state.get("conclusion"):
            parts += ["## 结论", "", str(state["conclusion"]), "", "---", ""]
        pool = state.get("source_pool") or []
        if pool:
            parts += ["## 参考文献", ""]
            parts += [f"- [{s.get('title', '未命名')}]({s.get('url', '')})"
                      for s in pool]
            parts += ["", "---", ""]
        warns = state.get("warnings") or []
        if warns or over or stopped:
            parts += ["## 待完善事项", ""]
            if stopped:
                parts.append("- 本场被用户中途停止，后续章节未调研")
            if over:
                parts.append("- token 预算耗尽，后续章节被熔断截断")
            for w in warns:
                parts.append(f"- 第{w.get('chapter')}章（{w.get('kind')}）："
                             f"{w.get('detail')}")
            parts += ["", "---", ""]
        parts += ["> 本报告由 AI 深度调研团队生成，重要决策请经专业人员核验。",
                  "> 所有引用来源请在重要场景下二次核验时效性与真实性。"]
        report = "\n".join(parts)
    else:
        report = _fallback_report(state)
        if over or stopped:
            report += "\n\n---\n\n## 待完善事项\n\n"
            if stopped:
                report += "- 本场被用户中途停止\n"
            if over:
                report += "- token 预算耗尽，后续章节被熔断截断\n"

    refs = [{"title": s.get("title", ""), "url": s.get("url", "")}
            for s in (state.get("source_pool") or [])]
    emit("dr_report", {"chars": len(report), "chapters": len(chapters),
                       "sources": len(refs), "fallback": not has_frame,
                       "budget_exhausted": over})
    # 成稿正文必须由 publish 亲自推流：前两支团队的成稿面板（含"导出 .md"）
    # 只在收到带 content 的 plan 事件时渲染——漏发这条的话直播全程看不到
    # 报告与下载入口，必须刷新走详情回放才出现（真实 bug）。
    # fallback/reason 复用前端 plan 事件的降级提示语（停止/预算熔断）。
    emit("dr_plan", {"content": report,
                     **({"fallback": True,
                         "reason": "stopped" if stopped else "budget_exhausted"}
                        if (not has_frame and (stopped or over)) else {})})
    out: dict = {"final_report": report, "references": refs}
    if over:
        out["budget_exhausted"] = True
    if stopped:
        clear_stop(state["session_id"])
    return out


# ---------- 图装配 ----------

async def build_deep_research_graph(ctx: WorkflowContext):
    """编译深度调研子图（独立 checkpointer，与主图/争鸣社/会讲各自持有连接）。"""
    from app.graph.workflow import _build_checkpointer

    builder = StateGraph(DeepResearchState)

    builder.add_node("scout", make_scout_node(ctx))
    builder.add_node("plan", make_plan_node(ctx))
    builder.add_node("research", make_research_node(ctx))
    builder.add_node("merge_sources", merge_sources_node)      # 纯代码
    builder.add_node("review", make_review_node(ctx))
    builder.add_node("revise", make_revise_node(ctx))
    builder.add_node("archive", archive_chapter_node)          # 纯代码
    builder.add_node("write_frame", make_write_frame_node(ctx))
    builder.add_node("publish", publish_node)                  # 纯代码

    builder.add_edge(START, "scout")
    builder.add_conditional_edges("scout", route_after_scout,
                                  {"plan": "plan", "publish": "publish"})
    builder.add_conditional_edges("plan", route_after_plan,
                                  {"research": "research",
                                   "publish": "publish"})

    # 章节主循环入口
    builder.add_edge("research", "merge_sources")
    builder.add_conditional_edges("merge_sources", route_after_merge,
                                  {"review": "review", "archive": "archive",
                                   "publish": "publish"})
    # 审稿 ⇄ 修订 内层循环
    builder.add_conditional_edges("review", route_after_review,
                                  {"revise": "revise", "archive": "archive"})
    builder.add_edge("revise", "review")
    # 归档 → 下一章 / 收尾
    builder.add_conditional_edges("archive", route_after_archive,
                                  {"research": "research",
                                   "write_frame": "write_frame",
                                   "publish": "publish"})
    builder.add_edge("write_frame", "publish")
    builder.add_edge("publish", END)

    saver = await _build_checkpointer(ctx.settings.database_url)
    return builder.compile(checkpointer=saver)

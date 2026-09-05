"""格致会讲子图测试：假模型全流程 + 议程确定性 + 聚合数学 + 停止。"""

import asyncio
import json
import math

from langchain_core.messages import AIMessage

from app.graph.seminar import (_SCORE_WEIGHTS, _stdev,
                               build_seminar_graph, aggregate_node)
from app.graph.nodes import WorkflowContext
from app.services.kb_service import KBService
from app.settings import Settings

_loop = asyncio.new_event_loop()          # 常驻事件循环（checkpoint 绑定）
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


CURATE_JSON = json.dumps({
    "field_positioning": "RAG评测可靠性位于IR与ML评测交叉",
    "adjacent_field": "心理测量学（信度理论可直接迁移）",
    "reading_assignments": {"historian": "梳理演化", "theorist": "盘点框架",
                            "experimentalist": "评估基准", "visitor": "迁移视角"},
    "ideation_directives": {"historian": "复活线索", "theorist": "公理化空白",
                            "experimentalist": "新基准", "visitor": "结构平移"},
}, ensure_ascii=False)

INSIGHT_JSON = json.dumps({
    "insights": [{"kind": "gap", "statement": "评测缺乏信度检验",
                  "origin": "experimentalist 汇报"}],
    "open_questions": ["信度理论能否适配开放生成任务？"],
}, ensure_ascii=False)

MERGE_JSON = json.dumps({"merged_cards": [
    {"card_id": "C1", "author": "visitor", "title": "信度理论迁移到RAG评测",
     "inspiration": "洞见#1", "hypothesis": "H", "validation_sketch": "V",
     "risk": "R", "merged_from": []},
]}, ensure_ascii=False)

SCORE_JSON = json.dumps([
    {"card_id": "C1", "novelty": 8, "significance": 7,
     "feasibility": 5, "risk_control": 4, "comment": "好"},
], ensure_ascii=False)

PROPOSE_JSON = json.dumps([
    {"title": "跨领域构想", "inspiration": "洞见#1", "hypothesis": "H",
     "validation_sketch": "V", "risk": "R"},
], ensure_ascii=False)


class SeminarFakeModel:
    """替身模型：按 system 提示词里的 JSON 字段名分发。
    字段名在真实提示词正文里都有（任务 1.2 的设计），测试与生产同构。"""

    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages, **kwargs):
        system = str(next((m.content for m in messages
                           if getattr(m, "type", "") == "system"), ""))
        if '"reading_assignments"' in system:
            return AIMessage(content=CURATE_JSON)
        if '"insights"' in system:
            return AIMessage(content=INSIGHT_JSON)
        if '"merged_cards"' in system:
            return AIMessage(content=MERGE_JSON)
        if '"novelty"' in system:
            return AIMessage(content=SCORE_JSON)
        if '"builds_on"' in system:          # propose/improve 都给卡片
            return AIMessage(content=PROPOSE_JSON)
        return AIMessage(content="领域事实陈述：现有基准缺乏信度检验。")

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class FakeSemLLMService:
    def __init__(self):
        self._model = SeminarFakeModel()

    def get_chat_model(self, user_id, temperature=None):
        return self._model


def _make_ctx():
    import tempfile
    from pathlib import Path
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    return WorkflowContext(settings, FakeSemLLMService(), KBService(settings))


def _sem_initial(user_id, sid, topic, **extra):
    return {
        "user_id": user_id, "session_id": sid, "topic": topic,
        "roles": [
            {"id": "historian", "name": "文献学家", "temperature": 0.3},
            {"id": "theorist", "name": "理论家", "temperature": 0.6},
            {"id": "experimentalist", "name": "实验家", "temperature": 0.4},
            {"id": "visitor", "name": "访问学者", "temperature": 0.9},
        ],
        "token_budget": 20_000_000,
        "agenda_index": 0, "token_budget_used": 0,
        "reading_notes": [], "evidence_pool": [], "qa_transcript": [],
        "insight_board": [], "open_questions": [], "idea_cards": [],
        "review_scores": [], "stopped": False, **extra,
    }


def test_seminar_full_run():
    """全流程：开题→4路研读→4场汇报问答→构想→评审→聚合→成稿。"""
    graph = _run(build_seminar_graph(_make_ctx()))
    result = _run(graph.ainvoke(
        _sem_initial("u1", "sem-t1", "RAG评测的可靠性提升"),
        config={"configurable": {"thread_id": "sem-t1"},
                "recursion_limit": 200}))
    assert len(result["reading_notes"]) == 4        # 四路并行汇合
    presents = [e for e in result["qa_transcript"] if e.get("kind") == "present"]
    asks = [e for e in result["qa_transcript"] if e.get("kind") == "ask"]
    answers = [e for e in result["qa_transcript"] if e.get("kind") == "answer"]
    assert len(presents) == 4                       # 议程确定：每位学者一场
    assert len(asks) == 4 * 3 and len(answers) == 4 * 3   # 每场其余3人各一问
    assert result["insight_board"]                   # 洞见池在增长
    assert result["final_proposal"]                  # 成稿非空
    assert result["idea_ranking"]                    # 聚合有输出


def test_aggregate_math():
    """聚合纯代码：均分/加权/分歧度/排序可精确断言（无 LLM）。"""
    state = {
        "card_registry": [{"card_id": "C1", "author": "visitor", "title": "t"}],
        "review_scores": [
            {"card_id": "C1", "reviewer": "a", "novelty": 8, "significance": 7,
             "feasibility": 5, "risk_control": 4, "comment": ""},
            {"card_id": "C1", "reviewer": "b", "novelty": 4, "significance": 5,
             "feasibility": 9, "risk_control": 6, "comment": ""},
        ],
    }
    out = aggregate_node(state)
    r = out["idea_ranking"][0]
    assert r["reviewers"] == 2
    assert r["dims"]["novelty"] == 6.0               # (8+4)/2
    w = _SCORE_WEIGHTS
    expect_a = (8 * w["novelty"] + 7 * w["significance"]
                + 5 * w["feasibility"] + 4 * w["risk_control"])
    expect_b = (4 * w["novelty"] + 5 * w["significance"]
                + 9 * w["feasibility"] + 6 * w["risk_control"])
    assert r["total"] == round((expect_a + expect_b) / 2, 2)
    expect_div = math.sqrt(((expect_a - (expect_a + expect_b) / 2) ** 2
                            + (expect_b - (expect_a + expect_b) / 2) ** 2) / 2)
    assert r["divergence"] == round(expect_div, 2)


def test_seminar_stop_produces_partial_result():
    """用户停止：研读截断，rapporteur 仍产出汇编降级稿。"""
    from app.core.cancel import request_stop
    graph = _run(build_seminar_graph(_make_ctx()))
    request_stop("sem-t2")
    result = _run(graph.ainvoke(
        _sem_initial("u1", "sem-t2", "议题"),
        config={"configurable": {"thread_id": "sem-t2"}, "recursion_limit": 200}))
    assert result["stopped"] is True
    assert result["final_proposal"]                # 汇编稿仍在


def test_seminar_budget_exhausted_skips_to_rapporteur():
    """预算熔断：跳过后续阶段直达 rapporteur（执笔人不受预算阻断）。"""
    graph = _run(build_seminar_graph(_make_ctx()))
    result = _run(graph.ainvoke(
        _sem_initial("u1", "sem-t3", "议题", token_budget_used=999_999_999),
        config={"configurable": {"thread_id": "sem-t3"}, "recursion_limit": 200}))
    presents = [e for e in result.get("qa_transcript", [])
                if e.get("kind") == "present"]
    assert len(presents) < 4                       # 议程被截断
    assert result["final_proposal"]                # 执笔人照常执行
    assert result.get("budget_exhausted") is True


def test_seminar_stop_during_propose_no_invalid_update():
    """回归（§5.2 纪律）：停止恰好落在 propose 并行段——四个学者节点同时
    返回 stopped 曾把单值 channel 打爆（InvalidUpdateError）。并行节点
    不得写单值字段，停止语义只走 is_stopped 注册表。"""
    from app.core.cancel import request_stop, clear_stop

    class StopOnProposeModel(SeminarFakeModel):
        async def astream(self, messages):
            system = str(next((m.content for m in messages
                               if getattr(m, "type", "") == "system"), ""))
            if "构想工作坊" in system:          # PROPOSE_SYSTEM 的标志串
                request_stop("sem-t4")          # 模拟用户在构想阶段点停止
            for c in super().astream(messages):
                yield c

    class Service(FakeSemLLMService):
        def __init__(self):
            self._model = StopOnProposeModel()

        def get_chat_model(self, user_id, temperature=None):
            return self._model

    base = _make_ctx()
    ctx = WorkflowContext(base.settings, Service(), base.kb_service)
    graph = _run(build_seminar_graph(ctx))
    clear_stop("sem-t4")
    result = _run(graph.ainvoke(
        _sem_initial("u1", "sem-t4", "议题"),
        config={"configurable": {"thread_id": "sem-t4"}, "recursion_limit": 200}))
    assert result["stopped"] is True              # rapporteur 收尾置位
    assert result["final_proposal"]               # 汇编降级稿仍在

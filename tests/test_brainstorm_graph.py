"""头脑风暴子图测试：假模型全流程 + 主持人降级 + 停止 + 预算熔断。

Fake 模型按 system 提示词关键词分发（镜像 test_graph.RouterAwareFakeModel
的做法）：PREPARE_PROMPT 含 "research_directives"、MODERATOR_PROMPT 含
"next_speaker"、其余按发言返回文本。
"""

import asyncio
import json

from langchain_core.messages import AIMessage

from app.graph.brainstorm import build_brainstorm_graph
from app.graph.nodes import WorkflowContext
from app.services.kb_service import KBService
from app.settings import Settings

# 常驻事件循环（AsyncPostgresSaver 绑定循环，同 test_graph）
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


PREP_JSON = json.dumps({
    "topic_restatement": "关于RAG评测可靠性的研究",
    "sub_questions": ["如何构造金标准"],
    "research_directives": {"innovator": "找交叉点", "critic": "找争议",
                            "methodologist": "找协议", "practitioner": "找数据集"},
}, ensure_ascii=False)

MOD_JSON = json.dumps({
    "next_speaker": "critic", "focus": "评估数据可得性",
    "consensus_level": 5, "consensus": [], "divergence": ["评测集构造方式"],
}, ensure_ascii=False)


class BrainstormFakeModel:
    """替身模型：按 system 提示词区分 prepare/moderator/发言。"""

    def __init__(self, moderator_json: str = MOD_JSON, captured=None):
        self._moderator_json = moderator_json
        self._captured = captured

    def bind_tools(self, schemas):
        return self                     # 吞掉工具绑定，不发工具调用

    async def ainvoke(self, messages, **kwargs):
        system = next((m.content for m in messages
                       if getattr(m, "type", "") == "system"), "")
        if self._captured is not None:
            self._captured.append(str(system))
        if "research_directives" in str(system):
            return AIMessage(content=PREP_JSON)
        if "next_speaker" in str(system):
            return AIMessage(content=self._moderator_json)
        return AIMessage(content="我认为应当从评测协议入手。@方法论专家 你的基线设计能覆盖吗？")

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class FakeBSLLMService:
    def __init__(self, moderator_json=MOD_JSON, captured=None):
        self._model = BrainstormFakeModel(moderator_json, captured)

    def get_chat_model(self, user_id, temperature=None):
        return self._model


def _make_ctx(moderator_json=MOD_JSON, captured=None):
    import tempfile
    from pathlib import Path
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    ctx = WorkflowContext(settings, FakeBSLLMService(moderator_json, captured),
                          KBService(settings))
    return ctx


def _bs_initial(user_id, sid, topic, **extra):
    return {
        "user_id": user_id, "session_id": sid, "topic": topic,
        "roles": [
            {"id": "innovator", "name": "创新者", "temperature": 0.9},
            {"id": "critic", "name": "批评者", "temperature": 0.4},
            {"id": "methodologist", "name": "方法论专家", "temperature": 0.3},
            {"id": "practitioner", "name": "实践者", "temperature": 0.5},
        ],
        "max_rounds": 2, "turn_count": 0, "token_budget_used": 0,
        "positions": [], "evidence_pool": [], "transcript": [],
        "stopped": False, **extra,
    }


def test_brainstorm_full_run():
    """全流程：prepare → 4路并行调研 → 辩论循环 → 成稿。"""
    ctx = _make_ctx()
    graph = _run(build_brainstorm_graph(ctx))
    result = _run(graph.ainvoke(
        _bs_initial("u1", "bs-t1", "如何提升RAG评测的可靠性"),
        config={"configurable": {"thread_id": "bs-t1"},
                "recursion_limit": 200}))     # 辩论循环步数多，必须放大
    assert len(result["positions"]) == 4            # 四份立场书并行汇合
    assert all(not p["failed"] for p in result["positions"])
    assert result["transcript"]                     # 辩论发生过
    assert result["final_proposal"]                 # 成稿非空
    # transcript 里的发言全部来自主持人点名的 critic（MOD_JSON 固定点 critic）
    assert all(t["agent_id"] == "critic" for t in result["transcript"])


def test_brainstorm_moderator_invalid_falls_back_round_robin():
    """主持人点名不合法（点到不存在的人）→ round-robin 顺序兜底。"""
    bad = json.dumps({"next_speaker": "hacker", "focus": "",
                      "consensus_level": 0, "consensus": [], "divergence": ["x"]},
                     ensure_ascii=False)
    ctx = _make_ctx(moderator_json=bad)
    graph = _run(build_brainstorm_graph(ctx))
    result = _run(graph.ainvoke(
        _bs_initial("u1", "bs-t2", "议题"),
        config={"configurable": {"thread_id": "bs-t2"}, "recursion_limit": 200}))
    speakers = [t["agent_id"] for t in result["transcript"]]
    # max_rounds=2 × 4人 = 8 次发言，顺序轮转
    assert speakers == ["innovator", "critic", "methodologist", "practitioner"] * 2


def test_brainstorm_budget_exhausted_skips_debate():
    """token 预算超限 → 熔断停止辩论（零发言），撰稿人不受预算约束、照常成稿。

    撰稿成本相对调研+辩论可忽略；若被熔断跳过，前期投入就全浪费了
    （预算熔断只砍辩论，不砍撰稿）。"""
    ctx = _make_ctx()
    graph = _run(build_brainstorm_graph(ctx))
    result = _run(graph.ainvoke(
        _bs_initial("u1", "bs-t3", "议题", token_budget_used=999_999_999),
        config={"configurable": {"thread_id": "bs-t3"}, "recursion_limit": 200}))
    assert result.get("transcript", []) == []       # 辩论被熔断跳过
    assert result.get("budget_exhausted") is True   # stats 截断标注
    assert result["final_proposal"].strip()         # 撰稿人照常执行
    assert "汇编" not in result["final_proposal"]    # 不是降级汇编稿


def test_brainstorm_stop_produces_partial_result():
    """用户停止：调研/辩论截断，synthesis 仍产出汇编稿（部分成果保留）。

    停止 ≠ 失败：被截断的角色标"已停止，调研中止"，不计为缺席失败。
    """
    from app.core.cancel import request_stop
    ctx = _make_ctx()
    graph = _run(build_brainstorm_graph(ctx))
    request_stop("bs-t4")                           # 预先置停止标记
    result = _run(graph.ainvoke(
        _bs_initial("u1", "bs-t4", "议题"),
        config={"configurable": {"thread_id": "bs-t4"}, "recursion_limit": 200}))
    assert result["stopped"] is True
    assert result["final_proposal"]                  # 汇编稿仍在
    assert all(not p["failed"] for p in result["positions"])
    assert all("已停止" in p["content"] for p in result["positions"])


def test_role_cfg_prefers_role_config_then_falls_back():
    """角色独立模型：role.config_id 命中 → 用该配置；配置已删除/非本人
    /未配置 → 回退用户生效配置。"""
    from app.graph.brainstorm import _role_cfg

    class CfgSvc:
        def get_config(self, uid, cid):
            return "ROLE_CFG" if cid == "c1" else None   # c1 有，c2 无

        def effective_config(self, uid):
            return "EFF"

    ctx = WorkflowContext(Settings.load(), CfgSvc(), None)
    state = {"user_id": "u1",
             "roles": [{"id": "critic", "config_id": "c1"},
                       {"id": "innovator", "config_id": "gone"},
                       {"id": "practitioner"}]}
    assert _role_cfg(ctx, state, "critic") == "ROLE_CFG"       # 角色配置命中
    assert _role_cfg(ctx, state, "innovator") == "EFF"         # 已删 → 回退
    assert _role_cfg(ctx, state, "practitioner") == "EFF"      # 未配置 → 回退


def test_role_cfg_model_override_same_provider():
    """同供应商不同模型：config_id 出凭证（key/base_url），model_id 覆盖模型；
    未选配置时覆盖作用于用户生效配置。"""
    from dataclasses import replace as dc_replace

    from app.abstractions.llm import LLMConfig
    from app.graph.brainstorm import _role_cfg

    BASE = LLMConfig(provider="custom", base_url="http://x/v1",
                     model_id="model-a", api_key="sk-x")

    class CfgSvc:
        def get_config(self, uid, cid):
            return BASE if cid == "c1" else None

        def effective_config(self, uid):
            return BASE

    ctx = WorkflowContext(Settings.load(), CfgSvc(), None)
    # 配置出凭证 + 覆盖模型
    state = {"user_id": "u1",
             "roles": [{"id": "critic", "config_id": "c1", "model_id": "model-b"}]}
    cfg = _role_cfg(ctx, state, "critic")
    assert cfg.model_id == "model-b"                    # 模型已覆盖
    assert cfg.base_url == BASE.base_url and cfg.api_key == "sk-x"  # 凭证继承
    assert cfg.model_id != BASE.model_id and BASE.model_id == "model-a"  # 原配置不可变

    # 未选配置：覆盖作用于生效配置
    state2 = {"user_id": "u1", "roles": [{"id": "innovator", "model_id": "model-c"}]}
    assert _role_cfg(ctx, state2, "innovator").model_id == "model-c"

    # 覆盖是不可变副本，不影响原配置对象
    assert BASE.model_id == "model-a"


def test_brainstorm_finish_before_full_round_is_ignored():
    """主持人第一轮就输出 finish → 忽略并 round-robin 辩满一轮后再收场。

    教程收敛门槛只防住了 consensus_level 路径；显式 finish 同样必须
    满足 turn >= len(ids)，否则辩论零发言直接成稿（真实事故模式）。
    """
    early = json.dumps({"next_speaker": "finish", "focus": "",
                        "consensus_level": 9, "consensus": ["x"],
                        "divergence": []}, ensure_ascii=False)
    ctx = _make_ctx(moderator_json=early)
    graph = _run(build_brainstorm_graph(ctx))
    result = _run(graph.ainvoke(
        _bs_initial("u1", "bs-t5", "议题"),
        config={"configurable": {"thread_id": "bs-t5"}, "recursion_limit": 200}))
    speakers = [t["agent_id"] for t in result["transcript"]]
    # 第一轮 finish 被忽略：4 个角色各发言一次（顺序轮转）后才允许收场
    assert speakers == ["innovator", "critic", "methodologist", "practitioner"]


def test_agent_speak_retries_empty_completion():
    """空完成防御（与主图 generate 同语义）：一次"零内容零工具调用"的
    空完成不再让整角色缺席——退避重试后恢复正常输出。"""
    from langchain_core.messages import AIMessageChunk

    from app.graph.brainstorm import _agent_speak

    class FlakyModel:
        def __init__(self):
            self.n = 0

        def bind_tools(self, schemas):
            return self

        async def astream(self, msgs):
            self.n += 1
            if self.n == 1:
                yield AIMessageChunk(content="")     # 空完成（供应商偶发）
            else:
                yield AIMessageChunk(content="恢复了")

    class Svc:
        def __init__(self):
            self.m = FlakyModel()

        def get_chat_model(self, uid, temperature=None):
            return self.m

    svc = Svc()
    ctx = WorkflowContext(Settings.load(), svc, None)
    state = {"user_id": "u1", "session_id": "bs-empty-1",
             "roles": [{"id": "critic"}]}
    content, used, stopped = _run(_agent_speak(
        ctx, state, "critic", "系统提示", "用户输入", 0.5, None, 3))
    assert content == "恢复了" and stopped is False
    assert svc.m.n == 2                                 # 空完成被重试吸收


def test_research_emits_actual_model(monkeypatch):
    """气泡起始事件携带该角色实际解析到的模型名——配置的模型与所选模型
    不一致（如网关透传上游报错）时用户可直接看到真相。"""
    import queue
    from types import SimpleNamespace

    import app.graph.brainstorm as bs
    from app.core.events import clear_event_sink, set_event_sink
    from app.services.kb_service import KBService

    class Svc:
        def effective_config(self, uid):
            return SimpleNamespace(model_id="deepseek-v4-flash",
                                   api_key="k", base_url="http://t/v1")

    captured = []
    q = queue.Queue()
    set_event_sink(q)
    try:
        ctx = WorkflowContext(Settings.load(), Svc(), KBService(Settings.load()))
        monkeypatch.setattr(bs, "_agent_speak",
                            async_stub := (lambda *a, **k: _async_return("内容", 10, False)))
        node = bs.make_research_node(ctx, "critic")
        state = {"user_id": "u1", "session_id": "bs-model-1",
                 "topic": "议题", "roles": [{"id": "critic"}],
                 "research_directives": {}}
        _run(node(state))
    finally:
        clear_event_sink()
    while not q.empty():
        captured.append(q.get_nowait())
    starts = [e for e in captured if e["type"] == "bs_agent_start"]
    assert starts and starts[0]["model"] == "deepseek-v4-flash"
    ends = [e for e in captured if e["type"] == "bs_agent_end"]
    assert ends and ends[0]["failed"] is False


def _async_return(*args):
    async def _inner():
        return args
    return _inner()


def test_agent_speak_forces_final_round_when_tool_budget_exhausted():
    """工具预算耗尽仍想调工具 → 追加一轮无工具调用强制产出。

    事故复盘：deepseek 风格"先调工具后写结论"的模型会把 6 轮预算全部
    花在工具上，循环结束时正文为空——被误标成"调研失败，本角色缺席"
    （批评者每次必现）。强制收尾轮后必然产出立场书。
    """
    from langchain_core.messages import AIMessageChunk

    from app.graph.brainstorm import _agent_speak

    class ToolHungryModel:
        """前 2 轮只想调工具（不写正文），之后才肯输出结论。"""

        def __init__(self):
            self.rounds = 0

        def bind_tools(self, schemas):
            return self

        async def astream(self, msgs):
            self.rounds += 1
            if self.rounds <= 2:
                yield AIMessageChunk(content="", tool_calls=[
                    {"name": "arxiv_search", "args": {"query": "x"},
                     "id": f"c{self.rounds}"}])
            else:
                yield AIMessageChunk(content="基于以上调研，我的结论是……")

    class Svc:
        def __init__(self):
            self.m = ToolHungryModel()

        def get_chat_model(self, uid, temperature=None):
            return self.m

    svc = Svc()
    ctx = WorkflowContext(Settings.load(), svc, None)
    state = {"user_id": "u1", "session_id": "bs-force-1",
             "roles": [{"id": "critic"}]}
    content, used, stopped = _run(_agent_speak(
        ctx, state, "critic", "系统提示", "用户输入", 0.5, None, 2))
    assert "结论" in content and stopped is False
    assert svc.m.rounds == 3                        # 2 轮工具 + 1 轮强制收尾


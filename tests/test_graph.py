import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.core.db import SessionLocal
from app.graph.nodes import WorkflowContext
from app.graph.workflow import build_graph
from app.models import KnowledgeBase, User
from app.services.kb_service import KBService
from app.settings import Settings


def _ensure_user(user_id: str):
    """kbs.owner_user_id 有外键指向 users，私人库测试前先建用户。"""
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()

# 关键：AsyncPostgresSaver 的连接绑定在构建它的事件循环上。
# asyncio.run() 每次新建并销毁循环，会导致"bound to a different event loop"。
# 所以全测试共享一个常驻循环。
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


class RouterAwareFakeModel:
    """替身模型：按系统提示词区分调用类型。

    - 系统提示词含「问答路由」→ 返回 route_json（模拟 LLM 意图判断）
    - 其他（生成 / 记忆抽取）→ 返回 answer
    """

    def __init__(self, answer: str, route_json: str, captured=None):
        self._answer = answer
        self._route_json = route_json
        self._captured = captured                 # 记录收到的 system 提示词（断言用）

    def bind_tools(self, schemas):
        return self

    async def ainvoke(self, messages):
        system = next((m.content for m in messages
                       if getattr(m, "type", "") == "system"), "")
        if self._captured is not None:
            self._captured.append(str(system))
        if "问答路由" in str(system):
            return AIMessage(content=self._route_json)
        return AIMessage(content=self._answer)

    async def astream(self, messages):
        yield await self.ainvoke(messages)


class FakeLLMService:
    """替身 LLM 服务：路由/生成按提示词分别返回预置内容。"""

    def __init__(self, answer: str, route_json: str, captured=None):
        self._answer = answer
        self._route_json = route_json
        self._captured = captured

    def get_chat_model(self, user_id: str, temperature=None):
        return RouterAwareFakeModel(self._answer, self._route_json,
                                    self._captured)


def _make_ctx(answer: str = "这是假模型回答",
              route_json: str = '{"needs_retrieval": true, "kbs": []}',
              captured=None):
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",   # 离线嵌入，测试确定性
    })
    kb_service = KBService(settings)
    ctx = WorkflowContext(settings, FakeLLMService(answer, route_json, captured),
                          kb_service)
    return ctx, kb_service


def _run_graph(graph, state: dict):
    return _run(graph.ainvoke(
        state, config={"configurable": {"thread_id": state["session_id"]}}))


def test_retrieve_chain_with_kb():
    """有知识库 + LLM 判断需要检索 → 按选中的库检索，结果带 scope 标签。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "测试库", "scope": "public"}]}')
    kb_service.create_kb("测试库", "public", "u1",
                         ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t1",
                                "query": "叠加态是什么",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert len(result["retrievals"]) == 1
    assert result["retrievals"][0]["scope"] == "public"
    assert result["retrievals"][0]["kb_name"] == "测试库"
    assert result["answer"] == "答案"


def test_llm_skips_retrieval_for_chitchat():
    """有知识库，但 LLM 判断无需检索（闲聊）→ 不检索直接生成。"""
    ctx, kb_service = _make_ctx(
        "你好呀", '{"needs_retrieval": false, "kbs": []}')
    kb_service.create_kb("测试库", "public", "u1",
                         ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t5",
                                "query": "你好",
                                "messages": [HumanMessage(content="你好")]})
    assert result.get("retrievals", []) == []
    assert result["answer"] == "你好呀"


def test_llm_selects_only_chosen_kbs():
    """LLM 只选公共库 → 私人库不被检索（不再全库覆盖）。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "公共文档", "scope": "public"}]}')
    _ensure_user("u1")
    kb_service.create_kb("公共文档", "public", "u1", ["公共资料：量子比特叠加态"])
    kb_service.create_kb("我的实验", "private", "u1", ["私人实验：pH=7.2"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t6",
                                "query": "叠加态",
                                "messages": [HumanMessage(content="叠加态")]})
    assert result["retrievals"]
    assert all(r["kb_name"] == "公共文档" for r in result["retrievals"])


def test_route_failure_falls_back_to_all_visible():
    """路由输出无法解析 → 降级为全部可见库检索（RAG 兜底）。"""
    ctx, kb_service = _make_ctx("答案", "抱歉，我无法理解你的意思")
    kb_service.create_kb("测试库", "public", "u1",
                         ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t7",
                                "query": "叠加态是什么",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert len(result["retrievals"]) == 1

def test_no_kb_skips_retrieve():
    """没有知识库 → supervisor 直接走 generate，不检索。"""
    ctx, _ = _make_ctx("无库回答")
    graph = _run(build_graph(ctx))
    result = _run_graph(graph, {"user_id": "u1", "session_id": "t2",
                                "query": "你好",
                                "messages": [HumanMessage(content="你好")]})
    assert result.get("retrievals", []) == []
    assert result["answer"] == "无库回答"


def test_private_kb_not_visible_to_others():
    """u1 的私人库对 u2 不可见（越权防护在图层生效）。"""
    ctx, kb_service = _make_ctx("答案")
    _ensure_user("u1")
    kb_service.create_kb("u1私密", "private", "u1", ["我的实验pH=7.2"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u2", "session_id": "t3",
                                "query": "实验参数",
                                "messages": [HumanMessage(content="实验参数")]})
    assert result.get("retrievals", []) == []   # u2 什么都查不到


def test_retrieve_with_mode_vector():
    """retrieval_mode=vector：retrieve_node 走纯向量，结果无 BM25 分数。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "测试库", "scope": "public"}]}')
    kb_service.create_kb("测试库", "public", "u1", ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-mode-1",
                                "query": "叠加态是什么",
                                "retrieval_mode": "vector",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert result["retrievals"]
    assert all(r.get("bm25_score") is None for r in result["retrievals"])
    assert all(r.get("distance") is not None for r in result["retrievals"])


def test_retrieve_with_mode_hybrid():
    """retrieval_mode=hybrid：retrieve_node 结果带 BM25 分数与融合分。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "测试库", "scope": "public"}]}')
    kb_service.create_kb("测试库", "public", "u1", ["量子比特可以处于叠加态"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-mode-2",
                                "query": "叠加态是什么",
                                "retrieval_mode": "hybrid",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert result["retrievals"]
    assert all(r.get("score") is not None for r in result["retrievals"])


def test_retrieve_counts_parameterized():
    """per_kb_k / total_k 生效：两库各 1 条，per_kb_k=1 + total_k=1 → 只取 1 条。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "库A", "scope": "public"}, {"name": "库B", "scope": "public"}]}')
    kb_service.create_kb("库A", "public", "u1", ["量子比特可以处于叠加态A"])
    kb_service.create_kb("库B", "public", "u1", ["蛋白质折叠预测B"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-k-1",
                                "query": "量子比特",
                                "per_kb_k": 1, "total_k": 1,
                                "messages": [HumanMessage(content="量子比特")]})
    assert len(result["retrievals"]) == 1


def test_retrieve_counts_default_from_settings():
    """不传 per_kb_k/total_k（0）→ 走全局配置默认（yaml：每库 3 / 总共 5）。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "库A", "scope": "public"}]}')
    kb_service.create_kb("库A", "public", "u1", ["量子比特可以处于叠加态A"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-k-2",
                                "query": "量子比特",
                                "per_kb_k": 0, "total_k": 0,
                                "messages": [HumanMessage(content="量子比特")]})
    assert len(result["retrievals"]) == 1     # 单库单条：默认值下正常返回


def test_retrieve_counts_clamped():
    """越界值被约束不崩溃：per_kb_k=999/total_k=999 → clamp 后正常返回。"""
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "库A", "scope": "public"}]}')
    kb_service.create_kb("库A", "public", "u1", ["量子比特可以处于叠加态A"])
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-k-3",
                                "query": "量子比特",
                                "per_kb_k": 999, "total_k": 999,
                                "messages": [HumanMessage(content="量子比特")]})
    assert len(result["retrievals"]) == 1


def test_parent_blocks_expanded_in_retrieval():
    """聚合返回：同组多命中 → 展开父块（hit_chunks 累计、完整段落），其余小 chunk。"""
    from conftest import make_pdf_pages as _pdf

    ctx, kb_service = _make_ctx()
    kb = kb_service.create_kb("聚合库", "public", None)
    kb_service.ingest_file(kb.kb_id, "paper.pdf",
                           _pdf(["Quantum computing fundamentals with superposition states. " * 60]))
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-parent-1",
                                "query": "Quantum computing",
                                "per_kb_k": 10, "total_k": 10, "parent_groups": 3,
                                "messages": [HumanMessage(content="Quantum computing")]})
    rets = result["retrievals"]
    parents = [r for r in rets if r.get("type") == "parent"]
    chunks = [r for r in rets if r.get("type") != "parent"]
    assert parents, "应当有父块展开"
    assert parents[0].get("source") == "paper.pdf"
    assert parents[0].get("hit_chunks", 1) >= 1
    assert len({(p.get("doc_id"), p.get("group")) for p in parents}) <= 3
    assert len(parents[0]["text"]) > 500               # 父块是完整段落
    assert all(c.get("type") == "chunk" for c in chunks)


def test_parent_groups_zero_disables_aggregation():
    """parent_groups=0：完全退化为小 chunk（不展开父块）。"""
    from conftest import make_pdf_pages as _pdf

    ctx, kb_service = _make_ctx()
    kb = kb_service.create_kb("退化库", "public", None)
    kb_service.ingest_file(kb.kb_id, "paper.pdf",
                           _pdf(["Quantum computing fundamentals with superposition states. " * 60]))
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-parent-0",
                                "query": "Quantum computing",
                                "per_kb_k": 10, "total_k": 10, "parent_groups": 0,
                                "messages": [HumanMessage(content="Quantum computing")]})
    rets = result["retrievals"]
    assert rets and all(r.get("type") == "chunk" for r in rets)


def test_kb_description_passed_to_router():
    """知识库介绍随选库目录传给 LLM：路由提示词包含 description。"""
    captured = []
    ctx, kb_service = _make_ctx(
        "答案", '{"needs_retrieval": true, "kbs": [{"name": "量子库", "scope": "public"}]}',
        captured=captured)
    kb_service.create_kb("量子库", "public", "u1", ["量子比特可以处于叠加态"],
                         description="收录量子计算与密码学论文，适合回答量子算法问题")
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-desc-1",
                                "query": "叠加态是什么",
                                "messages": [HumanMessage(content="叠加态是什么")]})
    assert result["answer"] == "答案"
    route_prompt = next(p for p in captured if "问答路由" in p)
    assert "量子库" in route_prompt
    assert "收录量子计算与密码学论文" in route_prompt      # description 进了提示词


def test_retrieve_skips_broken_kb(auth_factory):
    """P1-5 单库隔离：坏库（嵌入维度不匹配）被跳过，健康库照常出结果。

    场景还原真实事故路径：入库后用户改了库的嵌入配置（维度变化），
    该库查询必然在 Chroma 层崩溃——改造前会拖垮整轮对话，现在只损失
    这一个库的贡献，并推 retrieve_error 事件供前端提示。
    """
    ctx, kb_service = _make_ctx(
        "答案",
        '{"needs_retrieval": true, "kbs": ['
        '{"name": "好库", "scope": "public"}, {"name": "坏库", "scope": "public"}]}')
    kb_service.create_kb("好库", "public", "u1",
                         ["量子比特可以处于叠加态"], description="量子")
    bad = kb_service.create_kb("坏库", "public", "u1",
                               ["深度学习需要大量数据"], description="dl")
    with SessionLocal() as db:
        row = db.get(KnowledgeBase, bad.kb_id)
        row.embedding_dim += 1                  # 模拟"入库后改了维度配置"
        db.commit()
    graph = _run(build_graph(ctx))

    result = _run_graph(graph, {"user_id": "u1", "session_id": "t-dim-iso",
                                "query": "叠加态是什么"})
    assert result["answer"] == "答案"            # 整轮对话正常完成
    srcs = {r.get("kb_name") for r in result.get("retrievals", [])}
    assert "好库" in srcs                       # 健康库贡献照常
    assert "坏库" not in srcs                   # 坏库被隔离跳过


def test_kb_description_required_by_api(auth_factory):
    """API 建库：介绍为空 → 400；带介绍 → 落库并回显。"""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        h = auth_factory()
        r1 = c.post("/api/v1/kbs", headers=h, json={
            "name": "无介绍库", "scope": "public",
            "description": "   ", "embedding_provider": "local"})
        assert r1.status_code == 400                          # 空白介绍被拒
        r2 = c.post("/api/v1/kbs", headers=h, json={
            "name": "有介绍库", "scope": "public",
            "description": "深度学习论文合集", "embedding_provider": "local"})
        assert r2.status_code == 200
        assert r2.json()["description"] == "深度学习论文合集"


def test_checkpointer_continues_session():
    """同 thread_id 两次调用 → 消息累积（短期记忆/会话级隔离）。"""
    ctx, _ = _make_ctx("回答")
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "same-session"}}

    _run(graph.ainvoke({"user_id": "u1", "session_id": "x",
                        "query": "第一问",
                        "messages": [HumanMessage(content="第一问")]}, config=cfg))
    result = _run(graph.ainvoke({"user_id": "u1", "session_id": "x",
                                 "query": "第二问",
                                 "messages": [HumanMessage(content="第二问")]},
                                config=cfg))
    texts = [m.content for m in result["messages"]]
    assert "第一问" in texts and "第二问" in texts   # 历史消息都在


def test_fork_with_history_initial_state():
    """分支会话：history 随首次请求传入，初始 messages = history + 新提问。"""
    from app.api.chat import ChatRequest, _initial_state
    req = ChatRequest(session_id="branch-1", message="继续",
                      history=[
                          {"role": "user", "content": "第一问"},
                          {"role": "assistant", "content": "第一答"},
                          {"role": "user", "content": "分支点提问"},
                      ])
    state = _initial_state(req, "u1")
    assert [m["role"] for m in state["messages"]] == ["user", "assistant", "user", "user"]
    assert state["messages"][0]["content"] == "第一问"
    assert state["messages"][-1]["content"] == "继续"
    assert state["retrievals"] == [] and state["needs_retrieval"] is False


def test_fork_history_runs_graph():
    """分支历史真实跑图：新 thread 从 history 起步，历史 + 新提问 + 回答完整累积。"""
    from app.api.chat import ChatRequest, _initial_state
    ctx, _ = _make_ctx("分支回答")
    graph = _run(build_graph(ctx))
    req = ChatRequest(session_id="branch-2", message="继续",
                      history=[{"role": "user", "content": "分支点提问"}])
    result = _run(graph.ainvoke(
        _initial_state(req, "u1"),
        config={"configurable": {"thread_id": "branch-2"}}))
    assert result["answer"] == "分支回答"
    texts = [m.content for m in result["messages"]]
    assert texts == ["分支点提问", "继续", "分支回答"]   # 历史 + 新提问 + 回答


def test_rewind_regenerates_last_answer():
    """重新生成：回退旧回复 → 重生成，不残留旧回答、不重复提问。"""
    from app.api.chat import ChatRequest, _initial_state, _rewind_to_last_user

    cfg = {"configurable": {"thread_id": "rewind-1"}}
    # 第一轮：提问 → 旧回答
    ctx1, _ = _make_ctx("旧回答")
    graph1 = _run(build_graph(ctx1))
    req1 = ChatRequest(session_id="rewind-1", message="1+1等于几")
    r1 = _run(graph1.ainvoke(_initial_state(req1, "u1"), config=cfg))
    assert r1["answer"] == "旧回答"
    assert [m.content for m in r1["messages"]] == ["1+1等于几", "旧回答"]

    # 回退：checkpoint 截断到最后一条用户消息
    assert _run(_rewind_to_last_user(graph1, cfg)) is True
    snap = _run(graph1.aget_state(cfg))
    assert [m.content for m in snap.values["messages"]] == ["1+1等于几"]

    # 重新生成（同一 thread，不追加消息）：新回答，无旧回答残留、无重复提问
    ctx2, _ = _make_ctx("新回答")
    graph2 = _run(build_graph(ctx2))
    req2 = ChatRequest(session_id="rewind-1", message="1+1等于几")
    result = _run(graph2.ainvoke(_initial_state(req2, "u1", append_message=False),
                                 config=cfg))
    assert result["answer"] == "新回答"
    assert [m.content for m in result["messages"]] == ["1+1等于几", "新回答"]


def test_rewind_on_empty_thread_returns_false():
    """新 thread 无 checkpoint 历史：无可回退，返回 False（端点据此退化为追加消息）。"""
    from app.api.chat import _rewind_to_last_user
    ctx, _ = _make_ctx("x")
    graph = _run(build_graph(ctx))
    cfg = {"configurable": {"thread_id": "brand-new-thread"}}
    assert _run(_rewind_to_last_user(graph, cfg)) is False
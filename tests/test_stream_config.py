"""离线模型流与图配置契约：内存 checkpoint，不使用业务数据库。"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.abstractions.llm import LLMConfig, LLMFactory, RetryableChatModel
from app.abstractions.reasoning import ReasoningChatOpenAI
from app.core.events import clear_event_sink, set_event_sink
from app.graph import agent_runtime, workflow
from app.graph.brainstorm import make_agent_node
from app.graph.bs_state import BrainstormState
from app.graph.nodes import WorkflowContext, _langchain_route_round
from app.graph.state import AgentState


@pytest.fixture(autouse=True)
def clean_db():
    """覆盖仓库清库 fixture；本模块仅使用内存图与离线 HTTP 响应。"""
    yield


class ModelProbe(BaseCallbackHandler):
    def __init__(self):
        self.starts = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.starts.append(kwargs)


def _offline_model(answer="标准答案", reasoning="先核对配置", captured=None):
    """使用真实 ChatOpenAI 转换和回调，只替换最底层 HTTP transport。"""
    def response(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        if captured is not None:
            captured.append(body)
        wire = []
        for delta, finish in [
            ({"role": "assistant", "reasoning_content": reasoning}, None),
            ({"content": answer}, None),
            ({}, "stop"),
        ]:
            chunk = {
                "id": "chatcmpl-config", "object": "chat.completion.chunk",
                "created": 0, "model": "offline-reasoning",
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}],
            }
            wire.append("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n")
        wire.append("data: [DONE]\n\n")
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content="".join(wire).encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    model = ReasoningChatOpenAI(
        model="offline-reasoning", api_key="offline-key",
        base_url="https://offline.invalid/v1", http_async_client=client,
        streaming=True, stream_usage=True, max_retries=0, http_socket_options=())
    return model, client


def _config(sid, probe):
    return {"configurable": {"thread_id": sid}, "callbacks": [probe],
            "tags": ["parent-tag"], "metadata": {"request_marker": "kept"}}


def test_json_route_preserves_parameters_without_repeating_reasoning():
    async def scenario():
        captured = []
        answer = '{"needs_retrieval":false,"kbs":[]}'
        model, client = _offline_model(answer, "路由思考", captured)
        sink = asyncio.Queue()
        set_event_sink(sink)
        try:
            result = await _langchain_route_round(
                RetryableChatModel(model, max_retries=0),
                LLMConfig("custom", "https://offline.invalid/v1", "offline-reasoning", "key"),
                "判定是否检索", [HumanMessage(content="你好")], 0.3, "json-route",
                config={"metadata": {"request_marker": "route"}})
            events = []
            while not sink.empty():
                events.append(sink.get_nowait())
            assert json.loads(result) == {"needs_retrieval": False, "kbs": []}
            assert captured[0]["response_format"] == {"type": "json_object"}
            assert captured[0]["reasoning_effort"] == "low"
            assert [event["content"] for event in events
                    if event["type"] == "routing_reasoning"] == ["路由思考"]
        finally:
            clear_event_sink()
            await client.aclose()

    asyncio.run(scenario())


def _assert_parent_context(probe):
    assert probe.starts, "模型应通过标准回调报告调用，而不是绕过模型层"
    for start in probe.starts:
        assert "parent-tag" in start["tags"]
        assert start["metadata"]["request_marker"] == "kept"


def test_workflow_messages_preserve_reasoning_only_chunks_and_callbacks(monkeypatch):
    async def scenario():
        model, client = _offline_model()
        try:
            async def checkpointer(_database_url):
                return InMemorySaver()

            async def no_side_effects(ctx, state, config=None):
                return {}

            async def no_retrieval(ctx, state, config=None):
                return {"needs_retrieval": False}

            monkeypatch.setattr(workflow, "_build_checkpointer", checkpointer)
            monkeypatch.setattr(workflow, "load_memory_node", no_side_effects)
            monkeypatch.setattr(workflow, "compact_node", no_side_effects)
            monkeypatch.setattr(workflow, "supervisor_node", no_retrieval)
            service = SimpleNamespace(
                get_chat_model=lambda *_args, **_kwargs: RetryableChatModel(
                    model, max_retries=0))
            ctx = WorkflowContext(SimpleNamespace(database_url="unused"),
                                  service, SimpleNamespace())
            graph = await workflow.build_graph(ctx)
            probe = ModelProbe()
            config = _config("workflow-config", probe)
            state = {"user_id": "offline-user", "session_id": "workflow-config",
                     "messages": [HumanMessage(content="检查流式配置")],
                     "query": "检查流式配置", "memory": {}, "retrievals": []}
            events = [event async for event in graph.astream(
                state, config=config, stream_mode="messages")]
            reasoning_chunks = [(chunk, metadata) for chunk, metadata in events
                                if chunk.additional_kwargs.get("reasoning_content")]
            assert reasoning_chunks
            assert reasoning_chunks[0][0].content == ""
            assert reasoning_chunks[0][0].additional_kwargs[
                "reasoning_content"] == "先核对配置"
            assert reasoning_chunks[0][1]["langgraph_node"] == "generate"
            assert any(chunk.content == "标准答案" for chunk, _ in events)
            _assert_parent_context(probe)
            snapshot = await graph.aget_state(config)
            assert snapshot.values["answer"] == "标准答案"
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_team_node_stream_keeps_role_model_and_role_metadata(monkeypatch):
    async def scenario():
        model, client = _offline_model(answer="角色发言")
        try:
            default = LLMConfig("offline", "https://offline.invalid/v1",
                                "default-model", "default-key")
            saved = LLMConfig("offline", "https://offline.invalid/v1",
                              "saved-model", "saved-key")
            built = []

            def build(cfg):
                built.append(cfg)
                return model

            def get_config(user_id, config_id):
                assert (user_id, config_id) == ("offline-user", "saved-config")
                return saved

            monkeypatch.setattr(LLMFactory, "build", staticmethod(build))
            service = SimpleNamespace(effective_config=lambda _uid: default,
                                      get_config=get_config)
            ctx = WorkflowContext(SimpleNamespace(), service, SimpleNamespace())
            builder = StateGraph(BrainstormState)
            builder.add_node("agent_innovator", make_agent_node(ctx, "innovator"))
            builder.add_edge(START, "agent_innovator")
            builder.add_edge("agent_innovator", END)
            graph = builder.compile(checkpointer=InMemorySaver())
            state = {"user_id": "offline-user", "session_id": "role-config",
                     "topic": "配置传递", "turn_count": 1,
                     "roles": [{"id": "innovator", "name": "创新者",
                                "temperature": 0.4, "config_id": "saved-config",
                                "model_id": "role-override"}],
                     "positions": [], "evidence_pool": [], "transcript": [],
                     "moderator_notes": {}, "token_budget_used": 0}
            probe = ModelProbe()
            config = _config("role-config", probe)
            events = [event async for event in graph.astream(
                state, config=config, stream_mode="messages")]
            assert events
            assert built and built[0].model_id == "role-override"
            assert built[0].api_key == "saved-key"
            for _, metadata in events:
                assert metadata["team"] == "debate"
                assert metadata["agent"] == "innovator"
                assert "team:debate" in metadata["tags"]
                assert "agent:innovator" in metadata["tags"]
            _assert_parent_context(probe)
            snapshot = await graph.aget_state(config)
            assert snapshot.values["transcript"][-1]["content"] == "角色发言"
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_control_model_remains_observable_without_leaking_messages():
    async def scenario():
        model, client = _offline_model(answer="内部判定", reasoning="内部推理")
        try:
            service = SimpleNamespace(
                get_chat_model=lambda *_args, **_kwargs: model)
            ctx = WorkflowContext(SimpleNamespace(), service, SimpleNamespace())

            async def control(state, config: RunnableConfig):
                text, _used = await agent_runtime.llm_text(
                    ctx, state, "请作内部判定", "判断是否继续",
                    prefix="sem_", agent_id="chair", config=config)
                return {"answer": text}

            builder = StateGraph(AgentState)
            builder.add_node("control", control)
            builder.add_edge(START, "control")
            builder.add_edge("control", END)
            graph = builder.compile(checkpointer=InMemorySaver())
            probe = ModelProbe()
            config = _config("control-config", probe)
            events = [event async for event in graph.astream(
                {"user_id": "offline-user", "session_id": "control-config"},
                config=config, stream_mode="messages")]
            assert events == [], "nostream 控制调用不可泄露到用户消息流"
            _assert_parent_context(probe)
            assert all("nostream" in start["tags"] for start in probe.starts)
            assert probe.starts[0]["metadata"]["agent"] == "chair"
            snapshot = await graph.aget_state(config)
            assert snapshot.values["answer"] == "内部判定"
        finally:
            await client.aclose()

    asyncio.run(scenario())

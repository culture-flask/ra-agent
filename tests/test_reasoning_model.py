"""推理兼容模型与 Runnable 的离线契约，不访问网络或数据库。"""

import asyncio
from copy import deepcopy

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai.types.chat import ChatCompletion
from pydantic import Field

from app.abstractions.llm import LLMConfig, LLMFactory, RetryableChatModel
from app.abstractions.reasoning import RAW_USAGE_KEY, ReasoningChatOpenAI


def _model(**kwargs):
    return ReasoningChatOpenAI(
        model="offline-reasoner", api_key="sk-test",
        base_url="https://example.invalid/v1", use_responses_api=False,
        **kwargs)


def _chunk(delta, **kwargs):
    return {"choices": [{"index": 0, "delta": delta, **kwargs}]}


@pytest.mark.parametrize("field,value,expected", [
    ("reasoning_content", "先分析", "先分析"),
    ("reasoning", "换一种方言", "换一种方言"),
    ("reasoning", [{"text": "一"}, {"summary": "二"}], "一二"),
    ("reasoning_details", [{"type": "reasoning.text", "text": "摘要"},
                           {"type": "reasoning.encrypted", "data": "opaque"}], "摘要"),
])
def test_stream_preserves_reasoning_dialects_and_arrays(field, value, expected):
    wire = _chunk({"role": "assistant", "content": "答案", field: value})
    original = deepcopy(wire)
    generation = _model()._convert_chunk_to_generation_chunk(wire, AIMessageChunk, {})
    assert generation.message.content == "答案"
    assert generation.message.additional_kwargs["reasoning_content"] == expected
    assert wire == original


def test_reasoning_only_chunk_is_not_dropped():
    generation = _model()._convert_chunk_to_generation_chunk(
        _chunk({"role": "assistant", "reasoning_content": "尚未输出正文"}),
        AIMessageChunk, {"headers": {"x-request-id": "offline"}})
    assert isinstance(generation.message, AIMessageChunk)
    assert generation.message.content == ""
    assert generation.message.additional_kwargs["reasoning_content"] == "尚未输出正文"
    assert generation.generation_info["headers"]["x-request-id"] == "offline"


def test_usage_only_tail_preserves_authoritative_snapshot_and_normalizes_total():
    wire = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                                     "prompt_tokens_details": {"cached_tokens": 4}}}
    original = deepcopy(wire)
    generation = _model()._convert_chunk_to_generation_chunk(wire, AIMessageChunk, {})
    assert generation.message.usage_metadata == {
        "input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
        "input_token_details": {"cache_read": 4}, "output_token_details": {}}
    assert generation.message.response_metadata[RAW_USAGE_KEY]["total_tokens"] == 10
    assert generation.message.content == ""
    assert wire == original


def test_none_delta_retains_finish_reason_and_usage():
    wire = {**_chunk(None, finish_reason="stop"), "model": "offline-reasoner",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
    generation = _model()._convert_chunk_to_generation_chunk(wire, AIMessageChunk, {})
    assert generation.generation_info["finish_reason"] == "stop"
    assert generation.generation_info["model_name"] == "offline-reasoner"
    assert generation.message.usage_metadata["total_tokens"] == 3
    assert wire["choices"][0]["delta"] is None


def test_reasoning_conversion_preserves_tool_call_chunks_and_merging():
    model = _model()
    first = model._convert_chunk_to_generation_chunk(_chunk({
        "role": "assistant", "reasoning": "需要检索",
        "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query":'}}],
    }), AIMessageChunk, {})
    second = model._convert_chunk_to_generation_chunk(_chunk({
        "tool_calls": [{"index": 0, "function": {"arguments": '"RAG"}'}}],
    }, finish_reason="tool_calls"), AIMessageChunk, {})
    assert first.message.tool_call_chunks[0]["id"] == "call_1"
    assert first.message.additional_kwargs["reasoning_content"] == "需要检索"
    combined = first.message + second.message
    assert combined.tool_calls == [{"name": "lookup", "args": {"query": "RAG"},
                                    "id": "call_1", "type": "tool_call"}]
    assert second.generation_info["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("sdk_response", [False, True])
def test_nonstream_reasoning_preserves_tools_finish_and_usage(sdk_response):
    wire = {"id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
            "model": "offline-reasoner", "choices": [{"index": 0,
                "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "查一下",
                    "reasoning": [{"text": "需要外部证据"}],
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "lookup",
                                                 "arguments": '{"query":"RAG"}'}}]}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}
    original = deepcopy(wire)
    response = ChatCompletion.model_validate(wire) if sdk_response else wire
    result = _model()._create_chat_result(response)
    generation = result.generations[0]
    assert generation.message.content == "查一下"
    assert generation.message.additional_kwargs["reasoning_content"] == "需要外部证据"
    assert generation.message.tool_calls[0]["args"] == {"query": "RAG"}
    assert generation.generation_info["finish_reason"] == "tool_calls"
    assert generation.message.usage_metadata["total_tokens"] == 6
    assert wire == original


@pytest.mark.parametrize("replay", [False, True])
def test_request_replays_reasoning_only_when_explicitly_enabled(replay):
    assistant = AIMessage(content="检查结果", additional_kwargs={
        "reasoning_content": "展示用推理", "reasoning": "另一方言"},
        tool_calls=[{"name": "lookup", "args": {"query": "RAG"}, "id": "call_1"}])
    payload = _model(replay_reasoning=replay)._get_request_payload(
        [HumanMessage(content="问题"), assistant])
    encoded = payload["messages"][1]
    assert encoded["content"] == "检查结果"
    assert encoded["tool_calls"][0]["function"]["name"] == "lookup"
    assert "reasoning" not in encoded and "reasoning_details" not in encoded
    if replay:
        assert encoded["reasoning_content"] == "展示用推理"
    else:
        assert "reasoning_content" not in encoded


def test_factory_returns_reasoning_compatible_standard_chat_model():
    model = LLMFactory.build(LLMConfig(
        provider="custom", base_url="https://example.invalid/v1",
        model_id="offline-reasoner", api_key="sk-test"))
    assert isinstance(model, ReasoningChatOpenAI)
    assert isinstance(model, ChatOpenAI)
    assert isinstance(model, BaseChatModel)
    assert isinstance(model, Runnable)
    assert model.use_responses_api is False
    assert model.stream_usage is True
    assert model.http_socket_options == ()
    assert model.request_timeout == 180


class _OfflineChatModel(BaseChatModel):
    seen: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-contract"

    def bind_tools(self, schemas, **kwargs):
        return self.bind(tools=schemas, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(kwargs)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))


class _RecordingCallbacks(BaseCallbackHandler):
    def __init__(self):
        self.starts = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.starts.append(kwargs)


def test_retry_wrapper_retains_bind_options_with_config_callbacks_and_batch():
    underlying = _OfflineChatModel()
    callbacks = _RecordingCallbacks()
    model = RetryableChatModel(underlying, max_retries=0, label="offline")
    assert isinstance(model, Runnable)
    bound = model.bind_tools([{"name": "lookup"}], tool_choice="auto",
                             strict=True, parallel_tool_calls=False).with_config({
        "callbacks": [callbacks], "tags": ["contract"],
        "metadata": {"probe": "preserved"}})
    responses = bound.batch([[HumanMessage(content="a")], [HumanMessage(content="b")]])
    assert [response.content for response in responses] == ["ok", "ok"]
    assert len(callbacks.starts) == 2
    assert all("contract" in event["tags"]
               and event["metadata"]["probe"] == "preserved" for event in callbacks.starts)
    assert all(call["tools"] == [{"name": "lookup"}] and call["strict"] is True
               and call["parallel_tool_calls"] is False and call["tool_choice"] == "auto"
               for call in underlying.seen)


def test_retry_wrapper_supports_standard_async_stream_events():
    async def collect():
        model = RetryableChatModel(_OfflineChatModel(), max_retries=0)
        return [event async for event in model.astream_events(
            [HumanMessage(content="question")], version="v2",
            config={"tags": ["stream-contract"], "metadata": {"probe": "events"}})]

    events = asyncio.run(collect())
    streams = [event for event in events if event["event"] == "on_chat_model_stream"]
    assert streams and streams[0]["data"]["chunk"].content == "ok"
    assert "stream-contract" in streams[0]["tags"]
    assert streams[0]["metadata"]["probe"] == "events"
    assert any(event["event"] == "on_chat_model_end" for event in events)

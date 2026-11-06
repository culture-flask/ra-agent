"""OpenAI 兼容端点的窄适配：保留扩展推理字段，复用 ChatOpenAI 全部转换。

兼容钩子按 requirements.txt 中的 langchain-openai 1.2.2 固定；流式、
非流式、工具分片和 usage 的契约由离线响应测试覆盖。
"""

from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI


REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_details")
RAW_USAGE_KEY = "ra_usage"
_stream_scope: ContextVar[AsyncExitStack | None] = ContextVar("model_stream_scope", default=None)


@asynccontextmanager
async def model_stream_scope():
    """一次尝试拥有的模型流；停止时不依赖嵌套异步生成器的延迟回收。"""
    async with AsyncExitStack() as stack:
        token = _stream_scope.set(stack)
        try:
            yield
        finally:
            _stream_scope.reset(token)


def _own_stream(iterator):
    stack = _stream_scope.get()
    if stack is not None:
        stack.push_async_callback(iterator.aclose)
    return iterator


def reasoning_text(value: Any) -> str:
    """规范化可展示文本；加密推理或其他非文本数据不当成文本输出。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(reasoning_text(part) for part in value)
    if isinstance(value, dict):
        for key in ("text", "summary", "reasoning"):
            if value.get(key):
                return reasoning_text(value[key])
    return ""


def reasoning_from_delta(delta: dict) -> str:
    for key in REASONING_FIELDS:
        text = reasoning_text(delta.get(key))
        if text:
            return text
    return ""


def reasoning_from_message(message) -> str:
    text = reasoning_from_delta(getattr(message, "additional_kwargs", {}) or {})
    if text:
        return text
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return "".join(
            reasoning_text(block.get("reasoning") or block.get("summary")
                           or block.get("thinking"))
            for block in content if isinstance(block, dict)
            and block.get("type") in ("reasoning", "thinking")
        )
    return ""


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "") for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "output_text")
        )
    return ""


def _normalized_usage(usage: dict) -> dict:
    """补齐兼容端点可能省略的 total_tokens，不修改服务端原对象。"""
    normalized = dict(usage)
    normalized.setdefault("prompt_tokens", 0)
    normalized.setdefault("completion_tokens", 0)
    normalized.setdefault("total_tokens", normalized["prompt_tokens"]
                          + normalized["completion_tokens"])
    return normalized


class ReasoningChatOpenAI(ChatOpenAI):
    """仅扩展兼容协议字段；工具、回调、usage 和标准 Runnable 行为继承父类。"""

    # 仅显式选择的端点回传推理。普通 UI 推理不自动变成下一轮请求参数。
    replay_reasoning: bool = False

    async def astream(self, input, config=None, *, stop=None, **kwargs):
        iterator = _own_stream(super().astream(input, config=config, stop=stop, **kwargs))
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    async def _astream(self, *args, **kwargs):
        # 沿用 ChatOpenAI 的协议分支；直接管理实际持有 HTTP 上下文的父类流。
        # 标准 Runnable / ChatModel 的多层 async-for 转发不会逐层即时 aclose。
        method = (BaseChatOpenAI._astream_responses
                  if self._use_responses_api({**kwargs, **self.model_kwargs})
                  else BaseChatOpenAI._astream)
        iterator = _own_stream(method(self, *args, **kwargs))
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        wire = dict(chunk)
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices and choices[0].get("delta") is None:
            # 某些兼容网关只在空 delta 的末块报告 finish_reason。
            wire["choices"] = [{**choice, "delta": choice.get("delta") or {}}
                               for choice in choices]
        if chunk.get("usage"):
            wire["usage"] = _normalized_usage(chunk["usage"])
        generation = super()._convert_chunk_to_generation_chunk(
            wire, default_chunk_class, base_generation_info)
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return generation
        if choices:
            text = reasoning_from_delta(choices[0].get("delta") or {})
            if text:
                generation.message.additional_kwargs["reasoning_content"] = text
        if wire.get("usage"):
            # 保留本分片的权威快照，防重复累计快照被 AIMessageChunk.__add__ 求和。
            generation.message.response_metadata[RAW_USAGE_KEY] = wire["usage"]
        return generation

    def _create_chat_result(self, response, generation_info=None):
        wire = response if isinstance(response, dict) else response.model_dump()
        wire = dict(wire)
        if wire.get("usage"):
            wire["usage"] = _normalized_usage(wire["usage"])
        result = super()._create_chat_result(wire, generation_info)
        for generation, choice in zip(result.generations, wire.get("choices") or []):
            text = reasoning_from_delta(choice.get("message") or {})
            if text:
                generation.message.additional_kwargs["reasoning_content"] = text
        return result

    def _get_generation_chunk_from_completion(self, completion):
        generation = super()._get_generation_chunk_from_completion(completion)
        # SDK JSON 流末尾还会报告一次完整响应；推理已逐片发出，不再重复追加。
        for key in REASONING_FIELDS:
            generation.message.additional_kwargs.pop(key, None)
        return generation

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.replay_reasoning and "messages" in payload:
            messages = self._convert_input(input_).to_messages()
            for message, encoded in zip(messages, payload["messages"]):
                if isinstance(message, AIMessage):
                    text = reasoning_from_message(message)
                    if text:
                        encoded["reasoning_content"] = text
        return payload

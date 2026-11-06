"""标准 LangChain 模型流协调器（保留 native_round 导入名以兼容调用方）。

模型协议转换、工具参数分片与回调由 ChatModel 负责。本模块只协调请求内
SSE、停止、末块用量、失败尝试撤回和截断恢复，不直接调用 OpenAI SDK。
"""

import asyncio
import uuid

from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message
from langchain_core.runnables.config import merge_configs

from app.abstractions.llm import LLMConfig, LLMFactory, _is_retryable, model_call
from app.abstractions.reasoning import (
    RAW_USAGE_KEY, REASONING_FIELDS, message_text, model_stream_scope,
    reasoning_from_message,
)
from app.core.cancel import is_stopped
from app.core.events import emit
from app.core.logging import get_logger

logger = get_logger("model_stream")
_zero_reasoning_seen: set[str] = set()
_TOOL_MARKUP_MARK = "<｜｜DSML｜｜"


def retry_event_name(token_event: str) -> str:
    for prefix in ("bs_", "sem_", "dr_"):
        if token_event.startswith(prefix):
            return prefix + "retry"
    return "retry"


def reset_event_name(token_event: str) -> str:
    for prefix in ("bs_", "sem_", "dr_"):
        if token_event.startswith(prefix):
            return prefix + "stream_reset"
    return "stream_reset"


def log_zero_reasoning_once(cfg) -> None:
    if cfg is None:
        return
    key = f"{getattr(cfg, 'base_url', '')}::{getattr(cfg, 'model_id', '')}"
    if key not in _zero_reasoning_seen:
        _zero_reasoning_seen.add(key)
        logger.info("端点未返回推理文本 base=%s model=%s",
                    getattr(cfg, "base_url", ""), getattr(cfg, "model_id", ""))


def strip_tool_markup(text: str) -> str:
    index = text.find(_TOOL_MARKUP_MARK)
    return text[:index].rstrip() if index >= 0 else text


def leaked_tool_name(text: str) -> str:
    return text.partition('invoke name="')[2].partition('"')[0]


def _usage_from_chunk(chunk) -> dict | None:
    metadata = getattr(chunk, "response_metadata", {}) or {}
    wire = metadata.get(RAW_USAGE_KEY) or metadata.get("token_usage")
    if wire:
        inp = int(wire.get("prompt_tokens") or 0)
        out = int(wire.get("completion_tokens") or 0)
        details = wire.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or wire.get("prompt_cache_hit_tokens") or 0)
    else:
        usage = getattr(chunk, "usage_metadata", None) or {}
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details") or {}
        cached = int(details.get("cache_read") or 0)
    if inp + out <= 0:
        return None
    return {"input_tokens": inp, "output_tokens": out,
            "total_tokens": inp + out, "cached_tokens": cached}


def _message_chunk(chunk) -> AIMessageChunk:
    if isinstance(chunk, AIMessageChunk):
        result = chunk.model_copy(deep=True)
    elif isinstance(chunk, AIMessage):
        result = AIMessageChunk(**chunk.model_dump(exclude={"type"}))
    else:
        # 轻量离线替身仍通过标准消息聚合，不另造工具参数拼接逻辑。
        result = AIMessageChunk(
            content=getattr(chunk, "content", "") or "",
            tool_calls=getattr(chunk, "tool_calls", None) or [],
            additional_kwargs=getattr(chunk, "additional_kwargs", None) or {},
            response_metadata=getattr(chunk, "response_metadata", None) or {},
            usage_metadata=getattr(chunk, "usage_metadata", None),
        )
    result.response_metadata.pop(RAW_USAGE_KEY, None)
    # 每块可能带同一份累计快照，权威用量在协调器单独取最后一块。
    result.usage_metadata = None
    return result


def _final_message(aggregate, usage, stopped: bool, usage_metadata=None) -> AIMessage:
    response = (message_chunk_to_message(aggregate) if aggregate is not None
                else AIMessage(content=""))
    response = response.model_copy(deep=True)
    # 保持迁移前语义：可展示推理仅流向 UI，不写入普通消息 checkpoint。
    response.additional_kwargs = {
        key: value for key, value in response.additional_kwargs.items()
        if key not in REASONING_FIELDS
    }
    if isinstance(response.content, list):
        response.content = [block for block in response.content
                            if not isinstance(block, dict)
                            or block.get("type") not in ("reasoning", "thinking")]
    text = message_text(response)
    cleaned = strip_tool_markup(text)
    if cleaned != text:
        logger.warning("工具模板泄漏已剥离 tool=%s", leaked_tool_name(text) or "未知")
        response.content = cleaned
    if stopped:
        response.tool_calls = []
        response.invalid_tool_calls = []
    if usage:
        response.usage_metadata = {
            **(usage_metadata or {}),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "input_token_details": {
                **((usage_metadata or {}).get("input_token_details") or {}),
                "cache_read": usage["cached_tokens"],
            },
        }
    return response


async def _chunks(model, payload, config):
    async with model_stream_scope():
        if callable(getattr(model, "astream", None)):
            iterator = model_call(model, "astream", payload, config)
            try:
                async for chunk in iterator:
                    yield chunk
            finally:
                close = getattr(iterator, "aclose", None)
                if callable(close):
                    await close()
        else:
            # 控制类测试替身可能只实现 ainvoke。
            yield await model_call(model, "ainvoke", payload, config)


async def native_round(cfg, payload, tools, temperature, sid,
                       token_event: str = "token",
                       reasoning_event: str = "reasoning",
                       emit_extra: dict | None = None,
                       stream_events: bool = True,
                       emit_retry: bool = False,
                       config=None, model=None):
    """一轮标准模型流；每次重试重建消息聚合器，撤回失败尝试的 UI 增量。

    返回 (AIMessage, 最后权威 usage | None, stopped)。cfg=None 时必须给
    model；调用名为历史兼容保留，内部没有 SDK/工具 JSON 的手写转换。
    """
    if model is None:
        if cfg is None:
            raise ValueError("model or cfg is required")
        effective = LLMConfig(
            provider=getattr(cfg, "provider", "custom"),
            base_url=cfg.base_url, model_id=cfg.model_id,
            api_key=cfg.api_key, temperature=temperature,
        )
        model = LLMFactory.build(effective)
    max_retries = int(getattr(model, "retry_max_retries", 5))
    base_delay = float(getattr(model, "retry_base_delay", 1.0))
    if tools:
        model = model.bind_tools(tools)
    if callable(getattr(model, "bind", None)) and temperature is not None:
        model = model.bind(temperature=temperature)

    extra = emit_extra or {}
    truncation_retried = False
    previous_attempt_id = None
    attempt_id = uuid.uuid4().hex
    for attempt in range(max_retries + 1):
        aggregate = None
        last_usage = None
        last_usage_metadata = None
        seen_finish = False
        had_reasoning = False
        stopped = False
        token_chars = reasoning_chars = 0
        call_config = merge_configs(config, {"metadata": {"attempt_id": attempt_id}})
        try:
            stream = _chunks(model, payload, call_config)
            try:
                async for chunk in stream:
                    if is_stopped(sid):
                        stopped = True
                        break
                    usage = _usage_from_chunk(chunk)
                    if usage is not None:
                        last_usage = usage
                        last_usage_metadata = getattr(chunk, "usage_metadata", None)
                    metadata = getattr(chunk, "response_metadata", {}) or {}
                    seen_finish = seen_finish or bool(metadata.get("finish_reason"))
                    reasoning = reasoning_from_message(chunk)
                    text = message_text(chunk)
                    had_reasoning = had_reasoning or bool(reasoning)
                    if stream_events:
                        if reasoning:
                            emit(reasoning_event, {"content": reasoning, **extra,
                                                   "attempt_id": attempt_id})
                            reasoning_chars += len(reasoning)
                        if text:
                            emit(token_event, {"content": text, **extra,
                                               "attempt_id": attempt_id})
                            token_chars += len(text)
                    piece = _message_chunk(chunk)
                    aggregate = piece if aggregate is None else aggregate + piece
            finally:
                await stream.aclose()
            # 仅对真实兼容端点做缺结束标记且缺用量的截断探测；假模型没有协议末块。
            truncated = cfg is not None and not seen_finish and last_usage is None
            if not stopped and truncated and not truncation_retried and attempt < max_retries:
                truncation_retried = True
                retry_error = "响应被网关截断，整轮重试"
                delay = 0
            else:
                if not had_reasoning and not stopped:
                    log_zero_reasoning_once(cfg)
                return (_final_message(aggregate, last_usage, stopped, last_usage_metadata),
                        last_usage, stopped)
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            retry_error = str(exc)[:160]
            delay = base_delay * (2 ** attempt)
            logger.warning("model stream retry %d/%d: %s", attempt + 1, max_retries, exc)

        previous_attempt_id, attempt_id = attempt_id, uuid.uuid4().hex
        if stream_events and (token_chars or reasoning_chars):
            emit(reset_event_name(token_event), {
                **extra, "attempt_id": attempt_id,
                "previous_attempt_id": previous_attempt_id,
                "token_event": token_event, "reasoning_event": reasoning_event,
                "token_chars": token_chars, "reasoning_chars": reasoning_chars,
            })
        if stream_events or emit_retry:
            emit(retry_event_name(token_event), {
                "attempt": attempt + 1, "max": max_retries,
                "delay": round(delay, 1), "error": retry_error, **extra,
                "attempt_id": attempt_id,
            })
        if delay:
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")

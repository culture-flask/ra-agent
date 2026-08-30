"""原生 openai SDK 流式调用公共模块：主对话图与头脑风暴子图共用。

从 graph/nodes.py 抽出（纯搬家 + 事件参数化）：
- to_openai_messages / to_openai_tools：LangChain 消息与工具 schema 的协议转换
- usage_from_native：末块 usage → 内部用量格式（cached_tokens 三方言兼容）
- native_round：一轮流式调用（思考捕获、末块权威用量、截断检测重试、
  网络层指数退避）

事件参数化（与原 _native_round 的唯一行为差异，默认值保持原语义）：
- token_event / reasoning_event：事件名（主图默认 token/reasoning，
  头脑风暴传 bs_token/bs_reasoning）
- emit_extra：并入每个事件的附加字段（如 {"agent": role_id}）
- stream_events=False：控制类调用（路由/主持人判定）不向前端透传
"""

import json

from langchain_core.messages import AIMessage

from app.abstractions.llm import _is_retryable
from app.core.cancel import is_stopped
from app.core.events import emit
from app.core.logging import get_logger

logger = get_logger("native_llm")


def to_openai_messages(payload: list) -> list[dict]:
    """LangChain 消息序列 → 原生 openai SDK 的 Chat Completions 格式。

    相邻 user 消息合并为一条（检索块以 HumanMessage 插在问题之后的
    "资料附在问题末尾"单条 user 形态，思考最少）；system 只保留队首一条。
    """
    out: list[dict] = []
    for m in payload:
        role = getattr(m, "type", None)
        content = m.content
        if isinstance(content, list):      # 内容块列表 → 拼文本
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict))
        if role == "system":
            out.append({"role": "system", "content": content or ""})
        elif role == "human":
            if out and out[-1]["role"] == "user":   # 相邻 user → 并入上一条
                out[-1]["content"] = (out[-1]["content"] + "\n\n"
                                      + (content or ""))
            else:
                out.append({"role": "user", "content": content or ""})
        elif role == "ai":
            item: dict = {"role": "assistant", "content": content or ""}
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                item["tool_calls"] = [{
                    "id": tc.get("id", ""), "type": "function",
                    "function": {"name": tc.get("name", ""),
                                 "arguments": json.dumps(tc.get("args") or {},
                                                         ensure_ascii=False)},
                } for tc in tcs]
            out.append(item)
        elif role == "tool":
            out.append({"role": "tool", "content": content or "",
                        "tool_call_id": getattr(m, "tool_call_id", "")})
    return out


def to_openai_tools(schemas: list[dict]) -> list[dict]:
    """schemas_for_llm 的混合格式（裸 function dict 或已包 type/function）
    → 原生 SDK 的 tools 参数格式。"""
    out: list[dict] = []
    for s in schemas:
        if s.get("type") == "function" and isinstance(s.get("function"), dict):
            out.append(s)
        else:
            out.append({"type": "function",
                        "function": {"name": s.get("name", ""),
                                     "description": s.get("description", ""),
                                     "parameters": s.get("parameters", {})}})
    return out


def usage_from_native(usage) -> dict | None:
    """原生 SDK 末块 usage（CompletionUsage）→ 内部用量格式。"""
    if usage is None:
        return None
    inp = int(getattr(usage, "prompt_tokens", 0) or 0)
    outp = int(getattr(usage, "completion_tokens", 0) or 0)
    if inp + outp <= 0:
        return None
    cached = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached = int(getattr(ptd, "cached_tokens", 0) or 0)
    if not cached:      # DeepSeek 系网关的备选形态
        cached = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    return {"input_tokens": inp, "output_tokens": outp,
            "total_tokens": inp + outp, "cached_tokens": cached}


_zero_reasoning_seen: set[str] = set()   # 已记录过"零推理"的端点键（每进程一次）


def log_zero_reasoning_once(cfg) -> None:
    """整轮零推理内容时记一次日志：帮助区分'模型本就不思考'与'方言字段对不上'
    （如新版 vLLM 改发 reasoning 以外的键）这类静默降级。"""
    key = f"{getattr(cfg, 'base_url', '')}::{getattr(cfg, 'model_id', '')}"
    if key in _zero_reasoning_seen:
        return
    _zero_reasoning_seen.add(key)
    logger.info("端点未返回任何推理内容（若该模型应有思考过程，请检查方言字段）"
                " base=%s model=%s",
                getattr(cfg, "base_url", ""), getattr(cfg, "model_id", ""))


async def native_round(cfg, payload, tools, temperature, sid,
                       token_event: str = "token",
                       reasoning_event: str = "reasoning",
                       emit_extra: dict | None = None,
                       stream_events: bool = True):
    """原生 openai SDK 的一轮流式（统一路径）：捕获推理内容。

    - reasoning_content 流式 emit(reasoning_event)；主图的工具轮撤回/
      最终轮定格由调用方在轮末处理。
    - 网络层整轮重试：流中断从头重来，已 emit 的半截内容会重复（预期行为）。
    - 返回 (AIMessage, usage_dict|None, stopped: bool)。
    """
    extra = emit_extra or {}
    # 延迟导入是行为契约：monkeypatch openai.AsyncOpenAI 替换客户端（既有
    # 测试依赖），模块顶层导入会让补丁失效、测试打到真实网络
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url,
                         timeout=180.0, max_retries=0)
    msgs = to_openai_messages(payload)
    max_retries, base_delay = 5, 1.0
    truncation_retried = False       # 截断自动重试只补一次，防无限循环
    for attempt in range(max_retries + 1):
        stopped = False
        try:
            stream = await client.chat.completions.create(
                model=cfg.model_id, messages=msgs, tools=tools or None,
                temperature=temperature, stream=True,
                stream_options={"include_usage": True},
            )
            content_parts: list[str] = []
            tc_map: dict[int, dict] = {}    # index -> {"id","name","arguments"}
            last_usage = None
            had_reasoning = False           # 本轮是否见过推理内容（方言漂移观测）
            seen_finish = None              # finish_reason：正常完成的标志
            async for chunk in stream:
                if is_stopped(sid):
                    stopped = True
                    break
                cu = usage_from_native(getattr(chunk, "usage", None))
                if cu is not None:
                    last_usage = cu
                if not chunk.choices:
                    continue
                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr:
                    seen_finish = fr
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                # 多方言兼容：DeepSeek 系 reasoning_content / 新事实标准 reasoning
                rc = (getattr(delta, "reasoning_content", None)
                      or getattr(delta, "reasoning", None))
                if isinstance(rc, list):    # 分片数组形态 [{"text"/"summary": ..}]
                    rc = "".join((p.get("text") or p.get("summary") or "")
                                 for p in rc if isinstance(p, dict))
                if isinstance(rc, str) and rc:   # 类型守卫：非字符串不透传
                    had_reasoning = True
                    if stream_events:
                        emit(reasoning_event, {"content": rc, **extra})
                if delta.content:           # 正式答案
                    content_parts.append(delta.content)
                    if stream_events:
                        emit(token_event, {"content": delta.content, **extra})
                for tc in (delta.tool_calls or []):
                    idx = tc.index if tc.index is not None else 0
                    e = tc_map.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        e["id"] = tc.id
                    fn = tc.function
                    if fn is not None:
                        if fn.name:
                            e["name"] += fn.name
                        if fn.arguments:
                            e["arguments"] += fn.arguments
            try:                            # 停止提前退出时关闭流
                await stream.close()
            except Exception:
                pass
            tool_calls = []
            for idx in sorted(tc_map):      # 分片聚合 → langchain tool_calls 格式
                e = tc_map[idx]
                try:
                    args = json.loads(e["arguments"]) if e["arguments"].strip() else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({"name": e["name"], "args": args, "id": e["id"]})
            # 截断检测：网关可能静默切断长生成——流正常结束但既无 finish_reason
            # 也无 usage。首次遇此整轮重试一次；重试仍截断则接受部分内容。
            if not stopped and seen_finish is None and last_usage is None:
                partial = "".join(content_parts)
                if not truncation_retried:
                    truncation_retried = True
                    logger.warning(
                        "native stream truncated (no finish_reason/usage) "
                        "model=%s partial_chars=%d -- retry once",
                        cfg.model_id, len(partial))
                    continue
                logger.warning(
                    "native stream truncated again after retry model=%s "
                    "partial_chars=%d -- returning partial answer",
                    cfg.model_id, len(partial))
            if not had_reasoning and not stopped:
                log_zero_reasoning_once(cfg)
            # 本版本 langchain-core 的 AIMessage 不接受 tool_calls=None
            resp = AIMessage(content="".join(content_parts),
                             tool_calls=tool_calls)
            return resp, last_usage, stopped
        except Exception as e:
            if attempt >= max_retries or not _is_retryable(e):
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("native llm stream broken (attempt %d/%d) model=%s: %s; retry in %.1fs",
                           attempt + 1, max_retries, cfg.model_id, e, delay)
            import asyncio
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")

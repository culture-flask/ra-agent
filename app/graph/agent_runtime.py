"""多 agent 子图共享的 agent 运行时。

从 graph/brainstorm.py 抽出（纯搬家 + 事件前缀参数化）：
- effective_cfg / role_cfg：生效配置与角色级模型绑定（config_id + model_id 覆盖）
- llm_text：控制类 LLM 调用（不向前端透传，只要最终文本）
- agent_speak：一次流式发言（含工具子循环、空完成重试、强制收尾轮）
- short_reason：异常压成一句话人话原因

事件前缀参数化（与 brainstorm.py 原版的唯一行为差异）：
- prefix 缺省 "bs_" 保持争鸣社语义；会讲传 "sem_"。
  事件名 = f"{prefix}token" / f"{prefix}reasoning" /
           f"{prefix}tool_start" / f"{prefix}tool_end" / f"{prefix}internal"
"""

import asyncio
import json

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs

from app.core.events import emit
from app.core.logging import get_logger
from app.core.message_content import hydrate_messages, text_content
from app.graph.native_llm import native_round, retry_event_name
from app.graph.nodes import WorkflowContext

logger = get_logger("agent_runtime")


async def input_messages(ctx: WorkflowContext, state: dict, messages):
    """给本轮用户消息附上原始媒体引用，再按开关展开为模型输入。"""
    media = state.get("input_media") or []
    if media:
        messages = [message.model_copy(deep=True) for message in messages]
        last = next((message for message in reversed(messages)
                     if getattr(message, "type", "") == "human"), None)
        if last is not None:
            content = last.content
            blocks = ([{"type": "text", "text": content}] if isinstance(content, str)
                      else list(content))
            last.content = blocks + media
    return await asyncio.to_thread(
        hydrate_messages, messages, getattr(ctx, "multimodal_service", None),
        state["user_id"], state.get("multimodal"))


async def _persist_usage(state: dict, cfg, usage: dict | None) -> None:
    """多 agent 团队的单轮 LLM 用量落库（llm_usage 表，与主图 generate 同表
    同粒度），供 /usage/summary 报表——团队调用此前不入报表。

    best-effort：线程池异步写，失败只记日志，绝不影响研讨/辩论流程；
    供应商未提供 usage 时静默跳过。"""
    if not usage or not int(usage.get("total_tokens") or 0):
        return

    def _write() -> None:
        try:
            from app.core.db import SessionLocal
            from app.models.entities import LLMUsage
            with SessionLocal() as db:
                db.add(LLMUsage(
                    user_id=state["user_id"],
                    session_id=str(state.get("session_id") or "")[:64],
                    model=str(getattr(cfg, "model_id", "") or "chat"),
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                    cached_tokens=int(usage.get("cached_tokens") or 0)))
                db.commit()
        except Exception as e:
            logger.warning("agent usage persist failed (不影响流程): %s", e)

    await asyncio.to_thread(_write)


def effective_cfg(ctx: WorkflowContext, user_id: str):
    """生效 LLM 配置（用户配置 > 系统默认）；测试假服务无此方法 → None，
    改由 get_chat_model 提供模型给同一流协调器。"""
    getter = getattr(ctx.llm_service, "effective_config", None)
    if not callable(getter):
        return None
    try:
        return getter(user_id)
    except Exception:
        return None


def role_cfg(ctx: WorkflowContext, state: dict, agent_id: str):
    """角色独立模型：role.config_id 引用该用户已保存的一条 LLM 配置
    （get_config 校验属主，提供 key/base_url/默认模型）；role.model_id 可在
    该供应商下**覆盖模型**（凭证继承所选配置，同供应商任意模型直接填名字即可，
    不必为每个模型单独存一条配置）。配置已删除/非本人/解析失败 → 回退用户
    生效配置（model_id 覆盖仍生效）。典型用法：给部分角色配更快的非思考模型，
    压调研阶段的时长与 token。"""
    from dataclasses import replace

    role = next((r for r in state.get("roles") or [] if r.get("id") == agent_id),
                None)
    cid = (role or {}).get("config_id")
    cfg = None
    if cid:
        getter = getattr(ctx.llm_service, "get_config", None)
        if callable(getter):
            try:
                cfg = getter(state["user_id"], cid)
            except Exception:
                cfg = None
    if cfg is None:
        cfg = effective_cfg(ctx, state["user_id"])
    override = (role or {}).get("model_id")
    if cfg is not None and override:
        cfg = replace(cfg, model_id=str(override))
    return cfg


def role_run_config(config: RunnableConfig | None, state: dict,
                    agent_id: str, prefix: str = "bs_",
                    phase: str = "") -> RunnableConfig:
    """合并团队/角色标记，保留图注入的 callbacks、tags 与 metadata。"""
    team = {"bs_": "debate", "sem_": "seminar",
            "dr_": "deep_research"}.get(prefix, prefix.rstrip("_"))
    phase = phase or str(((config or {}).get("metadata") or {})
                        .get("langgraph_node") or "")
    metadata = {"team": team, "agent": agent_id,
                "session_id": state.get("session_id", ""),
                "user_id": state.get("user_id", "")}
    tags = [f"team:{team}", f"agent:{agent_id}"]
    if phase:
        metadata["phase"] = phase
        tags.append(f"phase:{phase}")
    return merge_configs(config, {"tags": tags, "metadata": metadata})


async def llm_text(ctx: WorkflowContext, state: dict,
                   system_prompt: str, human_content: str,
                   temperature: float = 0.2,
                   prefix: str = "bs_",
                   emit_retry: bool = False,
                   emit_extra: dict | None = None,
                   config: RunnableConfig | None = None,
                   agent_id: str = "control", cfg=None) -> tuple[str, int]:
    """控制类 LLM 调用（prepare / curate / moderator / merge）：不向前端
    流式透传，只需最终文本。返回 (文本, token用量)。失败上抛，由调用方降级。

    emit_retry / emit_extra：不流式但"用户在场"的调用（溯源社撰写人
    write_frame）也透出重试倒计时与失败原因，事件带 agent 路由到气泡。"""
    sid = state["session_id"]
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = await input_messages(ctx, state, [SystemMessage(content=system_prompt),
                                               HumanMessage(content=human_content)])
    if cfg is None:
        cfg = effective_cfg(ctx, state["user_id"])
    model = None
    if cfg is None:
        model = await asyncio.to_thread(ctx.llm_service.get_chat_model,
                                        state["user_id"], temperature=temperature)
    call_config = role_run_config(config, state, agent_id, prefix)
    # 控制调用不向消息流公开正文；回调和追踪仍由标准模型执行。
    call_config = merge_configs(call_config, {"tags": ["nostream"]})
    # 空完成防御（与 agent_speak / generate 同纪律）：部分网关偶发
    # "连上了但零内容"的完成。控制类调用若静默返回空串，下游会把失败
    # 当正常——实例：溯源社撰写人拿到空 JSON → 引言/结论全空 → 整场
    # 报告降级成汇编稿。指数退避整轮重试，耗尽仍空交由调用方兜底。
    EMPTY_RETRY_MAX = 2
    used = 0
    resp = None
    for attempt in range(EMPTY_RETRY_MAX + 1):
        resp, usage, stopped = await native_round(
            cfg, payload, None, temperature, sid,
            token_event=f"{prefix}internal",
            reasoning_event=f"{prefix}internal",
            stream_events=False,
            emit_retry=emit_retry, emit_extra=emit_extra,
            config=call_config, model=model)
        used += int((usage or {}).get("total_tokens") or 0)
        await _persist_usage(state, cfg, usage)
        if stopped:
            raise RuntimeError("stopped")
        if text_content(resp.content).strip():
            return text_content(resp.content), used
        if attempt < EMPTY_RETRY_MAX:
            delay = 0.5 * (2 ** attempt)
            logger.warning("llm_text empty completion (attempt %d/%d) "
                           "model=%s", attempt + 1, EMPTY_RETRY_MAX,
                           getattr(cfg, "model_id", "chat"))
            if emit_retry:            # 空完成重试同样让前端看到
                emit(retry_event_name(f"{prefix}x"),
                     {"attempt": attempt + 1, "max": EMPTY_RETRY_MAX,
                      "delay": round(delay, 1),
                      "error": "输出为空，自动重试",
                      **(emit_extra or {})})
            await asyncio.sleep(delay)
    return text_content(resp.content), used


async def _langchain_stream(model, msgs, sid: str, extra: dict, prefix: str,
                            config: RunnableConfig | None = None):
    """测试模型与真实模型复用相同的分片聚合、停止和重试协调器。"""
    return await native_round(
        None, msgs, None, getattr(model, "temperature", 0.3), sid,
        token_event=f"{prefix}token", reasoning_event=f"{prefix}reasoning",
        emit_extra=extra, config=config, model=model)


async def agent_speak(ctx: WorkflowContext, state: dict,
                      agent_id: str, system_prompt: str, human_content: str,
                      temperature: float, tools: list | None,
                      tool_loop_max: int,
                      cfg=None, prefix: str = "bs_",
                      config: RunnableConfig | None = None) -> tuple[str, int, bool]:
    """一个 agent 的一次流式发言（含 generate⇄tool_executor 式子循环）。

    返回 (发言全文, 累计token, stopped)。所有事件带 extra 字段。
    - 标准模型流：native_round 保留思考分片、工具分片与末块权威用量；
    - 测试假模型复用同一协调器；
    - 工具子循环上限 tool_loop_max，防止无限调研不发言/不收敛；
    - 工具预算耗尽仍想调工具 → 追加一轮无工具调用强制产出（强制收尾轮）；
    - 空完成（零内容零工具调用）指数退避重试（与主图 generate 同语义）；
    - 失败外抛前先发 agent_fail 事件（重试耗尽后的最终原因），前端气泡
      即时红字显示，不再只留一个空白气泡等回放。
    """
    try:
        return await _agent_speak_impl(ctx, state, agent_id, system_prompt,
                                       human_content, temperature, tools,
                                       tool_loop_max, cfg=cfg, prefix=prefix,
                                       config=config)
    except Exception as e:
        emit(f"{prefix}agent_fail", {"agent": agent_id,
                                     "error": short_reason(e)})
        raise


async def _agent_speak_impl(ctx: WorkflowContext, state: dict,
                            agent_id: str, system_prompt: str,
                            human_content: str, temperature: float,
                            tools: list | None, tool_loop_max: int,
                            cfg=None, prefix: str = "bs_",
                            config: RunnableConfig | None = None) -> tuple[str, int, bool]:
    """agent_speak 的实现体（失败事件包装见 agent_speak）。"""
    uid, sid = state["user_id"], state["session_id"]
    if cfg is None:                         # 调用方未解析（测试/内部兜底）
        cfg = role_cfg(ctx, state, agent_id)
    extra = {"agent": agent_id}
    call_config = role_run_config(config, state, agent_id, prefix)
    from langchain_core.messages import SystemMessage, HumanMessage

    msgs = await input_messages(ctx, state, [SystemMessage(content=system_prompt),
                                            HumanMessage(content=human_content)])
    used = 0
    resp = None
    EMPTY_RETRY_MAX = 2                     # 空完成防御（与主图 generate 同语义）
    empty_retried = 0
    rounds = max(1, tool_loop_max)
    round_idx = 0
    wrapup_notified = False
    while round_idx < rounds + 1:           # +1：工具预算耗尽后的强制收尾轮
        last_round = round_idx >= rounds
        if last_round and not wrapup_notified:
            # 收尾轮不声明工具，必须显式说明：否则模型仍想调工具时没有结构化
            # 通道，DeepSeek 系会把原生工具模板当正文吐出来（实测 <｜｜DSML｜｜ ...），
            # 直接污染研读笔记/成稿。临时消息只在本轮 msgs 里，不进 checkpoint。
            wrapup_notified = True
            msgs.append(HumanMessage(content=(
                "[系统] 本轮不再提供工具。请直接基于已获得的信息给出最终内容，"
                "不要再请求调用工具。")))
        if cfg is not None:
            resp, usage, stopped = await native_round(
                cfg, msgs,
                None if last_round else tools, temperature, sid,
                token_event=f"{prefix}token",
                reasoning_event=f"{prefix}reasoning",
                emit_extra=extra, config=call_config)
        else:
            model = await asyncio.to_thread(ctx.llm_service.get_chat_model,
                                            uid, temperature=temperature)
            eff_tools = None if last_round else tools
            if eff_tools:
                model = model.bind_tools(eff_tools)
            resp, usage, stopped = await _langchain_stream(model, msgs, sid,
                                                           extra, prefix,
                                                           config=call_config)
        round_idx += 1
        used += int((usage or {}).get("total_tokens") or 0)
        await _persist_usage(state, cfg, usage)
        text_out = text_content(resp.content)
        calls = getattr(resp, "tool_calls", None)
        if stopped:
            return text_out, used, stopped
        if calls and not last_round:
            # 工具轮：执行并回灌（复用 MCPToolAdapter：追踪+限长+结构化错误）
            msgs.append(resp)
            for call in calls:
                emit(f"{prefix}tool_start", {"agent": agent_id,
                                             "name": call["name"],
                                             "args": call["args"]})
                try:
                    out = await ctx.mcp_adapter.call(
                        call["name"], call["args"], sid, uid)
                except Exception as e:             # 工具异常也回灌，促模型重试/绕行
                    out = {"error": str(e), "name": call["name"]}
                # 可观测性：适配层的结构化错误（或执行异常）不能静默——
                # 落日志 + 随 tool_end 推给前端（红 chip），否则用户只能
                # 从模型转述里猜"基础设施错误"（真实教训：嵌入端点宕机窗口）
                err = out.get("error") if isinstance(out, dict) else None
                if err:
                    logger.warning("agent %s tool %s failed: %s",
                                   agent_id, call["name"], str(err)[:300])
                emit(f"{prefix}tool_end", {"agent": agent_id,
                                           "name": call["name"],
                                           **({"error": str(err)[:200]}
                                              if err else {})})
                msgs.append(ToolMessage(
                    content=json.dumps(out, ensure_ascii=False),
                    tool_call_id=call["id"]))
            continue
        # 无工具调用（含强制收尾轮）：有内容即结果
        if text_out.strip():
            return text_out, used, stopped
        # 空完成（无内容、无工具调用、非停止）：部分供应商偶发行为——
        # 不防御的话一次空完成就整角色缺席。指数退避重做当前轮。
        if empty_retried >= EMPTY_RETRY_MAX:
            return text_out, used, stopped
        empty_retried += 1
        logger.warning("agent %s empty completion, retry %d/%d",
                       agent_id, empty_retried, EMPTY_RETRY_MAX)
        await asyncio.sleep(0.5 * 2 ** (empty_retried - 1))
        round_idx -= 1                          # 重做当前轮（收尾轮重试仍不带工具）
    return text_content(resp.content), used, False


def short_reason(e: Exception) -> str:
    """把 LLM/工具异常压成一句话原因：识别常见形态给出人话
    （模型不支持/限流/超时/网络），其余截断原始消息。"""
    msg = str(e)
    low = msg.lower()
    if "not supported" in low and "model" in low:
        # 网关型端点（如 opencode zen）会把上游原始错误透传——错误里的模型名
        # 可能是网关内部路由名，未必是你选的模型。如实带上原始信息。
        return (f"端点拒绝该模型（{msg[:80]}）——若模型名选择无误，"
                "多为端点/网关上游故障，可重试或换模型")
    if "401" in msg or "unauthorized" in low or "invalid api key" in low:
        return "鉴权失败（API key 无效或该模型未对账号开放）"
    if "429" in msg or "rate" in low:
        return "触发限流（429）"
    if "timeout" in low or "timed out" in low:
        return "请求超时"
    if "connection" in low:
        return "网络连接失败"
    return msg[:120]

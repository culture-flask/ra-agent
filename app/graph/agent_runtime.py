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

from langchain_core.messages import AIMessage, ToolMessage

from app.core.cancel import is_stopped
from app.core.events import emit
from app.core.logging import get_logger
from app.graph.native_llm import native_round, to_openai_tools
from app.graph.nodes import WorkflowContext

logger = get_logger("agent_runtime")


def effective_cfg(ctx: WorkflowContext, user_id: str):
    """生效 LLM 配置（用户配置 > 系统默认）；测试假服务无此方法 → None
    → 各帮助函数回退 langchain 路径。与主图 _native_cfg 同语义。"""
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


async def llm_text(ctx: WorkflowContext, state: dict,
                   system_prompt: str, human_content: str,
                   temperature: float = 0.2,
                   prefix: str = "bs_") -> tuple[str, int]:
    """控制类 LLM 调用（prepare / curate / moderator / merge）：不向前端
    流式透传，只需最终文本。返回 (文本, token用量)。失败上抛，由调用方降级。"""
    sid = state["session_id"]
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = [SystemMessage(content=system_prompt),
               HumanMessage(content=human_content)]
    cfg = effective_cfg(ctx, state["user_id"])
    if cfg is not None:
        resp, usage, stopped = await native_round(
            cfg, payload, None, temperature, sid,
            token_event=f"{prefix}internal",
            reasoning_event=f"{prefix}internal",
            stream_events=False)          # 控制流不打字机
        if stopped:
            raise RuntimeError("stopped")
        return str(resp.content or ""), int((usage or {}).get("total_tokens") or 0)
    # 回退路径：测试假服务（与主图 generate 的回退条件一致）
    import asyncio as _asyncio
    model = await _asyncio.to_thread(ctx.llm_service.get_chat_model,
                                     state["user_id"], temperature=temperature)
    resp = await model.ainvoke(payload)
    return str(resp.content or ""), 0


async def _langchain_stream(model, msgs, sid: str, extra: dict, prefix: str):
    """langchain 回退路径的一轮流式（镜像主图 _langchain_round，
    事件换 f"{prefix}token" 并带 extra 字段）。"""
    resp = None
    stopped = False
    async for chunk in model.astream(msgs):
        if is_stopped(sid):                    # 用户点了停止：保留已生成部分
            stopped = True
            break
        resp = chunk if resp is None else resp + chunk
        text = chunk.content
        if isinstance(text, str) and text:
            emit(f"{prefix}token", {"content": text, **extra})
        elif isinstance(text, list):
            for part in text:
                if isinstance(part, dict) and part.get("text"):
                    emit(f"{prefix}token", {"content": part["text"], **extra})
    if resp is None:
        resp = AIMessage(content="")
    return resp, None, stopped


async def agent_speak(ctx: WorkflowContext, state: dict,
                      agent_id: str, system_prompt: str, human_content: str,
                      temperature: float, tools: list | None,
                      tool_loop_max: int,
                      cfg=None, prefix: str = "bs_") -> tuple[str, int, bool]:
    """一个 agent 的一次流式发言（含 generate⇄tool_executor 式子循环）。

    返回 (发言全文, 累计token, stopped)。所有事件带 extra 字段。
    - 原生路径：native_round（思考捕获 + 末块权威用量），工具需转原生格式；
    - 回退路径：langchain astream + bind_tools；
    - 工具子循环上限 tool_loop_max，防止无限调研不发言/不收敛；
    - 工具预算耗尽仍想调工具 → 追加一轮无工具调用强制产出（强制收尾轮）；
    - 空完成（零内容零工具调用）指数退避重试（与主图 generate 同语义）。
    """
    uid, sid = state["user_id"], state["session_id"]
    if cfg is None:                         # 调用方未解析（测试/内部兜底）
        cfg = role_cfg(ctx, state, agent_id)
    extra = {"agent": agent_id}
    native_tools = to_openai_tools(tools) if (tools and cfg is not None) else None
    from langchain_core.messages import SystemMessage, HumanMessage

    msgs = [SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)]
    used = 0
    resp = None
    EMPTY_RETRY_MAX = 2                     # 空完成防御（与主图 generate 同语义）
    empty_retried = 0
    rounds = max(1, tool_loop_max)
    round_idx = 0
    while round_idx < rounds + 1:           # +1：工具预算耗尽后的强制收尾轮
        last_round = round_idx >= rounds
        if cfg is not None:
            resp, usage, stopped = await native_round(
                cfg, msgs,
                None if last_round else native_tools, temperature, sid,
                token_event=f"{prefix}token",
                reasoning_event=f"{prefix}reasoning",
                emit_extra=extra)
        else:
            model = await asyncio.to_thread(ctx.llm_service.get_chat_model,
                                            uid, temperature=temperature)
            eff_tools = None if last_round else tools
            if eff_tools:
                model = model.bind_tools(eff_tools)
            resp, usage, stopped = await _langchain_stream(model, msgs, sid,
                                                           extra, prefix)
        round_idx += 1
        used += int((usage or {}).get("total_tokens") or 0)
        text_out = str(resp.content or "")
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
    return str(resp.content or ""), used, False


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

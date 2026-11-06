from collections.abc import Sequence
from dataclasses import dataclass
import asyncio
import inspect
import json
import random
import re
import time
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.abstractions.bm25 import rrf_fuse
from app.core.cancel import clear_stop
from app.core.events import emit
from app.core.logging import get_logger
from app.core.message_content import (
    hydrate_messages, text_content, with_text_content,
)
from app.services.memory_service import MEMORY_MAX, MemoryService
from app.graph.state import AgentState
from app.graph.native_llm import native_round as _native_round
from app.abstractions.llm import DEFAULT_TEMPERATURE, LLMConfig

logger = get_logger("graph_nodes")


def _model_config_kwargs(method, config: RunnableConfig | None) -> dict:
    """向标准 Runnable 传递调用上下文，同时兼容旧式轻量模型替身。"""
    if config is None:
        return {}
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return {"config": config}
    if any(p.name == "config" or p.kind == inspect.Parameter.VAR_KEYWORD
           for p in params):
        return {"config": config}
    return {}


@dataclass
class WorkflowContext:
    """编排层依赖注入：节点所需的 llm / 知识库等服务，编译时注入。"""
    settings: object
    llm_service: object
    kb_service: object
    mcp_adapter: object | None = None
    tracer: object | None = None
    memory_service: MemoryService | None = None
    multimodal_service: object | None = None

# ---------- 长记忆----------
EXTRACT_PROMPT = """从这段对话中提取值得记住的用户信息（研究主题、偏好、
项目背景等），输出 JSON：
{"memories":[{"key":"snake_case键名","value":"简短值",
              "tier":"core或short","topic":"主题标签（如 研究方向/项目/偏好）"}]}
tier 判定：研究方向、领域、稳定偏好、身份信息等长期有效 → core；
有时效性的事实（正在写某论文、本周任务、当前项目阶段）→ short。
下方会列出该用户已保存的记忆：已在其中的不需要重复输出；若本次对话是
对它的更新（状态变化、内容修正），沿用原 key 输出新值，不要另起键名。
没有值得记住的信息输出 {"memories":[]}。只输出 JSON，不要其他文字。"""

MERGE_PROMPT = """你是记忆整理助手。下面按主题分组了同一用户的多条长期记忆（JSON）。
请把每组内的多条合并为一条：保留全部要点、去掉重复表述，120 字以内，
key 沿用该组第一条的 key。输出 JSON：
{"merged":[{"topic":"原主题","key":"组内第一条的key","value":"合并后文本"}]}
只输出 JSON。"""

KEY_RE = re.compile(r"^[a-z0-9_]{2,32}$")     # 键名白名单：小写/数字/下划线

# 时效词：值里出现且 LLM 标为 core 时强制降级 short（LLM 标注的规则兜底）
TIME_WORDS = ("本周", "正在", "当前", "目前", "暂时", "最近在", "这几天", "今天", "这周")


def _loads_fuzzy(text: str) -> dict | None:
    """容错 JSON 解析：去围栏/从夹带文字里抠 JSON 对象，失败返回 None。"""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _parse_memories(text: str) -> list[dict]:
    """解析 LLM 返回的记忆 JSON（容忍围栏/夹带文字）。解析失败返回空列表。"""
    data = _loads_fuzzy(text)
    return data.get("memories", []) if data else []


def _review(memory: dict) -> bool:
    """写入前审核（规则版）：键名合法 + 值非空且够长 + 单条长度上限。"""
    key, value = memory.get("key", ""), memory.get("value", "")
    if not KEY_RE.match(key):
        return False
    if not isinstance(value, str) or len(value.strip()) < 4:
        return False
    return len(value) <= 200


def _normalize_memory(m: dict) -> dict:
    """LLM 输出归一化：tier 白名单 + 时效词强制降级 + topic 兜底。"""
    tier = m.get("tier") if m.get("tier") in ("core", "short") else "core"
    value = m.get("value", "")
    if tier == "core" and any(w in value for w in TIME_WORDS):
        tier = "short"                      # 临时性事实不进常驻层（标注兜底）
    topic = str(m.get("topic") or "").strip()[:64]
    if not topic:
        topic = m.get("key", "").split("_")[0]     # 兜底：key 首段（research_x → research）
    return {"key": m.get("key", ""), "value": value,
            "tier": tier, "topic": topic}


def _existing_memory_block(existing: dict[str, dict]) -> str:
    """已有记忆清单（拼进抽取 prompt）：让 LLM 认得“记过的”，别换键重记。

    值压平折行 + 截断——清单只用于去重识别，不需要完整正文，控 token 膨胀。
    """
    if not existing:
        return "\n\n该用户当前没有任何已保存的记忆。"
    lines = []
    for key, value in existing.items():        # get_all 已按 key 定序，清单稳定
        text = value.get("v", "") if isinstance(value, dict) else str(value)
        lines.append(f"- {key}: {' '.join(str(text).split())[:80]}")
    return "\n\n该用户已保存的记忆（勿重复提取）：\n" + "\n".join(lines)


# ---------- 自动上下文压缩 ----------
COMPACT_KEEP_ROUNDS = 4          # 压缩后保留的最近轮数
COMPACT_MIN_ROUNDS = 20          # 轮数超过该值触发压缩（>20 轮）
COMPACT_MIN_TOKEN_RATIO = 0.2    # 轮数路径的最小占用门槛

COMPACT_PROMPT = """您正在进行上下文检查点压缩。请为另一个LLM创建一个交接摘要，以便其继续本次对话。

内容包括：
- 用户的研究主题、关键问题及迄今得出的结论
- 对话中已确认的事实、参数、数据和用户偏好
- 未解决的问题或后续跟进事项（明确下一步行动）

请简洁、结构清晰，注重顺畅延续。
仅输出摘要正文：第三人称，按主题组织，400~1000个汉字，使用中文。
"""


def _round_count(messages: list) -> int:
    """轮数 = 用户提问条数（一对 user+assistant 算一轮）。"""
    return sum(1 for m in messages if getattr(m, "type", "") == "human")

# _estimate_tokens 已挪至 core/tokens.py：API 层也要用，
# 不能让 api 反向依赖编排层私有符号），此处按公共名导入。
from app.core.tokens import estimate_tokens


def _usage_from(obj) -> dict | None:
    """从单个消息/chunk 提取真实 token 用量（不做任何累加）。

    优先 LangChain 标准 usage_metadata，回退 OpenAI 风格
    response_metadata.token_usage；都没有返回 None。
    额外提取缓存命中（cached_tokens ⊆ input）：覆盖三家形态——
    OpenAI 的 prompt_tokens_details.cached_tokens、DeepSeek 的
    prompt_cache_hit_tokens、LangChain 归一的 input_token_details.cache_read。
    """
    um = getattr(obj, "usage_metadata", None)
    if isinstance(um, dict) and um.get("total_tokens"):
        details = um.get("input_token_details") or {}
        cached = int(details.get("cache_read") or 0)
        return {"input_tokens": int(um.get("input_tokens") or 0),
                "output_tokens": int(um.get("output_tokens") or 0),
                "total_tokens": int(um["total_tokens"]),
                "cached_tokens": cached}
    tu = (getattr(obj, "response_metadata", None) or {}).get("token_usage")
    if isinstance(tu, dict):
        inp = int(tu.get("prompt_tokens") or 0)
        out = int(tu.get("completion_tokens") or 0)
        if inp + out > 0:
            cached = int(tu.get("prompt_cache_hit_tokens") or 0) \
                or int((tu.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            return {"input_tokens": inp, "output_tokens": out,
                    "total_tokens": inp + out, "cached_tokens": cached}
    return None


def _usage_of(msg) -> dict | None:
    """从聚合后的完整响应提取用量。⚠️ 仅作兜底：

    流式场景下部分供应商会在**每个** chunk 都携带同一份 usage，
    LangChain 的 chunk 聚合（AIMessageChunk 相加）会把重复 usage 累加，
    聚合值可膨胀为真实值的数十倍——实测 1M 窗口的对话统计出千万级用量
    即此因。流式路径以 generate_node 逐 chunk 记录的"末块权威值"为准。
    """
    return _usage_from(msg)


def _split_keep_and_old(messages: list, keep_rounds: int):
    """按轮数切分：保留最近 keep_rounds 轮，其余归为待总结历史。

    用对象身份切分（消息 id 可能是 None——手工构造/输入转换的消息没有 id）。
    """
    keep, user_seen = [], 0
    for m in reversed(messages):
        keep.append(m)
        if getattr(m, "type", "") == "human":
            user_seen += 1
            if user_seen >= keep_rounds:
                break
    keep.reverse()
    keep_refs = {id(m) for m in keep}
    old = [m for m in messages if id(m) not in keep_refs]
    return keep, old


async def compact_node(ctx: WorkflowContext, state: AgentState,
                       config: RunnableConfig | None = None) -> dict:
    """自动上下文压缩（图的第一站）：token 估测达上下文上限 80%，或轮数 >20
    且占用 ≥ 窗口 20% 时，把最近 4 轮之外的历史交给 LLM 总结，
    后续生成使用「总结 + 最近 4 轮」。

    - 总结存入 conversation_summary（拼进系统提示词），旧消息用 RemoveMessage
      删除（messages 是 add_messages reducer，直接传列表只会追加）
    - 压缩失败（LLM 异常）静默跳过，绝不阻断主对话
    """
    messages = state.get("messages") or []
    if not messages:
        return {}
    # 上下文窗口优先从模型 /models 响应探测（LLMService.context_window_for），
    # 假服务/异常时回退配置默认值
    svc = ctx.llm_service
    if hasattr(svc, "context_window_for"):
        # 可能触发同步 httpx 探测，放线程池避免阻塞事件循环
        window = int(await asyncio.to_thread(svc.context_window_for,
                                             state["user_id"]))
    else:
        window = int(getattr(ctx.settings, "llm_context_window", 256000))
    rounds = _round_count(messages)
    # token 占用优先取上一轮 generate 的真实用量（含系统提示词+检索结果，
    # 比 _estimate_tokens 准）；没有时（首轮/假模型）回退字符估算
    last_usage = state.get("last_usage") or {}
    tokens = int(last_usage.get("total_tokens") or 0) or estimate_tokens(messages)
    triggered_by_tokens = tokens > window * 0.8
    if not triggered_by_tokens:
        if rounds <= COMPACT_MIN_ROUNDS:
            return {}
        # 可能只占窗口百分之几，此时总结丢细节纯属浪费；占用低于窗口 20%
        # 一律不压（token 达 80% 的主动路径不受此门槛约束）。
        if tokens <= window * COMPACT_MIN_TOKEN_RATIO:
            return {}

    keep, old = _split_keep_and_old(messages, COMPACT_KEEP_ROUNDS)
    if not old:
        return {}
    try:
        model = await asyncio.to_thread(
            ctx.llm_service.get_chat_model,
            state["user_id"], temperature=state.get("temperature"))
        # 旧摘要并入本次总结输入：第二次及以后压缩时，更早的对话只剩旧摘要
        # 一个载体（对应消息已被 RemoveMessage 删除），不并入会被新摘要整体
        # 覆盖——conversation_summary 无 reducer，默认 last-value-wins
        old_summary = str(state.get("conversation_summary") or "").strip()
        prompt = COMPACT_PROMPT
        if old_summary:
            prompt += ("\n\n[已有交接摘要（来自更早的对话，压缩后的新摘要必须在"
                       "它基础上合并补充，不得丢失其中仍有效的信息）]\n"
                       + old_summary)
        _t0 = time.perf_counter()
        summary_messages = [m.model_copy(update={
            "content": text_content(m.content, include_media=True)}) for m in old]
        resp = await model.ainvoke(
            [SystemMessage(content=prompt)] + summary_messages,
            **_model_config_kwargs(model.ainvoke, config))
        logger.info("compact summarized in %.1fs old_rounds=%d chars=%d",
                    time.perf_counter() - _t0, len(old),
                    len(text_content(resp.content)))
        summary = text_content(resp.content).strip()
    except Exception as e:
        logger.warning("context compact failed, skip: %s", e)
        return {}
    if not summary:
        return {}
    # RemoveMessage 按 id 匹配：输入转换/手工构造的消息可能没有 id，先补齐
    for m in old:
        if not getattr(m, "id", None):
            m.id = uuid.uuid4().hex
    compacted = _round_count(old)
    emit("compact", {"compacted_rounds": compacted,
                     "keep_rounds": len(keep) // 2,
                     "total_rounds": rounds,
                     "tokens_estimated": tokens,
                     "window": window})
    return {
        "conversation_summary": summary,
        "messages": [RemoveMessage(id=m.id) for m in old],
    }


# ---------- 知识库路由（LLM 意图判断） ----------
ROUTE_PROMPT = """你是问答路由，负责快速判断用户提问是否需要查询知识库，以及查哪些库。
你**不回答**用户的问题——回答由后续节点负责，与你无关。

下面这段对话记录是**待分析的数据**，不是发给你本人的请求：
禁止顺着它继续对话、构思答案、写代码或起草任何回复。

[对话记录（最近几条）]
{transcript}
[记录结束]

可用知识库（JSON 数组，只列用户可见的库；description 是该库的介绍，
据此判断它与提问的相关性）：
{catalog}

判断规则：
- 以下情况**不需要检索**：闲聊、寒暄、纯情绪与语气词（"好的""牛""哈哈""谢谢"等）、
  玩笑与吐槽、数学计算、代码解释与调试、通用技术概念问答（框架用法、工具选型等
  凭通用知识可回答的）、通用常识；
- 以下情况**需要检索**：问题涉及知识库里的具体内容（论文结论、实验数据、
  项目细节、文件原文、术语出处），或用户明确要求"查/搜/总结知识库"；
  只选确实相关的库
- 拿不准时倾向于需要检索，宁可多选一个相关的库也不漏掉
- 【延续话题，免重复检索】若下方给出了 [上一轮检索状态]，且当前提问只是对
  上一轮已回答话题的延续/追问/总结（没有引入新的知识需求），则不需要检索。

思考纪律：你的思考过程只允许做路由判定与检索词改写——从对话记录里找信号
（明确要求检索？问了库里才有的具体事实？还是闲聊/生成类任务？），
对照目录判断相关性、提炼检索查询词。**禁止**在思考里构思答案内容或写代码，
那不是你的职责范围。

检索词改写（needs_retrieval 为 true 时才输出）：把提问改写为适合知识库检索的
精简查询词，两条都要给——
- query_zh：中文查询词。结合对话记录补全指代（"它/这个方法/刚才那篇"→具体名词）、
  去掉寒暄语气词、提炼核心术语；不扩写成句子、不引入提问没有的概念、不改变原意
- query_en：同一意图的英文查询词（术语对应翻译；模型名/缩写/专有名词保留原文，
  如 qwen3、RAG、BM25）；提问本身是英文时，它即精简后的原句
- 每条不超过 60 字符；不需要检索时两条为空字符串

注意：你只输出 JSON 判定，不跟用户对话、不接梗、不闲聊、不解释。

只输出 JSON，不要任何其他文字：
{{"needs_retrieval": true或false, "kbs": [{{"name": "库名", "scope": "public或private"}}],
 "query_zh": "中文检索词", "query_en": "英文检索词"}}
不需要检索时 kbs 为 []、query_zh 与 query_en 为空字符串。"""


def _route_transcript(msgs: Sequence[BaseMessage]) -> str:
    """把最近几条消息转写成纯文本记录，作为路由提示词里的**引用数据**。

    为什么不作为对话轮传入：路由是元任务，真实用户轮会诱发模型"接题作答"
    ——思考过程直接开始构思回答/写代码（路由耗时暴涨、判定被"我已会答"
    的错觉污染）。转写成数据后，对话本能失去作用对象；对话轮里只留一句
    极短的判定指令。"""
    lines = []
    for m in msgs:
        role = "用户" if getattr(m, "type", "") == "human" else "助手"
        text = text_content(getattr(m, "content", ""), include_media=True).strip()
        lines.append(f"[{role}] {text[:800]}")
    return "\n".join(lines) or "（无）"


def _parse_route(text: str) -> dict:
    """解析路由 LLM 返回的 JSON（容忍围栏/夹带文字）。解析失败返回空 dict。"""
    return _loads_fuzzy(text) or {}


def _clean_query(text) -> str:
    """路由返回的改写检索词：类型/空白/长度清洗；不可用返回空串（回退原 query）。"""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:120]


def _looks_like_chitchat(state: AgentState) -> bool:
    """路由输出不可解析时的温和降级判定：最后一条用户消息像闲聊/语气词
    （短消息、无问句、无检索意图词）→ True（判不检索）。

    背景：带历史上下文时模型偶发"聊天式"输出（用户玩梗时模型
    接梗不输出 JSON）→ 解析失败走全库检索兜底，无关话题也会带上检索。
    json_object 模式已从源头压住，此判定作为第二道保险。
    """
    msgs = state.get("messages") or []
    if not msgs:
        return False
    text = text_content(getattr(msgs[-1], "content", "")).strip()
    if not text:
        return True                       # 空消息：按闲聊处理
    if len(text) > 20:
        return False                      # 长消息通常有真实意图
    if any(k in text for k in ("？", "?", "查", "搜", "论文", "文献", "资料",
                               "总结", "介绍", "解释", "库",
                               "什么", "为什么", "怎么", "如何", "哪", "谁",
                               "区别", "定义", "是")):
        return False                      # 含检索意图词/疑问词：不按闲聊降级
    return True


def _resolve_selected_kbs(kbs: list, picks: list) -> list:
    """把 LLM 选的 (name/scope) 解析回可见知识库对象。

    只允许命中「用户可见」的库：名称不存在、或属于他人私人库的名称一律忽略，
    防止用户（或 LLM 被诱导）越权检索。scope 写错/漏写时按名称兜底匹配。
    """
    selected, seen = [], set()
    for pick in picks or []:
        if not isinstance(pick, dict):
            continue
        name = str(pick.get("name", "")).strip()
        scope = str(pick.get("scope", "")).strip().lower()
        if not name:
            continue
        matches = [kb for kb in kbs
                   if kb.name == name and (not scope or kb.scope == scope)]
        if not matches:                          # scope 漏写/写错 → 名称兜底
            matches = [kb for kb in kbs if kb.name == name]
        for kb in matches:
            if kb.kb_id not in seen:
                seen.add(kb.kb_id)
                selected.append(kb)
    return selected


async def load_memory_node(ctx: WorkflowContext, state: AgentState,
                           config: RunnableConfig | None = None) -> dict:
    """跨会话读取：只把 core 层注入状态（short 不进 prompt，省 token 防噪音）。
    会话内只注入一次：首轮读库后随 checkpoint 持久化，后续轮直接复用快照——
    会话中新抽取的记忆源自历史消息本身，再注入只会与上下文重复；且快照
    不变让系统提示词跨轮字节级一致（前缀缓存更稳）。
    """
    if ctx.memory_service is None:
        return {"memory": {}}
    if state.get("memory"):                    # 非首轮：复用首轮快照，不重查
        emit("memory_load", {"count": len(state["memory"]), "cached": True})
        return {}
    uid = state["user_id"]
    memory = await asyncio.to_thread(ctx.memory_service.get_all, uid, "core")
    if memory:
        # 注入即使用：刷新 last_used_at（LRU 依据；core 因此永不被淘汰 = 常驻）
        await asyncio.to_thread(ctx.memory_service.touch, uid, list(memory))
    if random.random() < 0.05:                # 惰性清理（约 1/20 概率，避免每轮扫表）
        try:
            await asyncio.to_thread(ctx.memory_service.expire_short, uid)
        except Exception as e:
            logger.warning("memory expire_short failed: %s", e)
    emit("memory_load", {"count": len(memory)})
    return {"memory": memory}


def _strip_user_blocks(content) -> str:
    """剥离用户消息里的检索块与附件块，只留用户真实表述。

    两类块都是系统拼接的资料（_RET_MARK / _ATT_MARK 之后的内容），
    不应参与记忆抽取——否则资料正文会被误抽成"用户信息"。
    """
    from app.api.chat import _ATT_MARK       # 延迟导入：避免 graph→api 模块级循环
    text = text_content(content)
    text = text.split(_RET_MARK)[0]          # 检索块在整条消息末尾
    text = text.split(_ATT_MARK)[0]          # 附件块紧随问题之后
    return text.strip()


def _memory_view(messages: Sequence) -> list:
    """抽取 LLM 的输入视图：剥离资料块后只留 human/ai 文本（空内容跳过）。"""
    out = []
    for m in messages:
        t = getattr(m, "type", "")
        if t not in ("human", "ai"):
            continue                         # 工具往返等中间消息与用户信息无关
        text = _strip_user_blocks(getattr(m, "content", ""))
        if text:
            out.append(HumanMessage(content=text) if t == "human"
                       else AIMessage(content=text))
    return out


async def extract_memory_node(ctx: WorkflowContext, state: AgentState,
                              config: RunnableConfig | None = None) -> dict:
    """对话结束后抽取值得记住的信息：交给 LLM 从对话中提炼。

    抽取前带上该用户已有记忆，让 LLM 认得“记过的”——同一事实不换键重记，
    只对变化的部分沿用原 key 输出更新。输入剥离附件/检索块（_memory_view），
    杜绝资料正文被误抽为用户信息。
    重试耗尽仍失败时静默跳过——记忆抽取是锦上添花，绝不能打挂主对话。
    """
    if ctx.memory_service is None:
        return {"new_memories": []}
    view = _memory_view(state["messages"][-4:])      # 只看最近几轮
    memories: list = []
    if view:                                         # 全为资料/工具往返：跳过 LLM 调用
        try:
            model = await asyncio.to_thread(
                ctx.llm_service.get_chat_model,
                state["user_id"], temperature=state.get("temperature"))
            existing = await asyncio.to_thread(
                ctx.memory_service.get_all, state["user_id"])   # 全层级：去重不分 core/short
            system = SystemMessage(
                content=EXTRACT_PROMPT + _existing_memory_block(existing))
            resp = await model.ainvoke(
                [system] + view, **_model_config_kwargs(model.ainvoke, config))
            memories = _parse_memories(text_content(resp.content))
        except Exception as e:
            logger.warning("memory extract failed, skip: %s", e)
            memories = []
    emit("memory_extract", {"candidates": len(memories)})
    return {"new_memories": memories}


async def save_memory_node(ctx: WorkflowContext, state: AgentState,
                           config: RunnableConfig | None = None) -> dict:
    """写入前审核 → 归一化（tier/topic）→ 落库 → 超限触发膨胀控制管线。"""
    if ctx.memory_service is None:
        return {}
    saved = 0
    for m in state.get("new_memories", []):
        if not _review(m):
            continue                                   # 审核不通过，丢弃
        m = _normalize_memory(m)
        try:
            await asyncio.to_thread(ctx.memory_service.set,
                                    state["user_id"], m["key"], {"v": m["value"]},
                                    tier=m["tier"], topic=m["topic"])
            saved += 1
        except Exception:
            pass          # 记忆写入失败绝不影响主对话（锦上添花原则）
    stat = {"compressed": 0, "evicted": 0}
    try:
        count = await asyncio.to_thread(ctx.memory_service.count, state["user_id"])
        if count > MEMORY_MAX:                         # 超限：压缩 → LRU，而非丢弃新记忆
            stat = await _maintain_memories(ctx, state["user_id"], config=config)
            emit("memory_maintain", {"before": count, **stat})
    except Exception as e:
        logger.warning("memory maintain failed: %s", e)
    emit("memory_save", {"saved": saved, **stat})
    return {}


async def _maintain_memories(ctx: WorkflowContext, user_id: str,
                             config: RunnableConfig | None = None) -> dict:
    """膨胀控制管线：①LLM 主题压缩（同类合并）→ ②LRU 淘汰（先 short 后 core 兜底）。

    每步失败降级到下一步，绝不打挂主对话。
    """
    ms = ctx.memory_service
    stat = {"compressed": 0, "evicted": 0}
    # ① 主题压缩：同 topic ≥2 条交给 LLM 合并成 1 条
    try:
        groups = await asyncio.to_thread(ms.topic_groups, user_id)
        if groups:
            payload = {t: g for t, g in groups.items()}
            merged = await _merge_memories_llm(ctx, user_id, payload, config=config)
            if merged:
                # 附上组内 key 列表，apply_merge 据此删除被合并的旧条目
                for item in merged:
                    topic = item.get("topic")
                    if topic in payload:
                        item["group_keys"] = [r["key"] for r in payload[topic]]
                stat["compressed"] = await asyncio.to_thread(
                    ms.apply_merge, user_id, merged)
    except Exception as e:
        logger.warning("memory topic merge failed, fallback LRU: %s", e)
    # ② LRU 淘汰到上限内（先 short；全 short 不够 core 兜底）
    if await asyncio.to_thread(ms.count, user_id) > MEMORY_MAX:
        stat["evicted"] = await asyncio.to_thread(ms.evict_overflow, user_id)
    return stat


async def _merge_memories_llm(ctx: WorkflowContext, user_id: str,
                              groups: dict,
                              config: RunnableConfig | None = None) -> list[dict] | None:
    """一次 LLM 调用批处理所有分组的合并；失败返回 None（降级 LRU）。"""
    try:
        model = await asyncio.to_thread(
            ctx.llm_service.get_chat_model, user_id, temperature=0.0)
        resp = await model.ainvoke([
            SystemMessage(content=MERGE_PROMPT),
            HumanMessage(content=json.dumps(groups, ensure_ascii=False))],
            **_model_config_kwargs(model.ainvoke, config))
        data = _loads_fuzzy(text_content(resp.content))
        return data.get("merged", []) if data else None
    except Exception as e:
        logger.warning("memory merge LLM call failed: %s", e)
        return None

# 固定身份句：永不变化，置于系统提示词最前，保证所有请求共享同一前缀段
_IDENTITY = ("你是科研助手。若用户消息后附有【知识库检索结果】，优先据此回答，"
             "引用时标明来源（public/private）；否则根据你自己的知识或调用工具来回答。")


def _build_system_prompt(state: AgentState) -> str:
    """组装稳定的系统提示词前缀：固定身份 → 用户记忆（定序）→ 历史总结（低频变化）。

    检索结果不进 system（每轮都变，放前缀会打穿供应商侧前缀缓存），
    由 retrieve 节点并入最后一条用户消息（随 checkpoint 持久化）——
    system 与对话历史跨轮字节级一致，命中 KV 前缀缓存。
    """
    parts = [_IDENTITY]
    if state.get("memory"):
        # sort_keys：键序不随 DB 返回顺序漂移（get_all 已 ORDER BY key，双保险）
        parts.append(f"[用户记忆] "
                     f"{json.dumps(state['memory'], ensure_ascii=False, sort_keys=True)}")
    if state.get("conversation_summary"):
        parts.append(f"[历史对话总结] {state['conversation_summary']}")
    return "\n\n".join(parts)


_RET_MARK = "\n<<<检索结果>>>\n"   # 用户消息内 问题/附件 与 检索块 的边界标记


def _retrieval_block(retrievals: list | None) -> str | None:
    """把本轮检索结果打包为文本块（并入最后一条用户消息，随 checkpoint 持久化）。

    - 并入用户消息而非插入独立 system/消息：序列中间的 system 属分布外
      排列，会触发 thinking 模型长思考；user 形态是标准 RAG 拼装
    - 块以 _RET_MARK 与用户原文分隔：历史回放时拆出展示；重新生成会重跑
      检索，追加前先剥离旧块（幂等，不重复累积）
    """
    if not retrievals:
        return None
    lines = ["【知识库检索结果】回答本问题时优先依据以下内容，引用标明来源。"]
    for r in retrievals:
        if r.get("type") == "parent":
            # 聚合父块：完整段落，标注来源文件/页码与命中片段数
            loc = r.get("source") or "未知来源"
            if r.get("pages"):
                loc += f" 第{'-'.join(str(p) for p in r['pages'])}页"
            lines.append(
                f"[知识库检索结果·上下文段落 ({r.get('scope')} / {r.get('kb_name')} / {loc}，"
                f"含{r.get('hit_chunks')}个命中片段)] {r['text']}")
        else:
            lines.append(f"[知识库检索结果 ({r.get('scope')} / {r.get('kb_name')})] {r['text']}")
    return "\n".join(lines)


def _retrieval_patch(msgs: list, block: str) -> HumanMessage | None:
    """把检索块并入最后一条用户消息：同 id → add_messages 原地替换。

    持久化收益：后续轮次仍带本轮资料，每轮请求序列包含上一轮全部内容
    （供应商前缀缓存命中含检索块的历史）；重新生成重跑检索时先剥离旧块
    再追加。找不到用户消息/消息缺 id（理论不可达）→ None（跳过不影响主流程）。
    """
    last = next((m for m in reversed(msgs)
                 if getattr(m, "type", "") == "human"), None)
    mid = getattr(last, "id", None) if last is not None else None
    if mid is None:
        return None
    base = text_content(last.content).split(_RET_MARK)[0].rstrip()
    return last.model_copy(deep=True, update={
        "content": with_text_content(last.content, base + _RET_MARK + block)})


# 路由是简单分类任务，思考纯属浪费延迟（业内共识：提示词压不住思考模型，
# 必须参数级关闭）。各端点只认自家的"关思考"方言键，按序探测：
# 报参数不识别类错误就换下一个，全不认则裸调（行为同旧版）。
_THINK_OFF_DIALECTS: tuple[dict, ...] = (
    {"reasoning_effort": "low"},                           # OpenAI 系推理模型降档
    {"thinking": {"type": "disabled"}},                    # DeepSeek V4 / 智谱 GLM
    {"enable_thinking": False},                            # Qwen3（DashScope 兼容模式）
    {"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3（vLLM/SGLang 自部署）
)
# 探测结果按端点缓存（值 None 表示该端点不认任何方言、裸调即可），
# 进程内每个端点最多完整探测一次，后续路由零探测开销。
_route_dialect_cache: dict[tuple[str, str], "dict | None"] = {}
# 参数被端点拒绝的典型错误特征（小写匹配）：命中才换方言，
# 网络/鉴权等其它错误照旧走重试或上抛。
_PARAM_REJECT_MARKERS = (
    "unrecognized", "unexpected", "unknown", "unsupported",
    "not permitted", "not allowed", "does not support",
    "extra fields", "extra_forbidden", "invalid_request_error",
    "thinking", "enable_thinking", "chat_template_kwargs",
    "reasoning_effort",
)


def _is_param_rejected(e: Exception) -> bool:
    """判断异常是否为端点不识别请求参数（用于关思考方言探测换挡）。"""
    msg = str(e).lower()
    return any(m in msg for m in _PARAM_REJECT_MARKERS)


async def _langchain_route_round(model, cfg: LLMConfig | None, prompt: str,
                                 msgs_lc: Sequence[BaseMessage],
                                 temperature: float, sid: str,
                                 use_json: bool = True,
                                 config: RunnableConfig | None = None) -> str:
    """标准模型路由：探测并缓存端点的关闭思考方言，仍透传残余推理。

    绑定请求参数后复用统一流协调器，保留 Runnable callbacks、tags、metadata
    与停止语义；不支持 bind 的轻量替身沿用 ainvoke 路径。
    """
    payload = [SystemMessage(content=prompt)] + list(msgs_lc)
    bind = getattr(model, "bind", None)
    if not callable(bind):
        kwargs = {"response_format": {"type": "json_object"}} if use_json else {}
        resp = await model.ainvoke(
            payload, **_model_config_kwargs(model.ainvoke, config), **kwargs)
        return text_content(resp.content)

    key = (str(getattr(cfg, "base_url", "")),
           str(getattr(cfg, "model_id", getattr(model, "model_name", ""))))
    if key in _route_dialect_cache:
        candidates: list = [_route_dialect_cache[key]]
    else:
        candidates = list(_THINK_OFF_DIALECTS) + [None]
    last_err = None
    for extras in candidates:
        kwargs = {"response_format": {"type": "json_object"}} if use_json else {}
        if extras:
            kwargs["extra_body"] = extras
        bound = model.bind(**kwargs) if kwargs else model
        try:
            resp, _, _ = await _native_round(
                cfg, payload, None, temperature, sid,
                token_event="routing_internal", reasoning_event="routing_reasoning",
                config=config, model=bound)
            _route_dialect_cache[key] = extras
            return text_content(resp.content)
        except Exception as e:
            last_err = e
            # JSON 模式拒绝须交给 supervisor 降级，不能误当成思考方言拒绝。
            if "response_format" in str(e).lower() or "json_object" in str(e).lower():
                raise
            if not _is_param_rejected(e):
                raise
    raise last_err or RuntimeError("unreachable")


async def supervisor_node(ctx: WorkflowContext, state: AgentState,
                          config: RunnableConfig | None = None) -> dict:
    """路由决策（LLM 意图判断）：先看用户有没有可见知识库。

    - 没有可见库 → 无需检索，直接生成
    - 有可见库 → 把「库名 + scope」目录交给 LLM，让它判断本次提问
      是否需要检索、以及选哪几个库（按名称），再做可见性校验后落 state
    - LLM 判断异常/解析失败 → 降级为全部可见库检索（保持 RAG 兜底）
    """
    # 只用"可检索"的库（用户可自行禁用某库参与对话检索）。
    # team=None：普通对话不见任何多 agent 沉淀库（辩论式/研讨式/深度调研
    # agent纪要）——沉淀库只服务多 agent 团队的研读/调研与知识库检索工具。
    kbs = await asyncio.to_thread(ctx.kb_service.list_queryable_kbs,
                                  state["user_id"])
    if not kbs:
        emit("supervisor", {"needs_retrieval": False, "kb_count": 0, "selected": []})
        return {"needs_retrieval": False, "selected_kb_ids": [],
                "search_queries": [],
                "last_retrieval_state": {"kb_ids": [], "kb_names": [], "hit_count": 0}}

    catalog = [{"name": kb.name, "scope": kb.scope,
                "description": kb.description or ""} for kb in kbs]
    selected: list = []
    rewritten: list[str] = []      # 路由改写的中/英检索词（记入 state 供多路检索）
    # 路由前置事件：thinking 模型的路由调用可能耗时数十秒且此前前端无任何
    # 反馈（memory_load 与 supervisor 之间静默）——先告诉前端"正在判断意图"。
    emit("routing", {})
    route_cfg = _native_cfg(ctx, state["user_id"])
    latest = state["messages"][-4:]
    # 元任务防漂移：用户问题不作为对话轮传入（模型会"接题作答"——思考
    # 直接开始构思回答/写代码），而是转写成纯文本记录拼进路由提示词，
    # 对话轮只留一句极短的判定指令。
    msgs_route = [HumanMessage(content="请按系统指令只输出 JSON 判定。")]
    try:
        prompt = ROUTE_PROMPT.format(
            catalog=json.dumps(catalog, ensure_ascii=False),
            transcript=_route_transcript(latest))
        # 记录只带最近 4 条，指代上文的追问（"接着刚才那个
        # 方案说"）会因看不到上文被误判为无需检索——把压缩总结（低频变化、
        # 已定序，不破坏前缀缓存）附在目录后，让路由 LLM 知道"刚才在聊什么"。
        summary = str(state.get("conversation_summary") or "").strip()
        if summary:
            prompt += "\n\n[历史对话总结]\n" + summary[:500]
        # 第一优先：把上一轮的检索结果状态喂给路由——延续话题且上轮
        # 已命中可免重复检索。只拼进路由 prompt，绝不进 system/历史（保前缀缓存）。
        lrs = state.get("last_retrieval_state") or {}
        if lrs.get("hit_count"):
            prompt += ("\n\n[上一轮检索状态] 上一轮检索了知识库「%s」，共命中 %d 条。"
                       % ("、".join(lrs.get("kb_names") or ["?"]), lrs["hit_count"]))
        # 强制 JSON 输出：带历史上下文时模型偶发"聊天式"输出——用户
        # 玩梗时模型接梗不输出 JSON → 解析失败走全库检索降级，
        # 无关话题也会带上检索。json_object 模式从源头压住；端点不支持时回退
        # 普通调用（老版本网关/部分自建端点没有 JSON 模式）。
        _t0 = time.perf_counter()
        model = await asyncio.to_thread(
            ctx.llm_service.get_chat_model,
            state["user_id"], temperature=state.get("temperature"))
        try:
            text = await _langchain_route_round(
                model, route_cfg, prompt, msgs_route,
                _native_temperature(state), state["session_id"],
                use_json=True, config=config)
        except Exception as e:
            if ("response_format" not in str(e).lower()
                    and "json_object" not in str(e).lower()):
                raise
            logger.warning("router json mode unsupported, fallback plain: %s", e)
            text = await _langchain_route_round(
                model, route_cfg, prompt, msgs_route,
                _native_temperature(state), state["session_id"],
                use_json=False, config=config)
        logger.info("supervisor routed in %.1fs msgs=%d",
                    time.perf_counter() - _t0, len(latest))
        route = _parse_route(text)
        if not route:                         # LLM 没按 JSON 输出 → 无法判断意图
            if _looks_like_chitchat(state):
                needs, selected = False, []   # 明显闲聊：温和降级为不检索
            else:
                raise ValueError("unparseable route output")
        else:
            needs = bool(route.get("needs_retrieval"))
            if needs:
                selected = _resolve_selected_kbs(kbs, route.get("kbs"))
                if not selected:              # 说要查但一个库都没选中 → 全查，避免漏检索
                    selected = list(kbs)
                # 改写检索词（中/英各一条）：脏数据/缺失经 _clean_query 过滤后
                # 不入列表 → 检索侧回退原始 query 单路检索（兼容没给字段的旧行为）
                for key in ("query_zh", "query_en"):
                    q = _clean_query(route.get(key))
                    if q:
                        rewritten.append(q)
    except Exception as e:
        logger.warning("kb routing failed, fallback to all visible kbs: %s", e)
        needs, selected = True, list(kbs)
    emit("supervisor", {
        "needs_retrieval": needs,
        "kb_count": len(kbs),
        "selected": [{"name": kb.name, "scope": kb.scope} for kb in selected],
    })
    return {"needs_retrieval": needs,
            "selected_kb_ids": [kb.kb_id for kb in selected],
            "search_queries": rewritten,
            # 仅当本轮不检索时在此清空"上一轮检索状态"——检索轮由 retrieve_node
            # 覆写为真实值，避免"上一轮检索过、本轮没检索但仍沿用旧状态"的误判。
            "last_retrieval_state": (None if needs
                                     else {"kb_ids": [], "kb_names": [], "hit_count": 0})}


def route_supervisor(state: AgentState) -> str:
    return "retrieve" if state.get("needs_retrieval") else "generate"

def route_after_generate(state: AgentState) -> str:
    """LLM 要调工具 → 走 tool_executor；否则结束。用户手动停止 → 直接结束。"""
    if state.get("stopped"):                      # 中断的部分答复不再进工具循环
        return "done"
    last = state["messages"][-1]
    return "tool_executor" if getattr(last, "tool_calls", None) else "done"


async def tool_executor_node(ctx: WorkflowContext, state: AgentState,
                             config: RunnableConfig | None = None) -> dict:
    """执行 LLM 请求的工具：经 MCPToolAdapter → MCP tools/call，结果回灌。

    追踪行挂到触发它的 LLM 调用下（parent_id）——追踪页按调用树展示。
    工具的实时 event（trace_start/trace_end，带 args 与 parent_id）由 Tracer
    统一发出，这里不再另发 tool_start：否则同一调用会出现两条步骤。
    """
    last = state["messages"][-1]
    parent_id = state.get("llm_log_id")          # generate 本轮写入的 llm 行 id
    results = []
    for call in last.tool_calls:
        out = await ctx.mcp_adapter.call(call["name"], call["args"],
                                         state["session_id"], state["user_id"],
                                         parent_id=parent_id)
        results.append(ToolMessage(content=json.dumps(out, ensure_ascii=False),
                                   tool_call_id=call["id"]))
    return {"messages": results}

def _hit_rank_key(h: dict) -> float:
    """跨库合并排序键：重排分 > RRF 融合分 > 向量距离。

    启用重排时命中带 rerank_score（同一交叉编码器出的相关性分，越大越好），
    以它为准，否则跨库合并会按 RRF 分把重排结果重新打乱；未重排时沿用
    hybrid RRF 分（越大越好）/ vector 距离（越小越好）。
    """
    if h.get("rerank_score") is not None:
        return -float(h["rerank_score"])
    if h.get("score") is not None:
        return -float(h.get("score") or 0)
    return float(h.get("distance") or float("inf"))


def _chunk_group_of(hit: dict, group_size: int) -> int:
    """命中 chunk 的父块组号：chunk_id 形如 {kb_id}_{doc_id}_{i}，组 = i // group_size。"""
    try:
        i = int(str(hit.get("id", "")).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        i = 0
    return i // group_size


def _expand_parent_blocks(ctx, kb_hits: list, parent_budget: int,
                          group_size: int, max_chars: int,
                          total: int) -> list[dict]:
    """聚合返回：前 parent_budget 个不同父块展开为完整段落，其余按分数返回小 chunk。

    - 父块 key = (kb_id, doc_id, 组号)；已展开组内的命中 chunk 不再返回
      （其上下文已包含在父块里，只累计 hit_chunks）
    - 父块分数 = 组内最高命中分（按分数降序扫描，第一个命中的分即组内最高）
    - 父块与小 chunk 按分数混排（复用 _hit_rank_key）
    - 展开失败（磁盘缺失）自动降级为小 chunk；父块不足 budget 时自然退化
    """
    parents: dict = {}                       # key -> 父块条目
    expanded: set = set()
    for h in kb_hits:
        if len(expanded) >= parent_budget:
            break
        doc_id = (h.get("metadata") or {}).get("doc_id")
        if not doc_id:
            continue
        key = (h.get("kb_id"), doc_id, _chunk_group_of(h, group_size))
        if key in expanded:
            continue
        block = ctx.kb_service.get_parent_block(
            key[0], doc_id, key[2], group_size=group_size, max_chars=max_chars)
        if block is None:
            continue
        entry = {
            "type": "parent",
            "text": block["text"],
            "kb_id": h.get("kb_id"), "kb_name": h.get("kb_name"),
            "scope": h.get("scope"),
            "doc_id": doc_id, "group": key[2],
            "source": block.get("source"), "pages": block.get("pages"),
            "hit_chunks": 1,
            "score": h.get("score"), "distance": h.get("distance"),
            "bm25_score": h.get("bm25_score"), "method": h.get("method"),
            "metadata": {"scope": h.get("scope"), "doc_id": doc_id},
        }
        parents[key] = entry
        expanded.add(key)

    chunk_budget = max(0, total - len(expanded))   # 剩余名额给小 chunk
    chunks: list[dict] = []
    for h in kb_hits:
        doc_id = (h.get("metadata") or {}).get("doc_id")
        key = ((h.get("kb_id"), doc_id, _chunk_group_of(h, group_size))
               if doc_id else None)
        if key in expanded:
            parents[key]["hit_chunks"] += 1    # 上下文已在父块里，只累计不重复返回
            continue
        if len(chunks) >= chunk_budget:
            break
        chunks.append({**h, "type": "chunk"})
    all_entries = list(parents.values()) + chunks
    all_entries.sort(key=_hit_rank_key)
    return all_entries


async def retrieve_node(ctx: WorkflowContext, state: AgentState,
                        config: RunnableConfig | None = None) -> dict:
    """按 LLM 选定的知识库检索（仅限可见库），结果带 scope 标签（引用溯源）。

    检索模式：state.retrieval_mode（前端每轮传）→ 全局配置 retrieval_mode 兜底。
    """
    kbs = await asyncio.to_thread(ctx.kb_service.list_kbs, state["user_id"])
    selected_ids = set(state.get("selected_kb_ids") or [])
    # 双保险：即使 selected_ids 携带被禁用的库（路由与检索之间用户改了开关），
    # 也跳过——检索开关 per-user，只认当前用户自己的禁用列表
    uid = state["user_id"]
    targets = [kb for kb in kbs
               if kb.kb_id in selected_ids and ctx.kb_service.kb_queryable(kb, uid)]
    mode = (state.get("retrieval_mode")
            or getattr(ctx.settings, "retrieval_mode", "hybrid"))
    # 检索数量：每库 k 条 / 合并后总共 top 条（state 传入 → 全局配置兜底，带范围约束）
    per_kb = int(state.get("per_kb_k") or 0) or int(
        getattr(ctx.settings, "retrieval_per_kb_k", 3))
    total = int(state.get("total_k") or 0) or int(
        getattr(ctx.settings, "retrieval_total_k", 5))
    per_kb = max(1, min(per_kb, 20))
    total = max(1, min(total, 50))
    # 聚合返回：父块名额（0=关闭，全返回小 chunk；None 或 <0 = 全局配置默认）
    parent_groups = state.get("parent_groups")
    if parent_groups is None or parent_groups < 0:
        parent_groups = int(getattr(ctx.settings, "retrieval_parent_groups", 3))
    group_size = int(getattr(ctx.settings, "retrieval_parent_group_size", 3))
    max_chars = int(getattr(ctx.settings, "retrieval_parent_max_chars", 4000))
    # 检索词：路由改写的中/英检索词（空则回退原始 query 单路）。
    queries = [q for q in (state.get("search_queries") or []) if q] \
        or [state["query"]]
    # 多查询（路由改写词 >1）且库服务支持 search_multi：查询维度下沉到库层
    # （腿内先跨查询 RRF、腿间再 RRF），跨库在其结果上合并排序；
    # 单查询或无该接口（测试假服务）→ 旧路径（逐查询检索 + 查询间 RRF）。
    if len(queries) > 1 and callable(getattr(ctx.kb_service, "search_multi", None)):
        hits: list[dict] = []
        for kb in targets:
            try:
                hits.extend(await asyncio.to_thread(
                    ctx.kb_service.search_multi, kb.kb_id, queries, k=per_kb,
                    user_id=state["user_id"], mode=mode))
            except Exception as e:
                logger.warning("kb search failed, skip kb=%s(%s): %s",
                               kb.name, kb.kb_id, e)
                emit("retrieve_error", {"kb_id": kb.kb_id, "kb_name": kb.name,
                                        "error": str(e)[:200]})
                continue
        hits.sort(key=_hit_rank_key)             # 跨库合并：沿用排序键
    else:
        hit_lists: list[list[dict]] = []
        for q in queries:
            q_hits: list[dict] = []
            for kb in targets:
                # 检索链路全同步（查询嵌入 HTTP + Chroma + BM25 磁盘读），必须放线程池。
                # 单库隔离：一个库坏掉（嵌入维度不匹配/端点不可达/磁盘缺文件）
                # 不拖垮整轮对话——记日志、推 retrieve_error 事件供前端提示，
                # 继续其余健康库；全部失败时 hits 为空，generate 自然按自身知识兜底。
                try:
                    kb_hits = await asyncio.to_thread(
                        ctx.kb_service.search, kb.kb_id, q, k=per_kb,
                        user_id=state["user_id"], mode=mode)
                except Exception as e:
                    logger.warning("kb search failed, skip kb=%s(%s): %s",
                                   kb.name, kb.kb_id, e)
                    emit("retrieve_error", {"kb_id": kb.kb_id, "kb_name": kb.name,
                                            "error": str(e)[:200]})
                    continue
                q_hits.extend(kb_hits)
            if q_hits:
                q_hits.sort(key=_hit_rank_key)   # 同查询内：沿用跨库合并排序
                hit_lists.append(q_hits)
        if len(hit_lists) > 1:                   # 多路：RRF 融合（同 id 合并、融合分写入 score）
            hits = rrf_fuse(*hit_lists, top=sum(len(h) for h in hit_lists))
        else:
            hits = hit_lists[0] if hit_lists else []
    if parent_groups > 0:
        # 候选池要留足余量：已展开父块组内的命中会被跳过，多取一些候选
        # （_expand_parent_blocks 内部 get_parent_block 走磁盘，同样放线程池）
        top = await asyncio.to_thread(
            _expand_parent_blocks, ctx,
            hits[:max(total + parent_groups * 2, len(hits))],
            parent_budget=min(parent_groups, total),
            group_size=max(1, group_size),
            max_chars=max(200, max_chars),
            total=total)
    else:
        top = [{**h, "type": "chunk"} for h in hits[:total]]
    # 引用溯源推流：前端据此展示检索来源面板（text 截断，避免事件过大）
    emit("retrievals", {"mode": mode, "results": [
        {"type": h.get("type", "chunk"), "kb_name": h.get("kb_name"),
         "scope": h.get("scope"),
         "distance": h.get("distance"), "bm25_score": h.get("bm25_score"),
         "score": h.get("score"), "rerank_score": h.get("rerank_score"),
         "method": h.get("method"),
         "source": h.get("source") or (h.get("metadata") or {}).get("source"),
         "page": (h.get("metadata") or {}).get("page"),
         "pages": h.get("pages"), "hit_chunks": h.get("hit_chunks"),
         "text": str(h.get("text", ""))[:300]}
        for h in top]})
    out = {"retrievals": top,
           # 第一优先：落一份跨轮检索状态给下一轮的 supervisor 用。
           # hit_count>0 表示"上一轮检索确有命中、回答大概率基于证据"——
           # 供"延续话题免重复检索"的规则判断。kb_names 只用于提示文案。
           "last_retrieval_state": {
               "kb_ids": [kb.kb_id for kb in targets],
               "kb_names": [kb.name for kb in targets],
               "hit_count": len(top)}}
    # 检索块并入最后一条用户消息（随 checkpoint 持久化）：后续轮次仍带本轮
    # 资料，每轮请求序列包含上一轮全部内容——供应商前缀缓存命中全部历史
    block = _retrieval_block(top)
    if block:
        patch = _retrieval_patch(state["messages"], block)
        if patch is not None:
            out["messages"] = [patch]
    return out


# ---------- 标准 LangChain 模型流（推理扩展由模型适配层保留） ----------

def _native_cfg(ctx: WorkflowContext, user_id: str) -> LLMConfig | None:
    """用户生效配置；轻量测试服务没有此接口时返回 None。

    连接参数只交给标准模型流协调器，图节点不直接创建供应商 SDK 客户端。
    """
    getter = getattr(ctx.llm_service, "effective_config", None)
    if not callable(getter):
        return None
    try:
        return getter(user_id)
    except Exception:
        return None


def _native_temperature(state: AgentState) -> float:
    """生成温度：与 get_chat_model 的按轮覆盖语义一致
    （None → 默认 0.3；夹取 -2~2）。"""
    temp = state.get("temperature")
    if temp is None:
        return DEFAULT_TEMPERATURE
    return max(-2.0, min(2.0, float(temp)))


async def generate_node(ctx: WorkflowContext, state: AgentState,
                        config: RunnableConfig | None = None) -> dict:
    """通过标准模型流生成答案，保留工具调用、推理、用量与 Runnable 上下文。

    模型适配层保留第三方推理字段，统一流协调器处理事件与停止。
    工具轮撤回推理，最终答案轮定格；完整消息仍进入 checkpoint。
    """
    model = await asyncio.to_thread(
        ctx.llm_service.get_chat_model,
        state["user_id"], temperature=state.get("temperature"))
    schemas: list[dict] = []
    if ctx.mcp_adapter is not None:
        schemas = await ctx.mcp_adapter.schemas_for_llm() or []
    native_cfg = _native_cfg(ctx, state["user_id"])
    system = SystemMessage(content=_build_system_prompt(state))
    # 检索块已由 retrieve 节点并入最后一条用户消息（随 checkpoint 持久化），
    # 发送序列 = system + 全部历史：跨轮请求前缀稳定、缓存命中全部历史
    payload = await asyncio.to_thread(
        hydrate_messages, [system] + list(state["messages"]),
        ctx.multimodal_service, state["user_id"], state.get("multimodal"))
    log_id: str | None = None                     # 追踪行 id：回写 state 供工具挂父子
    if ctx.tracer is not None:
        log_id = await asyncio.to_thread(
            ctx.tracer.start, "llm", getattr(model, "model_name", "chat"),
            state["session_id"], state["user_id"])
    resp = None
    stopped = False
    sid = state["session_id"]
    last_chunk_usage = None                        # 末块权威用量（防聚合虚高）
    generation_temp = _native_temperature(state)

    # 空响应防御：部分供应商偶发返回"零内容、零工具调用"的空完成，
    # 旧逻辑视作正常结束——前端表现为没有任何输出就静默收尾。现在对这种
    # 轮次做快速指数退避的整轮流式重试（与 llm_retry 的网络重试分层：那层管
    # 连接/限流，这层管"连上了但什么都没说"）；重试耗尽仍为空则显式抛错，
    # 走 SSE error 通道给用户明确提示——绝不静默结束。
    EMPTY_RETRY_MAX = 2
    for attempt in range(EMPTY_RETRY_MAX + 1):
        resp = None
        last_chunk_usage = None
        stopped = False
        resp, last_chunk_usage, stopped = await _native_round(
            native_cfg, payload, schemas or None, generation_temp, sid,
            config=config, model=model)
        if resp is None:
            resp = AIMessage(content="")
        if stopped:
            break                                  # 停止：跳过空响应判定，直接收尾
        has_tool_calls = bool(getattr(resp, "tool_calls", None))
        valid = bool(text_content(resp.content).strip()) or has_tool_calls
        # 工具轮和空完成撤回推理，只有最终答案轮定格展示。
        if valid and not has_tool_calls:
            emit("reasoning_end", {})
        else:
            emit("reasoning_discard", {})
        if valid:
            break                                  # 有内容或有工具调用：有效轮次
        if attempt < EMPTY_RETRY_MAX:              # 空响应：退避后整轮重来
            wait = 0.5 * (2 ** attempt)
            logger.warning("模型返回空响应（第 %d 次），%.1fs 后整轮重试 model=%s",
                           attempt + 1, wait,
                           getattr(model, "model_name", "chat"))
            await asyncio.sleep(wait)

    if stopped:
        # 中断收尾：清标记（不影响下一轮）、剥离半截 tool_calls（防误进工具循环），
        # 部分答复正常写入 checkpoint（前端已累积的打字机文本与历史一致）
        clear_stop(sid)
        resp = AIMessage(content=text_content(resp.content))
        emit("stopped", {"chars": len(text_content(resp.content))})
        emit("reasoning_discard", {})              # 停止后思考面板不留存

    answer = text_content(resp.content)
    if ctx.tracer is not None:
        await asyncio.to_thread(ctx.tracer.success, log_id, answer[:2000])

    # 真实用量：末块权威值优先，聚合值仅兜底（见 _usage_of 的警示）
    usage = last_chunk_usage or _usage_of(resp)

    # 计量落库：每轮真实用量入 llm_usage 表，供 /usage/summary 报表
    # 与后续配额演进使用。best-effort：失败只记日志，绝不影响主对话。
    if usage and usage.get("total_tokens"):
        def _persist_usage() -> None:
            from app.core.db import SessionLocal
            from app.models import LLMUsage
            with SessionLocal() as db:
                db.add(LLMUsage(
                    user_id=state["user_id"], session_id=sid,
                    model=str(getattr(model, "model_name", "chat")),
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                    cached_tokens=int(usage.get("cached_tokens") or 0)))
                db.commit()
        try:
            await asyncio.to_thread(_persist_usage)
        except Exception as e:
            logger.warning("usage persist failed (不影响对话): %s", e)

    # 重试耗尽仍为空（且非用户主动停止）：显式报错走 SSE error 通道，
    # 让前端给出明确提示——绝不静默结束让用户面对空白回复。
    if not stopped and not answer.strip() \
            and not getattr(resp, "tool_calls", None):
        raise RuntimeError(
            "模型连续返回空响应（已自动重试），请稍后重试或更换模型/提供商")

    return {"messages": [resp], "answer": answer,
            "last_usage": usage, "stopped": stopped,
            "llm_log_id": log_id}                 # 工具执行节点据此把调用挂成本轮 LLM 的子节点

from dataclasses import dataclass
import asyncio
import json
import random
import re
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from app.core.cancel import clear_stop, is_stopped
from app.core.events import emit
from app.core.logging import get_logger
from app.services.memory_service import MEMORY_MAX, MemoryService
from app.graph.state import AgentState
from app.abstractions.llm import DEFAULT_TEMPERATURE, _is_retryable

logger = get_logger("graph_nodes")


@dataclass
class WorkflowContext:
    """编排层依赖注入：节点所需的 llm / 知识库等服务，编译时注入。"""
    settings: object
    llm_service: object
    kb_service: object
    mcp_adapter: object | None = None
    tracer: object | None = None
    memory_service: MemoryService | None = None

# ---------- 长记忆----------
EXTRACT_PROMPT = """从这段对话中提取值得记住的用户信息（研究主题、偏好、
项目背景等），输出 JSON：
{"memories":[{"key":"snake_case键名","value":"简短值",
              "tier":"core或short","topic":"主题标签（如 研究方向/项目/偏好）"}]}
tier 判定：研究方向、领域、稳定偏好、身份信息等长期有效 → core；
有时效性的事实（正在写某论文、本周任务、当前项目阶段）→ short。
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


# ---------- 自动上下文压缩 ----------
COMPACT_KEEP_ROUNDS = 4          # 压缩后保留的最近轮数
COMPACT_MIN_ROUNDS = 20          # 轮数超过该值触发压缩（>20 轮）
COMPACT_MIN_TOKEN_RATIO = 0.2    # 轮数路径的最小占用门槛

COMPACT_PROMPT = """您正在进行上下文检查点压缩。请为另一个LLM创建一个交接摘要，以便其继续本次对话。

内容包括：
- 用户的研究主题、关键问题及迄今得出的结论
- 对话中已确认的事实、参数和用户偏好
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


async def compact_node(ctx: WorkflowContext, state: AgentState) -> dict:
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
        _t0 = time.perf_counter()
        resp = await model.ainvoke([SystemMessage(content=COMPACT_PROMPT)] + old)
        logger.info("compact summarized in %.1fs old_rounds=%d chars=%d",
                    time.perf_counter() - _t0, len(old),
                    len(str(resp.content or "")))
        summary = str(resp.content or "").strip()
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
ROUTE_PROMPT = """你是问答路由，负责判断用户提问是否需要查询知识库，以及查哪些库。

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
- 判定标准：想象"没有知识库，这个问题能否答好"——能答好就不检索；
  只有确实需要库内具体内容才检索
- 拿不准时倾向于需要检索，宁可多选一个相关的库也不漏掉
- 【延续话题，免重复检索】若下方给出了 [上一轮检索状态]，且当前提问只是对
  上一轮已回答话题的延续/追问/总结（没有引入新的知识需求），则不需要检索。

注意：你只输出 JSON 判定，不跟用户对话、不接梗、不闲聊、不解释。

只输出 JSON，不要任何其他文字：
{{"needs_retrieval": true或false, "kbs": [{{"name": "库名", "scope": "public或private"}}]}}
不需要检索时 kbs 为 []。"""


def _parse_route(text: str) -> dict:
    """解析路由 LLM 返回的 JSON（容忍围栏/夹带文字）。解析失败返回空 dict。"""
    return _loads_fuzzy(text) or {}


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
    text = str(getattr(msgs[-1], "content", "") or "").strip()
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


async def load_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
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


async def extract_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """对话结束后抽取值得记住的信息：交给 LLM 从对话中提炼。

    重试耗尽仍失败时静默跳过——记忆抽取是锦上添花，绝不能打挂主对话。
    """
    if ctx.memory_service is None:
        return {"new_memories": []}
    try:
        model = await asyncio.to_thread(
            ctx.llm_service.get_chat_model,
            state["user_id"], temperature=state.get("temperature"))
        system = SystemMessage(content=EXTRACT_PROMPT)
        resp = await model.ainvoke([system] + state["messages"][-4:])   # 只看最近几轮
        memories = _parse_memories(str(resp.content or ""))
    except Exception as e:
        logger.warning("memory extract failed, skip: %s", e)
        memories = []
    emit("memory_extract", {"candidates": len(memories)})
    return {"new_memories": memories}


async def save_memory_node(ctx: WorkflowContext, state: AgentState) -> dict:
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
            stat = await _maintain_memories(ctx, state["user_id"])
            emit("memory_maintain", {"before": count, **stat})
    except Exception as e:
        logger.warning("memory maintain failed: %s", e)
    emit("memory_save", {"saved": saved, **stat})
    return {}


async def _maintain_memories(ctx: WorkflowContext, user_id: str) -> dict:
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
            merged = await _merge_memories_llm(ctx, user_id, payload)
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
                              groups: dict) -> list[dict] | None:
    """一次 LLM 调用批处理所有分组的合并；失败返回 None（降级 LRU）。"""
    try:
        model = await asyncio.to_thread(
            ctx.llm_service.get_chat_model, user_id, temperature=0.0)
        resp = await model.ainvoke([
            SystemMessage(content=MERGE_PROMPT),
            HumanMessage(content=json.dumps(groups, ensure_ascii=False))])
        data = _loads_fuzzy(str(resp.content or ""))
        return data.get("merged", []) if data else None
    except Exception as e:
        logger.warning("memory merge LLM call failed: %s", e)
        return None

# 固定身份句：永不变化，置于系统提示词最前，保证所有请求共享同一前缀段
_IDENTITY = ("你是科研助手。若用户消息后附有【知识库检索结果】，优先据此回答，"
             "引用时标明来源（public/private）；否则根据你自己的知识回答。")


def _build_system_prompt(state: AgentState) -> str:
    """组装稳定的系统提示词前缀：固定身份 → 用户记忆（定序）→ 历史总结（低频变化）。

    检索结果不进 system（每轮都变，放前缀会打穿供应商侧前缀缓存），
    改由 _retrieval_message 打包、_compose_llm_messages 插在最后一条
    用户消息之后——system 与对话历史跨轮字节级一致，命中 KV 前缀缓存。
    """
    parts = [_IDENTITY]
    if state.get("memory"):
        # sort_keys：键序不随 DB 返回顺序漂移（get_all 已 ORDER BY key，双保险）
        parts.append(f"[用户记忆] "
                     f"{json.dumps(state['memory'], ensure_ascii=False, sort_keys=True)}")
    if state.get("conversation_summary"):
        parts.append(f"[历史对话总结] {state['conversation_summary']}")
    return "\n\n".join(parts)


def _retrieval_message(state: AgentState) -> HumanMessage | None:
    """把本轮检索结果打包为一条临时消息（只进发送序列，不写 checkpoint）。

    角色用 HumanMessage 而非 SystemMessage：消息序列中间的 system
    是分布外排列，会触发 thinking 模型长思考/重复循环（A/B 实测思考
    2988 字→100 字）；user 角色是标准 RAG 形态。原生路径翻译时相邻 user
    会合并，最终仍以"资料附在问题末尾"的单条 user 发出。
    """
    if not state.get("retrievals"):
        return None
    lines = ["【知识库检索结果】回答本问题时优先依据以下内容，引用标明来源。"]
    for r in state["retrievals"]:
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
    return HumanMessage(content="\n".join(lines))


def _compose_llm_messages(state: AgentState, system: SystemMessage,
                          retrieval: HumanMessage | None) -> list:
    """发送给 LLM 的消息序列：检索块插在最后一条用户消息之后。

    - 不拼进用户消息内容：上轮请求是本序列的严格前缀，供应商前缀缓存
      （按 token 前缀匹配）可命中全部历史与 system
    - 工具循环第二轮（末尾已是 AI(tool_calls)/ToolMessage）仍按
      「最后 human 索引」插入，轮内两次请求前缀完全一致
    - 检索块不进 messages reducer → checkpoint 不被每轮检索结果污染
    """
    msgs = list(state["messages"])
    if retrieval is None:
        return [system] + msgs
    human_idx = [i for i, m in enumerate(msgs) if getattr(m, "type", "") == "human"]
    idx = human_idx[-1] if human_idx else len(msgs) - 1   # 无 human 时兜底置尾
    return [system] + msgs[:idx + 1] + [retrieval] + msgs[idx + 1:]


async def _native_route_round(cfg, prompt, msgs_lc, temperature, sid,
                              use_json=True):
    """supervisor 路由的原生流式调用：思考过程实时以 routing_reasoning
    事件透传——thinking 模型的路由可能耗时数十秒，让前端看到模型在工作
    而不是静默卡住。返回模型输出文本（应为 JSON）。

    与 generate 的区别：无"最终答案轮"概念，不需要 reasoning_end/discard；
    use_json 时带 response_format=json_object（端点不支持由调用方回退重试）。
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url,
                         timeout=180.0, max_retries=0)
    msgs = _to_openai_messages([SystemMessage(content=prompt)] + list(msgs_lc))
    payload = {"model": cfg.model_id, "messages": msgs,
               "temperature": temperature, "stream": True}
    if use_json:
        payload["response_format"] = {"type": "json_object"}
    last_err = None
    for attempt in range(2):          # 路由轻量：网络抖动最多补一次
        try:
            stream = await client.chat.completions.create(**payload)
            parts: list[str] = []
            async for chunk in stream:
                if is_stopped(sid):    # 用户停止：返回残文走既有降级路径即可
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                rc = (getattr(delta, "reasoning_content", None)
                      or getattr(delta, "reasoning", None))
                if isinstance(rc, list):   # 分片数组形态 → 拼接
                    rc = "".join((p.get("text") or p.get("summary") or "")
                                 for p in rc if isinstance(p, dict))
                if isinstance(rc, str) and rc:
                    emit("routing_reasoning", {"content": rc})
                if delta.content:
                    parts.append(delta.content)
            try:
                await stream.close()
            except Exception:
                pass
            return "".join(parts)
        except Exception as e:
            last_err = e
            if attempt >= 1 or not _is_retryable(e):
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
    raise last_err or RuntimeError("unreachable")


async def supervisor_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """路由决策（LLM 意图判断）：先看用户有没有可见知识库。

    - 没有可见库 → 无需检索，直接生成
    - 有可见库 → 把「库名 + scope」目录交给 LLM，让它判断本次提问
      是否需要检索、以及选哪几个库（按名称），再做可见性校验后落 state
    - LLM 判断异常/解析失败 → 降级为全部可见库检索（保持 RAG 兜底）
    """
    # 只用"可检索"的库（用户可自行禁用某库参与对话检索）
    kbs = await asyncio.to_thread(ctx.kb_service.list_queryable_kbs,
                                  state["user_id"])
    if not kbs:
        emit("supervisor", {"needs_retrieval": False, "kb_count": 0, "selected": []})
        return {"needs_retrieval": False, "selected_kb_ids": [],
                "last_retrieval_state": {"kb_ids": [], "kb_names": [], "hit_count": 0}}

    catalog = [{"name": kb.name, "scope": kb.scope,
                "description": kb.description or ""} for kb in kbs]
    selected: list = []
    # 路由前置事件：thinking 模型的路由调用可能耗时数十秒且此前前端无任何
    # 反馈（memory_load 与 supervisor 之间静默）——先告诉前端"正在判断意图"。
    emit("routing", {})
    route_cfg = _native_cfg(ctx, state["user_id"])   # 路由也走原生 SDK（思考可见）
    msgs_route = state["messages"][-10:]
    try:
        prompt = ROUTE_PROMPT.format(catalog=json.dumps(catalog, ensure_ascii=False))
        # P3-24：路由输入只带最近 4 条消息，指代上文的追问（"接着刚才那个
        # 方案说"）会因看不到上文被误判为无需检索——把压缩总结（低频变化、
        # 已定序，不破坏前缀缓存）附在目录后，让路由 LLM 知道"刚才在聊什么"。
        summary = str(state.get("conversation_summary") or "").strip()
        if summary:
            prompt += "\n\n[历史对话总结]\n" + summary[:500]
        # 第一优先（P3-33）：把上一轮的检索结果状态喂给路由——延续话题且上轮
        # 已命中可免重复检索。只拼进路由 prompt，绝不进 system/历史（保前缀缓存）。
        lrs = state.get("last_retrieval_state") or {}
        if lrs.get("hit_count"):
            prompt += ("\n\n[上一轮检索状态] 上一轮检索了知识库「%s」，共命中 %d 条。"
                       % ("、".join(lrs.get("kb_names") or ["?"]), lrs["hit_count"]))
        # 强制 JSON 输出（P3-35）：带历史上下文时模型偶发"聊天式"输出——用户
        # 玩梗（"牛来！"）时模型接梗不输出 JSON → 解析失败走全库检索降级，
        # 无关话题也会带上检索。json_object 模式从源头压住；端点不支持时回退
        # 普通调用（老版本网关/部分自建端点没有 JSON 模式）。
        _t0 = time.perf_counter()
        if route_cfg is not None:
            # 原生流式：思考过程以 routing_reasoning 事件实时透传
            try:
                text = await _native_route_round(
                    route_cfg, prompt, msgs_route,
                    _native_temperature(state), state["session_id"],
                    use_json=True)
            except Exception as e:
                if ("response_format" not in str(e).lower()
                        and "json_object" not in str(e).lower()):
                    raise
                logger.warning("router json mode unsupported, fallback plain: %s", e)
                text = await _native_route_round(
                    route_cfg, prompt, msgs_route,
                    _native_temperature(state), state["session_id"],
                    use_json=False)
        else:
            # 测试假服务回退 langchain 路径（行为与旧版一致）
            model = await asyncio.to_thread(
                ctx.llm_service.get_chat_model,
                state["user_id"], temperature=state.get("temperature"))
            try:
                resp = await model.ainvoke(
                    [SystemMessage(content=prompt)] + msgs_route,
                    response_format={"type": "json_object"})
            except Exception as e:
                if ("response_format" not in str(e).lower()
                        and "json_object" not in str(e).lower()):
                    raise
                logger.warning("router json mode unsupported, fallback plain: %s", e)
                resp = await model.ainvoke([SystemMessage(content=prompt)] + msgs_route)
            text = str(resp.content or "")
        logger.info("supervisor routed in %.1fs msgs=%d",
                    time.perf_counter() - _t0, len(msgs_route))
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


async def tool_executor_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """执行 LLM 请求的工具：经 MCPToolAdapter → MCP tools/call，结果回灌（§7.6）。"""
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        emit("tool_start", {"name": call["name"], "args": call["args"]})
        out = await ctx.mcp_adapter.call(call["name"], call["args"],
                                         state["session_id"], state["user_id"])
        results.append(ToolMessage(content=json.dumps(out, ensure_ascii=False),
                                   tool_call_id=call["id"]))
    return {"messages": results}

def _hit_rank_key(h: dict) -> float:
    """跨库合并排序键：hybrid 结果按 RRF 融合分（越大越好），vector 按距离（越小越好）。"""
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


async def retrieve_node(ctx: WorkflowContext, state: AgentState) -> dict:
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
    hits = []
    for kb in targets:
        # 检索链路全同步（查询嵌入 HTTP + Chroma + BM25 磁盘读），必须放线程池。
        # P1-5 单库隔离：一个库坏掉（嵌入维度不匹配/端点不可达/磁盘缺文件）
        # 不拖垮整轮对话——记日志、推 retrieve_error 事件供前端提示，
        # 继续其余健康库；全部失败时 hits 为空，generate 自然按自身知识兜底。
        try:
            kb_hits = await asyncio.to_thread(
                ctx.kb_service.search, kb.kb_id, state["query"], k=per_kb,
                user_id=state["user_id"], mode=mode)
        except Exception as e:
            logger.warning("kb search failed, skip kb=%s(%s): %s",
                           kb.name, kb.kb_id, e)
            emit("retrieve_error", {"kb_id": kb.kb_id, "kb_name": kb.name,
                                    "error": str(e)[:200]})
            continue
        hits.extend(kb_hits)
    hits.sort(key=_hit_rank_key)
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
         "score": h.get("score"), "method": h.get("method"),
         "source": h.get("source") or (h.get("metadata") or {}).get("source"),
         "page": (h.get("metadata") or {}).get("page"),
         "pages": h.get("pages"), "hit_chunks": h.get("hit_chunks"),
         "text": str(h.get("text", ""))[:300]}
        for h in top]})
    return {"retrievals": top,
            # 第一优先（P3-33）：落一份跨轮检索状态给下一轮的 supervisor 用。
            # hit_count>0 表示"上一轮检索确有命中、回答大概率基于证据"——
            # 供"延续话题免重复检索"的规则判断。kb_names 只用于提示文案。
            "last_retrieval_state": {
                "kb_ids": [kb.kb_id for kb in targets],
                "kb_names": [kb.name for kb in targets],
                "hit_count": len(top)}}


# ---------- generate 原生思考流（所有端点统一原生 SDK） ----------
# langchain-openai（1.2.2，升级 1.6.0 亦同——已源码验证）不提取 OpenAI 兼容
# 协议里的推理字段：delta→chunk 转换只取 content/function_call/tool_calls，
# reasoning_content 在转换层即被丢弃——langchain 路径上想试也没得试。
# 背景：thinking 模型（DeepSeek V4 等）复杂问题会先"思考"数十秒，期间
# content 为空 → 前端收不到任何 token，体感像卡死。
# 方案：generate 统一走原生 openai SDK 流式，对所有端点默认尝试捕获推理
# 内容，兼容两种方言键（DeepSeek 系 reasoning_content / 新事实标准
# reasoning）；无思考的模型零命中、前端空面板不渲染。仅当拿不到生效配置
# （测试假服务）才回退原 langchain 路径。工具调用轮的思考在轮末撤回，
# 最终输出轮的思考定格。

def _native_cfg(ctx: WorkflowContext, user_id: str):
    """generate 的生效 LLM 配置（用户配置 > 系统默认）。不再按端点名筛选：
    所有端点统一走原生 SDK 以尝试捕获推理内容；拿不到配置（测试假服务）
    返回 None → 回退 langchain 路径，行为与旧版一致。"""
    getter = getattr(ctx.llm_service, "effective_config", None)
    if not callable(getter):
        return None
    try:
        return getter(user_id)
    except Exception:
        return None


def _native_temperature(state: AgentState) -> float:
    """原生路径的生成温度：与 get_chat_model 的按轮覆盖语义一致
    （None → 默认 0.3；夹取 -2~2）。"""
    temp = state.get("temperature")
    if temp is None:
        return DEFAULT_TEMPERATURE
    return max(-2.0, min(2.0, float(temp)))


def _to_openai_messages(payload: list) -> list[dict]:
    """LangChain 消息序列 → 原生 openai SDK 的 Chat Completions 格式。

    相邻 user 消息合并为一条：检索块以 HumanMessage 插在问题之后，
    合并后即"资料附在问题末尾"的**单条 user**——A/B 实测该形态思考最少
    （中间 system 形态会触发 thinking 模型长思考/重复循环，2988 字→100 字）。
    system 只保留队首一条，前缀缓存不受影响。
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


def _to_openai_tools(schemas: list[dict]) -> list[dict]:
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


def _usage_from_native(usage) -> dict | None:
    """原生 SDK 末块 usage（CompletionUsage）→ 与 _usage_from 相同的内部格式。"""
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


async def _langchain_round(model, payload, sid):
    """langchain 路径的一轮流式（原 generate_node 的 astream 循环，逻辑原样）。"""
    resp = None
    last_usage = None
    stopped = False
    async for chunk in model.astream(payload):
        if is_stopped(sid):                    # 用户点了停止：保留已生成部分
            stopped = True
            break
        resp = chunk if resp is None else resp + chunk   # 逐 token 聚合为完整消息
        # 用量取"最后一个携带 usage 的 chunk"：供应商要么只在末块带、要么每
        # 块带同一份累计快照，两种语义下末块都等于真实总量；绝不能信聚合后
        # 的 resp——LangChain 会把重复出现的 usage 累加成数十倍虚高
        cu = _usage_from(chunk)
        if cu is not None:
            last_usage = cu
        # 每个 token 实时推给事件总线（SSE 端点持续 drain → 前端打字机效果）
        text = chunk.content
        if isinstance(text, str):
            if text:
                emit("token", {"content": text})
        elif isinstance(text, list):
            for part in text:
                if isinstance(part, dict) and part.get("text"):
                    emit("token", {"content": part["text"]})
    return resp, last_usage, stopped


_zero_reasoning_seen: set[str] = set()   # 已记录过"零推理"的端点键（每进程一次）


def _log_zero_reasoning_once(cfg) -> None:
    """整轮零推理内容时记一次日志：帮助区分'模型本就不思考'与'方言字段对不上'
    （如新版 vLLM 改发 reasoning 以外的键）这类静默降级。"""
    key = f"{getattr(cfg, 'base_url', '')}::{getattr(cfg, 'model_id', '')}"
    if key in _zero_reasoning_seen:
        return
    _zero_reasoning_seen.add(key)
    logger.info("端点未返回任何推理内容（若该模型应有思考过程，请检查方言字段）"
                " base=%s model=%s",
                getattr(cfg, "base_url", ""), getattr(cfg, "model_id", ""))


async def _native_round(cfg, payload, tools, temperature, sid):
    """原生 openai SDK 的一轮流式（统一路径）：捕获推理内容。

    - reasoning_content 流式 emit("reasoning")；是否展示由调用方轮末判定
      （工具轮 reasoning_discard 撤回 / 最终输出轮 reasoning_end 定格）。
    - 网络层整轮重试与 RetryableChatModel.astream 同语义：流中断从头重来，
      已 emit 的半截内容会重复（预期行为）。
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url,
                         timeout=180.0, max_retries=0)
    msgs = _to_openai_messages(payload)
    max_retries, base_delay = 10, 1.0
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
                cu = _usage_from_native(getattr(chunk, "usage", None))
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
                # 多方言兼容：DeepSeek 系 reasoning_content / 新事实标准 reasoning。
                # 优先级：reasoning_content 命中即不再看 reasoning；两者皆无 →
                # 零命中降级（不发事件，答案流照常，轮末记一次日志）
                rc = (getattr(delta, "reasoning_content", None)
                      or getattr(delta, "reasoning", None))
                if isinstance(rc, list):    # 分片数组形态 [{"text"/"summary": ..}]
                    rc = "".join((p.get("text") or p.get("summary") or "")
                                 for p in rc if isinstance(p, dict))
                if isinstance(rc, str) and rc:   # 类型守卫：非字符串不透传
                    had_reasoning = True
                    emit("reasoning", {"content": rc})
                if delta.content:           # 正式答案：与 langchain 路径同款 token 事件
                    content_parts.append(delta.content)
                    emit("token", {"content": delta.content})
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
            # 截断检测（P3-34 后续）：网关可能静默切断长生成——流正常结束但
            # 既无 finish_reason 也无 usage（实测 ox-alpha-free 长回答在半句
            # 处戛然而止、末块 usage 缺失）。首次遇此整轮重试一次；重试仍截断
            # 则接受部分内容并记 warning（部分答复好于没有，但要可观测）。
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
            # 注意：本版本 langchain-core 的 AIMessage 不接受 tool_calls=None
            # （必须传 list，空列表 OK）——不传 None 以免触发 pydantic 校验错误
            if not had_reasoning and not stopped:
                _log_zero_reasoning_once(cfg)
            resp = AIMessage(content="".join(content_parts),
                             tool_calls=tool_calls)
            return resp, last_usage, stopped
        except Exception as e:
            if attempt >= max_retries or not _is_retryable(e):
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("native llm stream broken (attempt %d/%d) model=%s: %s; retry in %.1fs",
                           attempt + 1, max_retries, cfg.model_id, e, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


async def generate_node(ctx: WorkflowContext, state: AgentState) -> dict:
    """用（用户级）LLM 生成答复：系统提示词带记忆/检索结果 + 用户消息。

    - 统一路径：原生 openai SDK 流式逐 token 生成（前端打字机效果），并对
      所有端点默认尝试捕获推理内容（reasoning_content / reasoning 双方言），
      以 reasoning 事件透传——工具调用轮的思考撤回（reasoning_discard），
      最终输出轮定格（reasoning_end）；无思考的模型零命中、面板不渲染。
    - 仅当拿不到生效配置（测试假服务）时回退 langchain astream 老路径。
    聚合后的完整 message 仍写入 state，tool_calls 也随之聚合，
    不影响 generate ⇄ tool_executor 循环。
    """
    model = await asyncio.to_thread(
        ctx.llm_service.get_chat_model,
        state["user_id"], temperature=state.get("temperature"))
    schemas: list[dict] = []
    if ctx.mcp_adapter is not None:
        schemas = await ctx.mcp_adapter.schemas_for_llm() or []
    # 所有端点统一走原生 SDK——langchain-openai 会丢弃推理字段，langchain
    # 路径上无法"尝试"；拿不到配置（假服务）才回退老路径
    native_cfg = _native_cfg(ctx, state["user_id"])
    if schemas and native_cfg is None:
        model = model.bind_tools(schemas)      # 告诉 LLM"你有这些工具可用"
    system = SystemMessage(content=_build_system_prompt(state))
    # 检索块作为临时消息插在最后一条用户消息后（不进 checkpoint），
    # 保证发送序列跨轮前缀稳定 → 供应商侧前缀缓存可命中历史
    payload = _compose_llm_messages(state, system, _retrieval_message(state))
    if ctx.tracer is not None:
        log_id = await asyncio.to_thread(
            ctx.tracer.start, "llm", getattr(model, "model_name", "chat"),
            state["session_id"], state["user_id"])
    resp = None
    stopped = False
    sid = state["session_id"]
    last_chunk_usage = None                        # 末块权威用量（防聚合虚高）
    native_tools = _to_openai_tools(schemas) if native_cfg is not None else None
    native_temp = _native_temperature(state) if native_cfg is not None else None

    # 空响应防御（P3-31）：部分供应商偶发返回"零内容、零工具调用"的空完成，
    # 旧逻辑视作正常结束——前端表现为没有任何输出就静默收尾。现在对这种
    # 轮次做快速指数退避的整轮流式重试（与 llm_retry 的网络重试分层：那层管
    # 连接/限流，这层管"连上了但什么都没说"）；重试耗尽仍为空则显式抛错，
    # 走 SSE error 通道给用户明确提示——绝不静默结束。
    EMPTY_RETRY_MAX = 2
    for attempt in range(EMPTY_RETRY_MAX + 1):
        resp = None
        last_chunk_usage = None
        stopped = False
        if native_cfg is not None:
            resp, last_chunk_usage, stopped = await _native_round(
                native_cfg, payload, native_tools, native_temp, sid)
        else:
            resp, last_chunk_usage, stopped = await _langchain_round(
                model, payload, sid)
        if resp is None:
            resp = AIMessage(content="")
        if stopped:
            break                                  # 停止：跳过空响应判定，直接收尾
        has_tool_calls = bool(getattr(resp, "tool_calls", None))
        valid = bool(str(resp.content or "").strip()) or has_tool_calls
        if native_cfg is not None:
            # 思考面板收尾：工具轮的思考过程撤回（用户不需要看"该调什么工具"），
            # 空响应轮同样撤回；只有最终输出轮定格展示
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
        resp = AIMessage(content=str(resp.content or ""))
        emit("stopped", {"chars": len(str(resp.content or ""))})
        if native_cfg is not None:
            emit("reasoning_discard", {})          # 停止后思考面板不留存

    answer = str(resp.content) if resp.content else ""
    if ctx.tracer is not None:
        await asyncio.to_thread(ctx.tracer.success, log_id, answer[:2000])

    # 真实用量：末块权威值优先，聚合值仅兜底（见 _usage_of 的警示）
    usage = last_chunk_usage or _usage_of(resp)

    # P3-20 计量落库：每轮真实用量入 llm_usage 表，供 /usage/summary 报表
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
            "last_usage": usage, "stopped": stopped}
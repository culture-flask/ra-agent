# 前缀缓存优化单元测试：不依赖 DB，纯函数验证消息组装的前缀稳定性。
#
# - _build_system_prompt：system 前缀跨轮稳定（不随检索/记忆键序变化）
# - _retrieval_block / _retrieval_patch：检索块打包并并入最后一条用户消息
#   （同 id 替换、重复生成幂等、排在附件块之后）
# - 前缀性质：检索块持久化后，下一轮请求完整包含上一轮请求（缓存命中）
import pytest
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

from app.graph.nodes import (_RET_MARK, _build_system_prompt,
                             _retrieval_block, _retrieval_patch)


def _sig(messages: list) -> list[tuple]:
    """消息序列签名：前缀缓存按 (角色, 内容) 序列匹配。"""
    return [(getattr(m, "type", ""), str(m.content)) for m in messages]


# ---------- _build_system_prompt ----------

def test_system_prompt_ignores_retrieval_flip():
    """检索开关翻转不改变 system（消除二态分支）。"""
    marker = "检索正文XYZ"
    base = {"messages": [], "memory": {"k": {"v": "x"}}}
    with_ret = {**base, "retrievals": [{"type": "chunk", "text": marker,
                                        "scope": "public", "kb_name": "kb"}],
                "needs_retrieval": True}
    without_ret = {**base, "retrievals": [], "needs_retrieval": False}
    assert _build_system_prompt(with_ret) == _build_system_prompt(without_ret)
    assert marker not in _build_system_prompt(with_ret)


def test_system_prompt_memory_keys_sorted():
    """记忆 dict 注入序无关：输出按键排序（键序漂移不打穿前缀）。"""
    m1 = {"messages": [], "memory": {"alpha": {"v": "1"}, "zeta": {"v": "2"}}}
    m2 = {"messages": [], "memory": {"zeta": {"v": "2"}, "alpha": {"v": "1"}}}
    assert _build_system_prompt(m1) == _build_system_prompt(m2)
    prompt = _build_system_prompt(m1)
    assert prompt.index("alpha") < prompt.index("zeta")


def test_system_prompt_includes_summary_and_identity():
    """固定身份句在最前 + 历史总结保留（兼容 test_compact 断言）。"""
    state = {"messages": [], "conversation_summary": "用户研究量子计算"}
    prompt = _build_system_prompt(state)
    assert prompt.startswith("你是科研助手")
    assert "[历史对话总结]" in prompt


# ---------- _retrieval_block / _retrieval_patch ----------

def test_retrieval_block_none_when_empty():
    assert _retrieval_block([]) is None
    assert _retrieval_block(None) is None


def test_retrieval_block_formats():
    """父块（来源/页码/命中数）与小 chunk 两种格式。"""
    block = _retrieval_block([
        {"type": "parent", "text": "完整段落", "scope": "private", "kb_name": "我的库",
         "source": "a.pdf", "pages": [2, 3], "hit_chunks": 2},
        {"type": "chunk", "text": "小片段", "scope": "public", "kb_name": "公共库"},
    ])
    assert "【知识库检索结果】" in block
    assert "a.pdf 第2-3页" in block and "含2个命中片段" in block
    assert "(public / 公共库)" in block


def test_retrieval_patch_merges_into_last_human():
    """检索块并入最后一条用户消息：同 id 替换（不新增消息）、标记分隔。"""
    u1 = HumanMessage(content="问题一", id="m1")
    a1 = AIMessage(content="答一")
    u2 = HumanMessage(content="问题二", id="m2")
    patch = _retrieval_patch([u1, a1, u2], "R")
    assert patch.id == "m2"                       # 同 id → add_messages 原地替换
    assert patch.content == "问题二" + _RET_MARK + "R"


def test_retrieval_patch_idempotent_on_regen():
    """重新生成会重跑检索：先剥离旧块再追加 → 幂等不累积。"""
    u1 = HumanMessage(content="问题一", id="m1")
    first = _retrieval_patch([u1], "旧块")
    again = _retrieval_patch([first], "新块")
    assert again.content == "问题一" + _RET_MARK + "新块"
    assert again.content.count(_RET_MARK) == 1


def test_retrieval_patch_after_attachment_block():
    """附件块保留在前，检索块追加在整条消息末尾（回放拆解顺序：先检索后附件）。"""
    u1 = HumanMessage(content="问题\n<<<附件>>>\n[附件：a.pdf]\n正文", id="m1")
    patch = _retrieval_patch([u1], "R")
    assert patch.content.index("<<<附件>>>") < patch.content.index(_RET_MARK)


def test_retrieval_patch_skips_without_id():
    """消息缺 id / 无用户消息（理论不可达）：跳过改写，不影响主流程。"""
    assert _retrieval_patch([HumanMessage(content="无 id")], "R") is None
    assert _retrieval_patch([AIMessage(content="a")], "R") is None


# ---------- 前缀缓存性质 ----------

def test_payload_tool_loop_round2_prefix_stable():
    """工具循环：第二轮（尾部为 AI(tool_calls)+Tool）是第一轮的严格前缀扩展。"""
    system = SystemMessage(content="S")
    u1 = HumanMessage(content="问题" + _RET_MARK + "R", id="m1")
    ai_tool = AIMessage(content="", tool_calls=[
        {"name": "web_search", "args": {"query": "q"}, "id": "call_1"}])
    tool_msg = ToolMessage(content='{"ok": 1}', tool_call_id="call_1")
    round1 = [system, u1]
    round2 = [system, u1, ai_tool, tool_msg]
    assert _sig(round2)[:len(round1)] == _sig(round1)


def test_payload_cross_turn_covers_previous_request():
    """跨轮（持久化收益）：第 2 轮请求完整包含第 1 轮请求（含检索块）。"""
    system = SystemMessage(content="S")
    u1 = HumanMessage(content="问题一" + _RET_MARK + "R1")
    a1 = AIMessage(content="答一")
    u2 = HumanMessage(content="问题二" + _RET_MARK + "R2")
    turn1 = [system, u1]
    turn2 = [system, u1, a1, u2]
    assert _sig(turn2)[:len(turn1)] == _sig(turn1)

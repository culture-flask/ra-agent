# 前缀缓存优化单元测试：不依赖 DB，纯函数验证消息组装的前缀稳定性。
#
# 覆盖三个纯函数：
# - _build_system_prompt：system 前缀跨轮稳定（不随检索/记忆键序变化）
# - _retrieval_message：检索结果打包（父块/小 chunk/空检索）
# - _compose_llm_messages：检索块插在最后一条 human 之后；
#   轮内工具循环两次请求前缀一致；跨轮请求分叉点在旧 human 之后
import pytest
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

from app.graph.nodes import (_build_system_prompt, _compose_llm_messages,
                             _retrieval_message)


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


# ---------- _retrieval_message ----------

def test_retrieval_message_none_when_empty():
    assert _retrieval_message({"messages": [], "retrievals": []}) is None
    assert _retrieval_message({"messages": []}) is None


def test_retrieval_message_formats():
    """父块（来源/页码/命中数）与小 chunk 两种格式。"""
    state = {"retrievals": [
        {"type": "parent", "text": "完整段落", "scope": "private", "kb_name": "我的库",
         "source": "a.pdf", "pages": [2, 3], "hit_chunks": 2},
        {"type": "chunk", "text": "小片段", "scope": "public", "kb_name": "公共库"},
    ]}
    msg = _retrieval_message(state)
    assert isinstance(msg, SystemMessage)
    content = msg.content
    assert "【知识库检索结果】" in content
    assert "a.pdf 第2-3页" in content and "含2个命中片段" in content
    assert "(public / 公共库)" in content


# ---------- _compose_llm_messages ----------

def test_compose_passthrough_without_retrieval():
    """无检索：直接 [system] + messages。"""
    system = SystemMessage(content="S")
    msgs = [HumanMessage(content="hi")]
    out = _compose_llm_messages({"messages": msgs}, system, None)
    assert _sig(out) == _sig([system] + msgs)


def test_compose_inserts_after_last_human():
    """检索块插在最后一条 human 之后（不在 human 之前、不进 messages 尾部）。"""
    system = SystemMessage(content="S")
    retrieval = SystemMessage(content="R")
    msgs = [HumanMessage(content="u1"), AIMessage(content="a1"),
            HumanMessage(content="u2")]
    out = _compose_llm_messages({"messages": msgs}, system, retrieval)
    assert _sig(out) == _sig([system, msgs[0], msgs[1], msgs[2], retrieval])


def test_compose_tool_loop_round2_prefix_stable():
    """轮内工具循环：第二轮（尾部为 AI(tool_calls)+Tool）仍插在 human 后，
    且第二轮请求是第一轮的严格前缀扩展 → 前缀缓存全命中。"""
    system = SystemMessage(content="S")
    retrieval = SystemMessage(content="R")
    u1 = HumanMessage(content="u1")
    ai_tool = AIMessage(content="", tool_calls=[
        {"name": "web_search", "args": {"query": "q"}, "id": "call_1"}])
    tool_msg = ToolMessage(content='{"ok": 1}', tool_call_id="call_1")

    round1 = _compose_llm_messages({"messages": [u1]}, system, retrieval)
    round2 = _compose_llm_messages({"messages": [u1, ai_tool, tool_msg]},
                                   system, retrieval)
    # 第二轮前缀 = 第一轮完整序列
    assert _sig(round2)[:len(round1)] == _sig(round1)
    # 检索块紧跟 human 之后（而非 tool 消息之后）
    assert _sig(round2)[2] == ("system", "R")


def test_compose_cross_turn_prefix_stable():
    """跨轮：第 2 轮请求与第 1 轮的共同前缀覆盖 [system, u1]，
    分叉点恰在旧 human 之后——历史全部可命中，新增量仅为 a1/R2/u2。"""
    system = SystemMessage(content="S")
    u1 = HumanMessage(content="u1")
    a1 = AIMessage(content="a1")
    u2 = HumanMessage(content="u2")
    r1 = SystemMessage(content="检索结果一")
    r2 = SystemMessage(content="检索结果二")

    turn1 = _compose_llm_messages({"messages": [u1]}, system, r1)
    turn2 = _compose_llm_messages({"messages": [u1, a1, u2]}, system, r2)

    sig1, sig2 = _sig(turn1), _sig(turn2)
    # 共同前缀 = [system, u1]（turn2 的 a1 与 turn1 的 R1 在此处分叉）
    common = 0
    for s1, s2 in zip(sig1, sig2):
        if s1 != s2:
            break
        common += 1
    assert common == 2
    assert sig2 == _sig([system, u1, a1, u2, r2])


def test_compose_no_human_fallback_appends():
    """防御：没有 human 消息时检索块置尾（不会抛异常）。"""
    system = SystemMessage(content="S")
    retrieval = SystemMessage(content="R")
    msgs = [AIMessage(content="a")]
    out = _compose_llm_messages({"messages": msgs}, system, retrieval)
    assert _sig(out) == _sig([system, msgs[0], retrieval])


def test_compose_does_not_mutate_state_messages():
    """组装是纯函数：state['messages'] 不被修改（检索块不进 checkpoint）。"""
    system = SystemMessage(content="S")
    retrieval = SystemMessage(content="R")
    msgs = [HumanMessage(content="u1")]
    state = {"messages": msgs}
    _compose_llm_messages(state, system, retrieval)
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)

import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.core.tracing import Tracer
from app.graph.nodes import WorkflowContext
from app.graph.workflow import build_graph
from app.mcp.adapter import MCPToolAdapter
from app.mcp.host import MCPHost
from app.services.kb_service import KBService
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


class ScriptedModel:
    """按脚本依次返回预置消息的假模型（支持 bind_tools 假装绑定）。

    注意：responses 必须传共享列表（不要拷贝）——generate 节点会调用
    多次 get_chat_model，每次拿到的模型必须继续消费同一个响应队列。
    """

    def __init__(self, responses: list[AIMessage]):
        self._responses = responses       # 共享引用，不拷贝

    def bind_tools(self, schemas):
        return self          # 测试用：假装绑定了工具

    async def ainvoke(self, messages):
        return self._responses.pop(0)

    async def astream(self, messages):
        yield await self.ainvoke(messages)    # generate 节点用 astream 逐 token 流式


class ScriptedLLMService:
    def __init__(self, responses: list[AIMessage]):
        self._responses = responses       # 全测试共享这一个队列

    def get_chat_model(self, user_id: str, temperature=None):
        return ScriptedModel(self._responses)


def _make_ctx(responses: list[AIMessage]):
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    kb_service = KBService(settings)
    tracer = Tracer()
    host = MCPHost(settings.mcp_servers, base_dir=Path(__file__).resolve().parent.parent)
    adapter = MCPToolAdapter(host, tracer, kb_service=kb_service)
    ctx = WorkflowContext(settings, ScriptedLLMService(responses),
                          kb_service, adapter, tracer)
    return ctx, tracer


def test_tool_call_loop():
    """LLM 要调工具 → tool_executor 真实执行 MCP add → 结果回灌 → 最终回答。"""
    tool_call_msg = AIMessage(content="", tool_calls=[
        {"name": "add", "args": {"a": 1.5, "b": 2.7},
         "id": "call_1", "type": "tool_call"}])
    final_msg = AIMessage(content="结果是 4.2")
    ctx, tracer = _make_ctx([tool_call_msg, final_msg])
    graph = _run(build_graph(ctx))

    result = _run(graph.ainvoke(
        {"user_id": "u1", "session_id": "t-tool-1", "query": "1.5+2.7",
         "messages": [HumanMessage(content="1.5+2.7")]},
        config={"configurable": {"thread_id": "t-tool-1"}}))
    assert result["answer"] == "结果是 4.2"

    # 追踪：llm → tool(add) → llm
    kinds = [x["kind"] for x in tracer.list(session_id="t-tool-1")]
    assert kinds[0] == "llm" and kinds[1] == "tool"
    add_log = tracer.list(session_id="t-tool-1")[1]
    assert "4.2" in add_log["output"]


def test_no_tool_call_ends():
    """LLM 直接回答不调工具 → 图直接结束，无 tool 日志。"""
    ctx, tracer = _make_ctx([AIMessage(content="直接回答")])
    graph = _run(build_graph(ctx))

    result = _run(graph.ainvoke(
        {"user_id": "u1", "session_id": "t-tool-2", "query": "你好",
         "messages": [HumanMessage(content="你好")]},
        config={"configurable": {"thread_id": "t-tool-2"}}))
    assert result["answer"] == "直接回答"
    kinds = [x["kind"] for x in tracer.list(session_id="t-tool-2")]
    assert "tool" not in kinds


# ---------- 原生工具：list_kb_files（列出用户可检索库的文件名） ----------
def test_native_tool_in_catalog():
    """工具目录包含原生 list_kb_files（与外部 MCP 工具一起给 LLM 绑定）。"""
    ctx, _ = _make_ctx([])
    schemas = _run(ctx.mcp_adapter.schemas_for_llm())
    names = [s["function"]["name"] if "function" in s else s["name"]
             for s in schemas]
    assert "list_kb_files" in names
    assert "add" in names                              # 外部 MCP 工具仍在


def test_native_list_kb_files_per_user():
    """list_kb_files：按当前用户过滤（可检索库 + 该库文件名），并写追踪日志。"""
    import json as _json

    ctx, tracer = _make_ctx([])
    ks = ctx.kb_service
    kb1 = ks.create_kb("文件列表库A", "public", None,
                       ["量子比特可以处于叠加态"], description="A库")
    kb2 = ks.create_kb("文件列表库B", "public", None,
                       ["Shor算法可以分解大整数"], description="B库")
    ks.ingest_file(kb1.kb_id, "论文甲.txt", "神经区分器分析论文内容".encode())
    ks.set_retrieval(kb2.kb_id, "u1", enabled=False)      # u1 禁用 B 库

    out = _run(ctx.mcp_adapter.call("list_kb_files", {}, "t-native-1", "u1"))
    data = _json.loads(out["output"])
    assert data["user_id"] == "u1"
    by_name = {k["kb_name"]: k for k in data["kbs"]}
    assert "文件列表库B" not in by_name                   # u1 禁用的库不出现
    entry = by_name.get("文件列表库A")
    assert entry and "论文甲.txt" in entry["files"]        # 文件名正确列出

    # u2 未禁用 → B 库对 u2 可见（per-user 隔离）
    out2 = _run(ctx.mcp_adapter.call("list_kb_files", {}, "t-native-2", "u2"))
    data2 = _json.loads(out2["output"])
    assert "文件列表库B" in {k["kb_name"] for k in data2["kbs"]}

    # 追踪：原生工具执行同样落 ToolCallLog
    logs = tracer.list(session_id="t-native-1")
    assert logs and logs[0]["kind"] == "tool"
    assert logs[0]["name"] == "list_kb_files"


# ---------- 原生工具：get_local_document（取完整文章原文） ----------
def test_native_get_local_document():
    """多 chunk 文件 → 去掉 overlap 重复后精确还原原文。"""
    import json as _json

    ctx, tracer = _make_ctx([])
    ks = ctx.kb_service
    kb = ks.create_kb("全文库", "public", None, description="全文")
    # 5 段有序标记，总长 >1000 → 会被切成多个 chunk（相邻重叠 150 字符）
    body = "".join(f"[段{i}]" + "内容" * 500 for i in range(5))
    ks.ingest_file(kb.kb_id, "全文论文.txt", body.encode())

    out = _run(ctx.mcp_adapter.call(
        "get_local_document",
        {"kb_id": kb.kb_id, "file_name": "全文论文.txt"},
        "t-native-3", "u1"))
    data = _json.loads(out["output"])
    assert data["file_name"] == "全文论文.txt"
    assert data["chunk_count"] >= 2                    # 确实被分块过
    # 去重拼接后与原文完全一致（长度相等，不能是简单串接的更长版本）
    assert data["total_chars"] == len(body)
    assert data["full_text"] == body
    assert data["truncated"] is False
    # 追踪落 ToolCallLog
    logs = tracer.list(session_id="t-native-3")
    assert logs[0]["kind"] == "tool"
    assert logs[0]["name"] == "get_local_document"


def test_native_get_local_document_errors():
    """禁检索的库 / 不存在的文件 → 结构化错误（不抛异常）。"""
    import json as _json

    ctx, _ = _make_ctx([])
    ks = ctx.kb_service
    kb = ks.create_kb("权限库", "public", None, description="x")
    ks.ingest_file(kb.kb_id, "甲.txt", "短内容".encode())
    ks.set_retrieval(kb.kb_id, "u1", enabled=False)    # u1 禁检索

    # 用户禁检索 → 拒绝取原文
    out = _run(ctx.mcp_adapter.call(
        "get_local_document",
        {"kb_id": kb.kb_id, "file_name": "甲.txt"}, "t-native-4", "u1"))
    data = _json.loads(out["output"])
    assert "不可检索" in data["error"]

    # 文件不存在 → 错误 + 回传库内文件名列表（引导先 list_kb_files）
    out2 = _run(ctx.mcp_adapter.call(
        "get_local_document",
        {"kb_id": kb.kb_id, "file_name": "不存在.txt"}, "t-native-5", "u2"))
    data2 = _json.loads(out2["output"])
    assert "不存在文件" in data2["error"]
    assert data2["kb_files"] == ["甲.txt"]


# ---------- 停止生成：用户中断 → 保留部分答复，不再进工具循环 ----------
class StopAfterFirstTokenModel:
    """第一个 token 后设置停止标记的假模型：模拟用户在流式中途点停止。"""

    def __init__(self):
        from langchain_core.messages import AIMessageChunk
        self._chunks = [AIMessageChunk(content="生成到一半"),
                        AIMessageChunk(content="这段不该出现")]

    def bind_tools(self, schemas):
        return self

    async def astream(self, messages):
        from app.core.cancel import request_stop
        for i, c in enumerate(self._chunks):
            if i == 1:
                request_stop("t-stop-1")               # 首 token 后用户点停止
            yield c


def test_stop_generation_keeps_partial():
    """停止标记 → generate 保留部分答复（不含后续 token）、图正常收尾、标记被清。"""
    from app.core.cancel import clear_stop, is_stopped
    from langchain_core.messages import AIMessage

    class Svc:
        def get_chat_model(self, user_id, temperature=None):
            return StopAfterFirstTokenModel()

    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    ctx = WorkflowContext(settings, Svc(), KBService(settings), None, Tracer())
    graph = _run(build_graph(ctx))

    result = _run(graph.ainvoke(
        {"user_id": "u1", "session_id": "t-stop-1", "query": "随便问",
         "messages": [HumanMessage(content="随便问")]},
        config={"configurable": {"thread_id": "t-stop-1"}}))
    # 部分答复保留，停止点之后的 token 不进来
    assert result["answer"] == "生成到一半"
    assert result["stopped"] is True
    # 中断消息是干净的 AIMessage（无残留 tool_calls）
    assert isinstance(result["messages"][-1], AIMessage)
    assert not getattr(result["messages"][-1], "tool_calls", None)
    # 标记已被消费清除
    assert not is_stopped("t-stop-1")
    clear_stop("t-stop-1")


def test_chat_stop_endpoint(auth_factory):
    """POST /chat/stop 设置停止标记；下一轮对话开始时自动清除。

    P0-1 后需登录态；未登记的会话（首轮进行中）放行——属主校验只拦已登记
    且属于他人的会话。
    """
    from app.core.cancel import clear_stop, is_stopped
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/v1/chat/stop",
                   headers=auth_factory(),
                   json={"session_id": "t-stop-api"})
        assert r.status_code == 200
        assert r.json()["stopped"] is True
        assert is_stopped("t-stop-api")
        clear_stop("t-stop-api")

def test_cap_observation_truncates_long_output():
    """P3-32：工具观测超长截断——保头保尾 + 省略标记；短输出原样透传。"""
    from app.mcp.adapter import TOOL_OUTPUT_MAX_CHARS, cap_observation

    short = "正常输出"
    assert cap_observation(short) == short              # 短输出原样

    big = "HEAD" + "x" * (TOOL_OUTPUT_MAX_CHARS * 2) + "TAILMARK"
    capped = cap_observation(big)
    assert len(capped) <= TOOL_OUTPUT_MAX_CHARS + 80    # 含标记仍在限长附近
    assert capped.startswith("HEAD")                    # 保头
    assert capped.endswith("TAILMARK")                  # 保尾（结论/统计常在尾部）
    assert "已截断" in capped and str(len(big)) in capped  # 标记含原始长度


def test_stop_marker_cleared_after_endpoint(auth_factory):
    """端点设停 → 断言标记存在（收尾清理由下一轮 chat 开始时负责）。"""
    from app.core.cancel import clear_stop, is_stopped
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/v1/chat/stop",
                   headers=auth_factory(),
                   json={"session_id": "t-stop-api2"})
        assert r.status_code == 200
        assert is_stopped("t-stop-api2")
        clear_stop("t-stop-api2")
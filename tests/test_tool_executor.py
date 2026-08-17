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
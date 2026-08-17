"""MCP 适配层：统一调用 + 追踪 + 错误回灌，上层无感协议细节。

除转发外部 MCP Server 的工具外，还支持「原生工具」——直接跑在本进程里、
可访问应用内部服务（如知识库）的工具。对 LLM 而言两者无差别：都出现在
schemas_for_llm 的工具目录里，都经 call() 统一执行/追踪。
原生工具按 user_id 隔离（每次调用注入当前用户，天然私有）。
"""

import asyncio
import json

from langchain_core.utils.function_calling import convert_to_openai_function

from app.core.tracing import Tracer
from app.mcp.host import MCPHost


# ---------- 原生工具定义（本进程内执行，可访问 KBService） ----------
def _native_tool_specs() -> list[dict]:
    """原生工具目录：name/description/OpenAI parameters schema + 实现函数。

    实现签名 async def(args: dict, user_id: str) -> dict——user_id 由
    call() 注入（LLM 看不到也伪造不了），保证按用户隔离。
    """
    return [
        {
            "name": "list_kb_files",
            "description": (
                "列出当前用户可检索的每个知识库中的源文件名列表。"
                "用于回答「我的知识库里有哪些文件/资料」这类问题。"
                "无需参数，自动按当前用户过滤（只含本人可见且未被禁用检索的库）。"),
            "parameters": {"type": "object", "properties": {}, "required": []},
            "func": _list_kb_files,
        },
    ]


async def _list_kb_files(args: dict, user_id: str, kb_service) -> dict:
    """实现：可检索库 → 每库的源文件名列表（同步盘读放线程池）。

    带显式 file_count——LLM 数 80+ 项长列表会数错，读数字不会。
    """
    kbs = await asyncio.to_thread(kb_service.list_queryable_kbs, user_id)
    out = []
    for kb in kbs:
        docs = await asyncio.to_thread(kb_service.list_documents, kb.kb_id)
        files = [d.get("filename") or d.get("doc_id") for d in docs]
        out.append({
            "kb_id": kb.kb_id,
            "kb_name": kb.name,
            "file_count": len(files),
            "files": files,
        })
    return {"user_id": user_id, "kb_count": len(out), "kbs": out}


class MCPToolAdapter:
    """把 MCP 工具包装为统一接口：schema 生成、执行、追踪、错误处理一次搞定。"""

    def __init__(self, host: MCPHost, tracer: Tracer,
                 kb_service=None):
        self.host = host
        self.tracer = tracer
        self.kb_service = kb_service
        self._native = _native_tool_specs()

    async def ensure_catalog(self) -> None:
        """启动时/首次使用前发现工具目录。"""
        if self.host.tools is None:
            await self.host.discover()

    def _native_by_name(self, name: str):
        return next((t for t in self._native if t["name"] == name), None)

    async def schemas_for_llm(self) -> list[dict]:
        """工具目录（外部 MCP + 原生）转成 LLM 能理解的 OpenAI function schema。"""
        await self.ensure_catalog()
        schemas = [convert_to_openai_function(t) for t in self.host.tools]
        schemas += [{"type": "function",
                     "function": {"name": t["name"],
                                  "description": t["description"],
                                  "parameters": t["parameters"]}}
                    for t in self._native]
        return schemas

    async def call(self, name: str, args: dict, session_id: str,
                   user_id: str, parent_id: str | None = None) -> dict:
        """执行一个工具：全链路写 ToolCallLog，错误结构化返回（促 LLM 重试）。"""
        log_id = await asyncio.to_thread(
            self.tracer.start, "tool", name, session_id, user_id, args, parent_id)
        try:
            native = self._native_by_name(name)
            if native is not None:                     # 原生工具：本进程执行
                result = await native["func"](args, user_id, self.kb_service)
            else:                                      # 外部 MCP：tools/call
                tool = next(t for t in self.host.tools if t.name == name)
                result = await tool.ainvoke(args)
            output = json.dumps(result, ensure_ascii=False, default=str) \
                if not isinstance(result, str) else result
            await asyncio.to_thread(self.tracer.success, log_id, output)
            return {"output": output}
        except Exception as e:
            await asyncio.to_thread(self.tracer.error, log_id, str(e))
            return {"error": str(e), "name": name}     # 结构化错误回灌 LLM

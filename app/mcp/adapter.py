"""MCP 适配层 ：统一调用 + 追踪 + 错误回灌，上层无感协议细节。"""

import json

from langchain_core.utils.function_calling import convert_to_openai_function

from app.core.tracing import Tracer
from app.mcp.host import MCPHost


class MCPToolAdapter:
    """把 MCP 工具包装为统一接口：schema 生成、执行、追踪、错误处理一次搞定。"""

    def __init__(self, host: MCPHost, tracer: Tracer):
        self.host = host
        self.tracer = tracer

    async def ensure_catalog(self) -> None:
        """启动时/首次使用前发现工具目录。"""
        if self.host.tools is None:
            await self.host.discover()

    async def schemas_for_llm(self) -> list[dict]:
        """把工具目录转成 LLM 能理解的 OpenAI function schema（供 bind_tools）。"""
        await self.ensure_catalog()
        return [convert_to_openai_function(t) for t in self.host.tools]

    async def call(self, name: str, args: dict, session_id: str,
                   user_id: str, parent_id: str | None = None) -> dict:
        """执行一个工具：全链路写 ToolCallLog，错误结构化返回（促 LLM 重试）。"""
        log_id = self.tracer.start("tool", name, session_id, user_id, args, parent_id)
        try:
            tool = next(t for t in self.host.tools if t.name == name)
            result = await tool.ainvoke(args)          # 经 MCP tools/call
            output = json.dumps(result, ensure_ascii=False, default=str) \
                if not isinstance(result, str) else result
            self.tracer.success(log_id, output)
            return {"output": output}
        except Exception as e:
            self.tracer.error(log_id, str(e))
            return {"error": str(e), "name": name}     # 结构化错误回灌 LLM
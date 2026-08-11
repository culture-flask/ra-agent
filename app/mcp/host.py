"""MCP Host/Client（架构文档 §7.3）：连接多 Server，动态发现并聚合工具目录。"""

import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPHost:
    """连接配置里登记的所有 MCP Server，tools/list 动态发现。"""

    def __init__(self, servers_cfg: dict, base_dir: Path | None = None):
        self._base_dir = base_dir or Path.cwd()
        connections = {}
        for name, cfg in servers_cfg.items():
            conn = {"transport": cfg.get("transport", "stdio"), **cfg}
            if conn.get("command") == "python":
                conn["command"] = sys.executable      # 用当前解释器拉起子进程
            conn["args"] = [self._resolve(a) for a in conn.get("args", [])]
            connections[name] = conn
        self._client = MultiServerMCPClient(connections=connections)
        self._tools: list[BaseTool] | None = None

    def _resolve(self, arg: str) -> str:
        """把相对路径（servers/xxx.py）解析为绝对路径——不依赖启动目录。"""
        p = Path(arg)
        if p.suffix or "/" in arg or "\\" in arg:   # 有扩展名或路径分隔符才当路径处理
            return str(p if p.is_absolute() else self._base_dir / p)
        return arg    # 普通参数原样返回

    async def discover(self) -> list[BaseTool]:
        """动态发现：聚合所有 Server 的 tools/list，支持运行时刷新。"""
        self._tools = await self._client.get_tools()
        return self._tools

    @property
    def tools(self) -> list[BaseTool] | None:
        return self._tools
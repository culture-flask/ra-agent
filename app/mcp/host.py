"""MCP Host/Client（架构文档 §7.3）：连接多 Server，动态发现并聚合工具目录。"""

import asyncio
import logging
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("ra-agent.mcp")

DISCOVER_TIMEOUT = 10.0   # 秒：tools/list 动态发现超时。MCP 是可选能力，失败不阻塞服务启动


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
        """动态发现：聚合所有 Server 的 tools/list，支持运行时刷新。

        超时/失败不抛异常——MCP 连不上只降级为「空工具目录」，
        绝不能卡死应用启动（服务可用性优先于可选能力）。
        """
        try:
            self._tools = await asyncio.wait_for(
                self._client.get_tools(), timeout=DISCOVER_TIMEOUT)
        except Exception as e:                      # noqa: BLE001
            logger.warning("MCP 工具发现失败（%.0fs 内未完成）：%s —— 本次启动禁用 MCP 工具",
                           DISCOVER_TIMEOUT, e)
            self._tools = []
            # 尽力关闭 client，回收可能残留的子进程，不阻塞主流程
            try:
                await asyncio.wait_for(self._client.__aexit__(None, None, None), timeout=3)
            except Exception:                       # noqa: BLE001
                pass
        return self._tools

    @property
    def tools(self) -> list[BaseTool] | None:
        return self._tools
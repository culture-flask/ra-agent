# servers/research_server.py
"""示例 MCP Server：工具注册与自述。

FastMCP 装饰器声明工具 + 类型注解 → 自动生成 JSON Schema。
Agent 启动时 tools/list 即发现；新增工具 = 加一个 @mcp.tool() 函数，业务零改动。
"""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research")

SEARXNG_URL = "http://127.0.0.1:8190/search"

@mcp.tool()
def echo(message: str) -> str:
    """原样回显输入，用于连通性自检。"""
    return message


@mcp.tool()
def add(a: float, b: float) -> float:
    """计算两个数字之和。"""
    return a + b

@mcp.tool()
def web_search(query: str, top_k: int = 5) -> list[dict]:
    """联网搜索（SearXNG）：按关键词搜索互联网，返回标题/链接/摘要。

    适合查询实时信息、最新动态、本地知识库没有的内容。返回按相关度排序。
    """
    r = httpx.get(SEARXNG_URL, params={"q": query, "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return [
        {"title": item.get("title", ""), "url": item.get("url", ""),
         "content": (item.get("content") or "")[:200]}
        for item in data.get("results", [])[:top_k]
    ]

@mcp.tool()
def search_public_papers(query: str, top_k: int = 5) -> list[dict]:
    """在公共学术文献库中语义检索论文（骨架返回模拟结果）。"""
    return [
        {"title": f"《{query}》综述研究", "authors": ["Zhang, L."],
         "year": 2026, "abstract": f"围绕 {query} 的综述性研究。", "link": "https://doi.org/example"},
        {"title": f"{query} 的关键技术分析", "authors": ["Wang, X."],
         "year": 2025, "abstract": f"{query} 核心技术路线与对比。", "link": "https://doi.org/example2"},
    ][:top_k]


if __name__ == "__main__":
    mcp.run(transport="stdio")      # 本地：由 Agent 以子进程拉起```

# servers/research_server.py
"""示例 MCP Server：工具注册与自述。

FastMCP 装饰器声明工具 + 类型注解 → 自动生成 JSON Schema。
Agent 启动时 tools/list 即发现；新增工具 = 加一个 @mcp.tool() 函数，业务零改动。

学术检索三件套（全部免费、零密钥可用）：
- arxiv_search   预印本（物理/数学/计算机/密码学），最新研究
- openalex_search 跨学科全库（2.5 亿+ 篇，含被引数），最稳定
- s2_search      Semantic Scholar（含被引数；免费共享额度易 429，
                 可选 S2_API_KEY 提额，见 .env）
配套阅读工具：
- read_webpage   读取网页正文（检索到链接后深入阅读），零依赖 HTML 提取
"""
import os
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research")

SEARXNG_URL = "http://127.0.0.1:8190/search"
ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

_TIMEOUT = 20


def _load_dotenv() -> None:
    """把项目根 .env 里的可选密钥带进本子进程环境。

    MCP server 由主应用拉起，只继承 os.environ——主进程经
    pydantic-settings 读的 .env 文件内容并不会传过来，这里补读一次。
    setdefault：真实环境变量优先于文件。
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def _clamp_k(top_k: int, lo: int = 1, hi: int = 20) -> int:
    """top_k 夹取：LLM 偶尔会传 0/负数/超大值。"""
    return max(lo, min(int(top_k), hi))


# ---------- 网页正文提取（stdlib HTMLParser，零依赖） ----------

# 整体跳过的标签：脚本/样式/导航/页脚等噪音（跳过标签内的全部内容，而非仅标签本身）
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe",
              "nav", "header", "footer", "aside", "form",
              "button", "select", "option"}
# 块级标签：前后补换行，保住段落结构
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "td", "th", "hr",
               "h1", "h2", "h3", "h4", "h5", "h6",
               "section", "article", "blockquote", "pre", "table", "ul", "ol"}


class _TextExtractor(HTMLParser):
    """HTML → (标题, 正文文本)：跳过噪音标签，块级标签断行。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)   # 实体（&amp; 等）自动解码
        self._skip_depth = 0                      # 嵌套的噪音标签深度
        self._in_title = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        elif self._skip_depth == 0 and data.strip():
            self.parts.append(data)


def _extract_text(html: str) -> tuple[str, str]:
    """HTML 源码 → (标题, 正文)。纯函数，便于离线测试。

    纯文本（text/plain 页面）经 HTMLParser 原样通过，同样适用。
    """
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    title = " ".join("".join(parser.title_parts).split())
    lines = []
    for chunk in "".join(parser.parts).split("\n"):
        line = " ".join(chunk.split())       # 行内连续空白折叠为单空格
        if line:
            lines.append(line)
    return title, "\n".join(lines)


def _assert_public_http_url(url: str) -> None:
    """只允许公网 http/https URL：防 SSRF——网页内容若被注入恶意指令，
    可能诱导 LLM 读取本机服务（如本系统自己的 API）或内网地址。"""
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        raise ValueError(f"仅支持 http/https 链接，不支持：{u.scheme or '(空)'}")
    host = (u.hostname or "").lower()
    if not host:
        raise ValueError("URL 缺少主机名")
    private = (host == "localhost" or host.endswith(".localhost")
               or host in ("::1", "0.0.0.0")
               or host.startswith(("127.", "10.", "192.168.", "169.254.", "0."))
               or (host.startswith("172.")
                   and host.count(".") >= 1
                   and host.split(".")[1].isdigit()
                   and 16 <= int(host.split(".")[1]) <= 31))
    if private:
        raise ValueError("禁止访问内网/本机地址")


def _reconstruct_abstract(inverted: dict | None) -> str:
    """OpenAlex 的倒排摘要 {词: [位置列表]} → 正常顺序文本。"""
    if not inverted:
        return ""
    positions = []
    for word, idxs in inverted.items():
        for i in idxs or []:
            positions.append((i, word))
    return " ".join(word for _, word in sorted(positions))


def _parse_arxiv_atom(xml_text: str) -> list[dict]:
    """解析 arXiv Atom XML → 统一的论文条目（纯函数，便于离线测试）。"""
    atom = "{http://www.w3.org/2005/Atom}"
    root = ElementTree.fromstring(xml_text)
    out = []
    for e in root.findall(f"{atom}entry"):
        title = " ".join((e.findtext(f"{atom}title") or "").split())
        abstract = " ".join((e.findtext(f"{atom}summary") or "").split())
        authors = [(a.findtext(f"{atom}name") or "").strip()
                   for a in e.findall(f"{atom}author")]
        year = (e.findtext(f"{atom}published") or "")[:4]
        out.append({
            "title": title,
            "authors": [a for a in authors if a][:8],
            "year": int(year) if year.isdigit() else None,
            "abstract": abstract[:300],
            "url": (e.findtext(f"{atom}id") or "").strip(),
            "source": "arxiv",
        })
    return out


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
def arxiv_search(query: str, top_k: int = 5) -> list[dict]:
    """在 arXiv 预印本库检索论文（物理/数学/计算机/密码学/AI 等，免费无需密钥）。

    适合查找最新研究、算法与理论细节（很多领域论文先上 arXiv 再进期刊）。
    注意：arXiv 只收录英文文献——中文问题请先翻译成英文关键词再检索。
    返回标题/作者/年份/摘要/链接，按相关度排序。
    """
    q = query.strip()
    if not q:
        raise ValueError("query 不能为空")
    k = _clamp_k(top_k)
    r = httpx.get(ARXIV_API, params={
        "search_query": f"all:{q}",       # 空格按 AND 处理，多关键词直接传
        "start": 0, "max_results": k,
    }, timeout=_TIMEOUT)
    r.raise_for_status()
    return _parse_arxiv_atom(r.text)[:k]


@mcp.tool()
def openalex_search(query: str, top_k: int = 5) -> list[dict]:
    """跨学科学术文献检索（OpenAlex：2.5 亿+ 篇期刊/会议/预印本元数据，免费无需密钥，稳定不限流）。

    返回标题/作者/年份/摘要/被引次数/链接。被引次数可评估论文影响力。
    适合综述性查找、跨库查找、按引用量筛选重要工作；英文检索效果最佳。
    """
    q = query.strip()
    if not q:
        raise ValueError("query 不能为空")
    k = _clamp_k(top_k)
    params = {"search": q, "per_page": k}
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto    # 礼貌池：更稳定（可选配置）
    r = httpx.get(OPENALEX_API, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", [])[:k]:
        authors = [a["author"]["display_name"]
                   for a in (w.get("authorships") or [])
                   if isinstance(a, dict) and isinstance(a.get("author"), dict)]
        landing = ((w.get("primary_location") or {}).get("landing_page_url")
                   or w.get("doi") or w.get("id") or "")
        out.append({
            "title": w.get("display_name") or "",
            "authors": authors[:8],
            "year": w.get("publication_year"),
            "abstract": _reconstruct_abstract(w.get("abstract_inverted_index"))[:300],
            "cited_by_count": w.get("cited_by_count") or 0,
            "url": landing,
            "source": "openalex",
        })
    return out


@mcp.tool()
def s2_search(query: str, top_k: int = 5) -> list[dict]:
    """Semantic Scholar 学术检索（1 亿+ 篇，含被引次数，CS 领域覆盖尤佳）。

    免费共享额度有限，限流（429）时请改用 openalex_search。
    可在 .env 配置 S2_API_KEY 获得专属额度（可选）。
    返回标题/作者/年份/摘要/被引次数/链接。
    """
    q = query.strip()
    if not q:
        raise ValueError("query 不能为空")
    k = _clamp_k(top_k)
    headers = {}
    key = os.environ.get("S2_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    r = httpx.get(S2_API, params={
        "query": q, "limit": k,
        "fields": "title,authors,year,abstract,citationCount,url,externalIds",
    }, headers=headers, timeout=_TIMEOUT)
    if r.status_code == 429:
        # 结构化错误回灌 LLM（adapter 捕获转 {"error": ...}），引导它换工具
        raise RuntimeError("Semantic Scholar 免费额度限流（429），请改用 openalex_search")
    r.raise_for_status()
    out = []
    for p in r.json().get("data", [])[:k]:
        out.append({
            "title": p.get("title") or "",
            "authors": [a.get("name") or "" for a in (p.get("authors") or [])][:8],
            "year": p.get("year"),
            "abstract": (p.get("abstract") or "")[:300],
            "cited_by_count": p.get("citationCount") or 0,
            "url": p.get("url")
                or (p.get("externalIds") or {}).get("DOI") or "",
            "source": "semantic_scholar",
        })
    return out


@mcp.tool()
def read_webpage(url: str, max_chars: int = 8000) -> dict:
    """读取网页正文：给定 URL，抓取页面并提取标题与正文文本（自动剔除导航栏/脚本/样式/广告等噪音）。

    适合在 web_search 或学术检索找到链接后，深入阅读页面具体内容
    （论文摘要页、博客技术文章、文档页等）。
    返回 {title, url, content, chars, truncated}：content 为正文文本，
    超过 max_chars（默认 8000，最大 30000）时截断并标 truncated=true，
    需要更多内容时可带更大的 max_chars 重新调用。
    """
    _assert_public_http_url(url)
    limit = max(500, min(int(max_chars), 30000))
    r = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0 (compatible; ra-agent/0.1)"})
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if "html" not in ctype and "text/plain" not in ctype:
        raise ValueError(f"不支持的页面类型：{ctype or '未知'}（本工具只读 HTML/文本；"
                         f"PDF 请下载后通过知识库上传）")
    if len(r.content) > 5 * 1024 * 1024:
        raise ValueError("页面超过 5MB，过大无法读取")
    title, text = _extract_text(r.text)
    return {
        "title": title,
        "url": str(r.url),               # 跟随重定向后的最终地址
        "content": text[:limit],
        "chars": len(text),
        "truncated": len(text) > limit,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")      # 本地：由 Agent 以子进程拉起

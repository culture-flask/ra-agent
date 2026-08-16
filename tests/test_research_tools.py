"""MCP 学术检索工具的纯函数测试（离线，不依赖网络与 DB）。

网络调用本身不做单测（CI 无外网/接口限流），解析逻辑抽成纯函数在此覆盖；
上线后可用 manual_test.py 或直接对话做端到端冒烟。
"""

from servers.research_server import (
    _assert_public_http_url,
    _clamp_k,
    _extract_text,
    _parse_arxiv_atom,
    _reconstruct_abstract,
)


# ---------- top_k 夹取 ----------
def test_clamp_k():
    assert _clamp_k(5) == 5
    assert _clamp_k(0) == 1          # LLM 传 0 → 至少 1 条
    assert _clamp_k(-3) == 1
    assert _clamp_k(999) == 20       # 超大值封顶


# ---------- OpenAlex 倒排摘要还原 ----------
def test_reconstruct_abstract():
    inverted = {"differential": [1], "attack": [2], "on": [0], "Gimli": [3]}
    assert _reconstruct_abstract(inverted) == "on differential attack Gimli"
    # 位置列表乱序、一词多位置
    assert _reconstruct_abstract({"b": [0], "a": [1, 3], "c": [2]}) == "b a c a"


def test_reconstruct_abstract_empty():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


# ---------- arXiv Atom XML 解析 ----------
_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2004.12345v1</id>
    <updated>2020-04-27T00:00:00-04:00</updated>
    <published>2020-04-26T00:00:00-04:00</published>
    <title>Improving Attacks on
       Round-Reduced Gimli
       with  More  Spaces</title>
    <summary>We study the differential properties
       of Gimli.</summary>
    <author><name>Alice Wang</name></author>
    <author><name>Bob Zhang</name></author>
    <author><name>Carol Li</name></author>
  </entry>
</feed>
"""


def test_parse_arxiv_atom():
    entries = _parse_arxiv_atom(_ARXIV_XML)
    assert len(entries) == 1
    e = entries[0]
    # 标题/摘要里的换行与连续空格折叠为单空格
    assert e["title"] == "Improving Attacks on Round-Reduced Gimli with More Spaces"
    assert e["abstract"].startswith("We study the differential properties of Gimli.")
    assert e["authors"] == ["Alice Wang", "Bob Zhang", "Carol Li"]
    assert e["year"] == 2020                      # 取 published 前 4 位
    assert e["url"] == "http://arxiv.org/abs/2004.12345v1"
    assert e["source"] == "arxiv"


def test_parse_arxiv_atom_empty():
    # 无 entry 的空 feed → 空列表（不抛异常）
    empty = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
             '<title>ArXiv Query</title></feed>')
    assert _parse_arxiv_atom(empty) == []


# ---------- 网页正文提取 ----------
_HTML = """<html><head><title>  Gimli  差分
    分析  </title><style>body{color:red}</style>
<script>var x = 1; alert("noise")</script></head>
<body>
<nav><a href="/">首页</a> <a href="/about">关于</a></nav>
<header>网站横幅广告</header>
<h1>Gimli 置换的差分性质</h1>
<p>我们研究了 Gimli 的&差分;&amp;积分性质，
   并给出完整轮数的区分器。</p>
<div>实验结果表明该攻击需要 2<sup>38</sup> 次查询。</div>
<footer>© 2020 示例站点 | 联系我们</footer>
<script src="/analytics.js"></script>
</body></html>
"""


def test_extract_text_strips_noise():
    title, text = _extract_text(_HTML)
    # 标题：换行与连续空格折叠
    assert title == "Gimli 差分 分析"
    # 噪音整体剔除：script/style/nav/header/footer 内容不出现
    for noise in ("alert", "color:red", "首页", "关于", "横幅", "联系我们",
                  "analytics"):
        assert noise not in text
    # 正文保留：块级标签断行成独立行
    lines = text.split("\n")
    assert "Gimli 置换的差分性质" in lines
    assert any("完整轮数的区分器" in ln for ln in lines)
    # 内联标签（sup 等）的文本直接拼接不加空格——否则会拆散跨标签的单词
    assert any("238 次查询" in ln for ln in lines)


def test_extract_text_decodes_entities():
    _, text = _extract_text("<p>A &amp; B &lt;tag&gt; C&nbsp;D</p>")
    assert "A & B <tag> C D" in text          # 实体自动解码


def test_extract_text_plain_text_passthrough():
    # text/plain 页面：无标签，原样通过
    title, text = _extract_text("plain data\nsecond line")
    assert title == ""
    assert text == "plain data\nsecond line"


# ---------- URL 安全校验（防 SSRF） ----------
def test_url_allows_public():
    _assert_public_http_url("https://arxiv.org/abs/2004.12345")
    _assert_public_http_url("http://example.com/path?q=1")
    _assert_public_http_url("https://api.openalex.org/works?search=x")


def test_url_blocks_local_and_private():
    import pytest
    blocked = [
        "file:///etc/passwd",                  # 非 http 协议
        "ftp://example.com/x",
        "http://localhost:8000/api/v1/kbs",    # 本机（含本系统自身 API）
        "http://127.0.0.1:8000/health",
        "http://0.0.0.0/x",
        "http://[::1]/x",
        "http://192.168.1.10/admin",
        "http://10.0.0.5/internal",
        "http://172.16.0.1/db",                # 172.16~31 私网段
        "http://172.31.255.1/db",
        "http://169.254.169.254/latest/meta-data",   # 云元数据端点
        "http://sub.localhost/x",
        "not-a-url",
    ]
    for url in blocked:
        with pytest.raises(ValueError):
            _assert_public_http_url(url)


def test_url_allows_172_public_range():
    # 172.x 里 16~31 是私网，其余（如 172.217.x = Google）是公网，不能误杀
    _assert_public_http_url("http://172.217.161.78/")

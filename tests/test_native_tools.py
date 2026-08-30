"""头脑风暴新增原生工具离线测试：search_knowledge_base / save_document /
export_bibtex（网络类 MCP 新工具不在此测——离线确定性纪律，外部 API
失败模式由适配层结构化错误兜底，与既有降级纪律一致）。"""

import asyncio
import json
import tempfile
from pathlib import Path

from app.core.tracing import Tracer
from app.mcp.adapter import MCPToolAdapter
from app.mcp.host import MCPHost
from app.services.kb_service import KBService
from app.settings import Settings

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


def _make_adapter():
    """构造真实 KBService + 原生工具适配器（不拉起外部 MCP 子进程）。"""
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    kb_service = KBService(settings)
    host = MCPHost({}, base_dir=Path(__file__).resolve().parent.parent)
    adapter = MCPToolAdapter(host, Tracer(), kb_service=kb_service)
    return adapter, kb_service


def test_new_native_tools_in_catalog():
    """工具目录含全部新增原生工具（与外部 MCP 工具一起给 LLM 绑定）。"""
    adapter, _ = _make_adapter()
    schemas = _run(adapter.schemas_for_llm())
    names = {s.get("name") or (s.get("function") or {}).get("name")
             for s in schemas}
    assert {"search_knowledge_base", "save_document", "export_bibtex"} <= names


def test_search_knowledge_base_tool():
    """search_knowledge_base：LLM 主动按任意 query 检索可见库（含单库限定）。"""
    adapter, ks = _make_adapter()
    kb = ks.create_kb("头脑风暴检索库", "public", None,
                      ["量子比特可以处于叠加态"], description="量子")
    ks.ingest_file(kb.kb_id, "量子论文.txt", "量子纠缠与贝尔不等式实验".encode())

    # 任意角度 query 都能主动检索（不依赖议题原样）
    out = _run(adapter.call("search_knowledge_base", {"query": "叠加态"},
                            "t-bs-tool-1", "u1"))
    data = json.loads(out["output"])
    assert data["hit_count"] >= 1
    assert data["results"][0]["kb_name"] == "头脑风暴检索库"
    assert data["results"][0]["source"] == "text.txt"   # 建库初始文本的来源

    # 来源可溯：ingest_file 的命中回文件名（不是"未知来源"）
    out = _run(adapter.call("search_knowledge_base", {"query": "贝尔不等式"},
                            "t-bs-tool-1", "u1"))
    data = json.loads(out["output"])
    assert data["hit_count"] >= 1
    assert data["results"][0]["source"] == "量子论文.txt"

    # kb_name 限定：错误名 → 结构化错误
    out = _run(adapter.call("search_knowledge_base",
                            {"query": "叠加态", "kb_name": "不存在的库"},
                            "t-bs-tool-2", "u1"))
    data = json.loads(out["output"])
    assert "不存在或当前用户不可检索" in data["error"]

    # 空 query → 结构化错误
    out = _run(adapter.call("search_knowledge_base", {"query": "  "},
                            "t-bs-tool-3", "u1"))
    assert "query 不能为空" in json.loads(out["output"])["error"]


def test_search_knowledge_base_respects_privacy():
    """u2 搜不到 u1 的私库（权限校验在 KBService 层，工具天然继承）。"""
    from conftest import ensure_user
    ensure_user("u1")
    adapter, ks = _make_adapter()
    ks.create_kb("u1私密成果", "private", "u1", ["我的私藏实验数据pH=7.2"],
                 description="私有")

    out = _run(adapter.call("search_knowledge_base", {"query": "实验数据"},
                            "t-bs-tool-4", "u1"))
    data = json.loads(out["output"])
    assert data["hit_count"] >= 1                     # 属主能搜到

    out2 = _run(adapter.call("search_knowledge_base", {"query": "实验数据"},
                             "t-bs-tool-5", "u2"))
    data2 = json.loads(out2["output"])
    assert data2["hit_count"] == 0                    # 他人搜不到


def test_save_document_and_bibtex():
    """save_document：写入用户隔离目录 + 文件名白名单防穿越；
    export_bibtex：渲染 BibTeX + 同键消歧。"""
    from app.mcp.adapter import _render_bibtex

    adapter, _ = _make_adapter()

    # 正常保存（自动补 .md 后缀）
    out = _run(adapter.call("save_document",
                            {"filename": "research-proposal",
                             "content": "# 研究方案\n正文"},
                            "t-bs-tool-6", "u1"))
    data = json.loads(out["output"])
    assert data["saved"] is True and data["filename"] == "research-proposal.md"
    saved = Path(data["path"])
    assert saved.exists() and saved.read_text(encoding="utf-8") == "# 研究方案\n正文"
    assert saved.parent.name == "u1"                  # 按用户隔离目录

    # 路径穿越被拒
    out = _run(adapter.call("save_document",
                            {"filename": "../evil.md", "content": "x"},
                            "t-bs-tool-7", "u1"))
    assert "文件名只能包含" in json.loads(out["output"])["error"]

    # BibTeX 渲染：同键消歧 + 字段缺失容错
    bib = _render_bibtex([
        {"title": "Attention Is All You Need", "year": 2017,
         "authors": ["Vaswani"], "venue": "NeurIPS", "url": "https://a"},
        {"title": "Attention Is All You Need", "year": 2017},
    ])
    assert "@article{attention2017," in bib
    assert "attention2017-1" in bib                   # 同键加序号消歧
    assert "journal = {NeurIPS}" in bib

    # export_bibtex 端到端：写入 .bib 文件
    out = _run(adapter.call("export_bibtex",
                            {"filename": "refs",
                             "entries": [{"title": "Deep Residual Learning",
                                          "year": 2015, "authors": ["He"]}]},
                            "t-bs-tool-8", "u1"))
    data = json.loads(out["output"])
    assert data["saved"] is True and data["entry_count"] == 1
    assert Path(data["path"]).read_text(encoding="utf-8").startswith("@article{")

    # 空 entries → 结构化错误
    out = _run(adapter.call("export_bibtex", {"filename": "r.bib", "entries": []},
                            "t-bs-tool-9", "u1"))
    assert "entries 不能为空" in json.loads(out["output"])["error"]


def test_tracer_accepts_brainstorm_session_ids():
    """头脑风暴会话 id（"bs-"+uuid，39 字符）必须能正常写追踪日志。

    事故复盘：tool_call_log.session_id 曾是 VARCHAR(36)，前端生成的
    39 字符 id 让每条 INSERT 报 StringDataRightTruncation——适配层在
    写追踪时抛错，所有工具调用全军覆没且不留任何日志。
    """
    import uuid as _uuid

    from conftest import ensure_user
    ensure_user("u1")
    adapter, _ = _make_adapter()
    sid = "bs-" + str(_uuid.uuid4())            # 39 字符，与前端一致
    assert len(sid) == 39
    out = _run(adapter.call("get_utc_time", {}, sid, "u1"))
    assert "error" not in json.loads(out["output"]) or "error" not in out
    from app.core.db import SessionLocal
    from app.models import ToolCallLog
    with SessionLocal() as db:
        row = db.query(ToolCallLog).filter(
            ToolCallLog.session_id == sid,
            ToolCallLog.name == "get_utc_time").one_or_none()
        assert row is not None, "39 字符会话 id 的追踪日志未落库"
        assert row.error is None


def test_get_local_document_tolerates_non_string_args():
    """LLM 偶发把 kb_id/文件名传成数字 → 强制转 str，不再 AttributeError。"""
    adapter, ks = _make_adapter()
    kb = ks.create_kb("参数容错库", "public", None, description="x")
    ks.ingest_file(kb.kb_id, "甲.txt", "短内容".encode())
    out = _run(adapter.call("get_local_document",
                            {"kb_id": 123456, "file_name": 789},
                            "t-bs-tool-10", "u1"))
    data = json.loads(out["output"])
    assert "error" in data and "AttributeError" not in data["error"]


def test_add_documents_filenames_and_rolling_cap():
    """沉淀库改进：①入库可带可溯源文件名（不再是千篇一律的 text.txt）；
    ②滚动窗口按文件名（日期前缀字典序）保留最新 N 份、删除更旧。"""
    adapter, ks = _make_adapter()
    kb = ks.create_kb("滚动窗口库", "public", None, description="cap")
    for i in range(12):                        # 超过 ARCHIVE_KEEP_DOCS(10)
        fname = f"202608{i:02d}-争鸣社-test{i:02d}.md"
        ks.add_documents(kb.kb_id, [f"# 报告{i}"], filenames=[fname])
    docs = ks.list_documents(kb.kb_id)
    assert len(docs) == 12
    names = sorted(d["filename"] or "" for d in docs)
    assert names[0].startswith("20260800") and names[-1].startswith("20260811")

    from app.api.seminar import ARCHIVE_KEEP_DOCS, _cap_archive_docs
    removed = _cap_archive_docs(ks, kb.kb_id)
    assert removed == 12 - ARCHIVE_KEEP_DOCS
    docs = ks.list_documents(kb.kb_id)
    assert len(docs) == ARCHIVE_KEEP_DOCS
    names = sorted(d["filename"] or "" for d in docs)
    assert names[0].startswith(f"202608{12 - ARCHIVE_KEEP_DOCS:02d}")  # 最旧的已删

    # 溯源：检索命中能带出真实文件名（不是 text.txt）
    ks.add_documents(kb.kb_id, ["量子比特叠加态"], filenames=["溯源-报告.md"])
    out = _run(adapter.call("search_knowledge_base", {"query": "叠加态"},
                            "t-bs-tool-cap", "u1"))
    data = json.loads(out["output"])
    assert data["hit_count"] >= 1
    assert data["results"][0]["source"] == "溯源-报告.md"

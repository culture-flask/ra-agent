"""KB 入库流水线测试：解析→分块→落盘→向量化→ready 状态机、幂等、可见性。"""

import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.db import SessionLocal
from app.main import app
from app.models import User
from app.services.kb_service import KBService
from app.settings import Settings
from conftest import make_pdf_pages as _make_pdf_pages


def _ensure_user(user_id: str):
    """kbs.owner_user_id 有外键指向 users，私人库测试前先建用户。"""
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()


def _make_ctx():
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    return KBService(settings)


def test_create_kb_persisted_in_postgres():
    """元数据落 Postgres：创建后能从数据库读回，字段完整（含模型标注）。"""
    ks = _make_ctx()
    _ensure_user("u1")
    kb = ks.create_kb("测试库", "private", "u1", ["量子比特可以处于叠加态"])
    loaded = ks.get_kb(kb.kb_id)
    assert loaded.name == "测试库"
    assert loaded.owner_user_id == "u1"
    assert loaded.status == "ready"
    assert loaded.embedding_provider == "local"


def test_chunks_saved_to_disk():
    """chunk 落本地盘（对象存储简化）：文件按 kb_id 目录组织。"""
    ks = _make_ctx()
    kb = ks.create_kb("入库库", "public", None, ["a" * 1200])
    chunk_files = list((Path(ks._chunk_dir) / kb.kb_id).glob("*.txt"))
    assert len(chunk_files) == 2           # 1200 字，size=1000/overlap=150 → 2 块


def test_ingest_file_then_searchable():
    """docx 文件入库后立即可检索。"""
    import io
    from docx import Document
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("量子比特是量子计算的基本单元，可以处于叠加态。")
    doc.save(buf)

    ks = _make_ctx()
    kb = ks.create_kb("文档库", "public", None)
    ks.ingest_file(kb.kb_id, "note.docx", buf.getvalue())
    assert ks.get_kb(kb.kb_id).status == "ready"

    hits = ks.search(kb.kb_id, "叠加态", k=3, user_id="u1")
    assert len(hits) == 1
    assert "叠加态" in hits[0]["text"]


def test_ingest_idempotent():
    """同一文件重复入库 → chunk id 由内容哈希决定 → 数量不翻倍（幂等）。"""
    ks = _make_ctx()
    kb = ks.create_kb("幂等库", "public", None)
    n1 = ks.ingest_file(kb.kb_id, "a.txt", "量子比特是基本单元。".encode())
    n2 = ks.ingest_file(kb.kb_id, "a.txt", "量子比特是基本单元。".encode())
    assert n1 == n2
    vs = ks._vector_store(ks.get_kb(kb.kb_id))
    assert vs._col.count() == n1           # 覆盖写入，不重复


def test_private_kb_visibility():
    """私人库只对属主可见。"""
    ks = _make_ctx()
    _ensure_user("u1")
    ks.create_kb("u1私密", "private", "u1", ["我的实验pH=7.2"])
    ks.create_kb("公共库", "public", None, ["量子计算入门"])
    u1_kbs = [k.name for k in ks.list_kbs("u1")]
    u2_kbs = [k.name for k in ks.list_kbs("u2")]
    assert "u1私密" in u1_kbs
    assert "u1私密" not in u2_kbs
    assert "公共库" in u2_kbs


def test_upload_document_state_machine():
    """API 批量上传：一次带多个文件 → 立即返回 indexing → 轮询到 ready → 可检索。"""
    with TestClient(app) as c:
        kb = c.request("POST", "/api/v1/kbs",
                       json={"name": "上传库", "scope": "public", "user_id": "u1"}).json()

        r = c.post(f"/api/v1/kbs/{kb['kb_id']}/documents",
                   files=[("files", ("hello.txt", "量子比特可以处于叠加态。".encode(), "text/plain")),
                          ("files", ("second.md", "# 批量上传\n第二个文件的内容。".encode(), "text/markdown"))])
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "indexing"
        assert body["count"] == 2                 # 两个文件都在一批里

        for _ in range(30):                       # 轮询最多 15 秒
            status = c.get(f"/api/v1/kbs/{kb['kb_id']}").json()["status"]
            if status == "ready":
                break
            time.sleep(0.5)
        assert status == "ready"
        cur = c.get(f"/api/v1/kbs/{kb['kb_id']}").json()
        assert len(cur["source_doc_ids"]) == 2    # 两个文件都已入库

        hits = c.get(f"/api/v1/kbs/{kb['kb_id']}/search",
                     params={"query": "叠加态", "user_id": "u1"}).json()
        assert hits and "量子比特" in hits[0]["text"]   # 上传内容可检索到


def test_chunk_metadata_has_source_and_page():
    """chunk 元数据带源文件名与页码（PDF 逐页）；检索结果透出。"""
    ks = _make_ctx()
    kb = ks.create_kb("元数据库", "public", None)
    # 两页 PDF（页 1 空页、页 2 有内容 → 页码应标 2）
    ks.ingest_file(kb.kb_id, "paper.pdf", _make_pdf_pages(["", "Quantum Page Two"]))

    hits = ks.search(kb.kb_id, "Quantum", k=3, user_id="u1", mode="vector")
    assert hits
    meta = hits[0]["metadata"]
    assert meta["source"] == "paper.pdf"
    assert meta["page"] == 2

    # txt 无分页概念：页码为 None，文件名保留
    ks.ingest_file(kb.kb_id, "notes.txt", "深度学习需要显卡算力".encode())
    hits2 = ks.search(kb.kb_id, "深度学习", k=5, user_id="u1", mode="vector")
    txt_hit = next(h for h in hits2 if h["metadata"]["source"] == "notes.txt")
    assert txt_hit["metadata"].get("page") is None   # 无分页概念：不落 page 键


def test_read_chunks_restores_metadata():
    """从磁盘读 chunk（重建/BM25 路径）恢复 source/page（伴生 meta 文件）。"""
    ks = _make_ctx()
    kb = ks.create_kb("恢复库", "public", None)
    ks.ingest_file(kb.kb_id, "paper.pdf", _make_pdf_pages(["", "Page Two Text"]))
    kb2 = ks.get_kb(kb.kb_id)

    chunks = ks._read_chunks(kb.kb_id, kb2)
    assert chunks
    assert all(c.payload["source"] == "paper.pdf" for c in chunks)
    assert all(c.payload["page"] == 2 for c in chunks)

    # BM25 索引（同样从磁盘构建）也带 source/page
    idx = ks._bm25_index(kb2)
    bm_hits = idx.search("Page Two", k=3)
    assert bm_hits and bm_hits[0]["metadata"]["source"] == "paper.pdf"


def test_get_parent_block_aggregates_group():
    """父块聚合：同 doc 内同组 chunk 拼接、页码收集、不存在的组返回 None。"""
    from app.services.kb_service import _chunk_index

    assert _chunk_index("abc123_def456_7") == 7
    assert _chunk_index("abc123_def456_0") == 0

    ks = _make_ctx()
    kb = ks.create_kb("父块库", "public", None)
    # PDF helper 仅支持 ASCII：用英文长文本，每页足够切成多个 chunk
    page1 = "Quantum computing fundamentals with superposition states. " * 60
    page2 = "Protein folding research in bioinformatics. " * 60
    ks.ingest_file(kb.kb_id, "paper.pdf", _make_pdf_pages([page1, page2]))
    kb2 = ks.get_kb(kb.kb_id)

    chunks = ks._doc_chunks(kb.kb_id, kb2)
    assert len(chunks) >= 3

    block = ks.get_parent_block(kb.kb_id, chunks[0].payload["doc_id"], 0,
                                group_size=3, max_chars=4000)
    assert block is not None
    assert block["text"]
    assert block["source"] == "paper.pdf"
    assert block["chunk_ids"]
    assert "Quantum computing" in block["text"]     # 组 0 是第 1 页内容
    assert block["pages"] == [1]                    # 页码收集

    assert ks.get_parent_block(kb.kb_id, "nope", 99) is None   # 不存在的组


def test_search_mode_vector_vs_hybrid():
    """同一查询两种检索模式都返回结果：vector 只有向量距离，hybrid 带 BM25 分数。"""
    ks = _make_ctx()
    kb = ks.create_kb("模式库", "public", None,
                      ["量子比特是量子计算的基本单元，可以处于叠加态"])
    vec = ks.search(kb.kb_id, "量子比特", k=3, user_id="u1", mode="vector")
    hyb = ks.search(kb.kb_id, "量子比特", k=3, user_id="u1", mode="hybrid")

    assert vec and hyb
    assert all("distance" in h for h in vec)
    assert all(h.get("bm25_score") is None for h in vec)     # 纯向量无 BM25 字段
    assert all(h["method"] == "vector" for h in vec)

    assert all("score" in h for h in hyb)                    # RRF 融合分
    assert any(h.get("bm25_score") is not None for h in hyb) # 有 BM25 命中
    assert all(h["method"] in ("vector", "bm25", "hybrid") for h in hyb)
    assert all(h["kb_name"] == "模式库" for h in vec + hyb)  # 归属字段一致


def test_search_mode_default_from_settings():
    """mode 缺省时取全局配置 retrieval_mode（yaml 默认 hybrid）。"""
    ks = _make_ctx()
    kb = ks.create_kb("默认模式库", "public", None, ["深度学习需要显卡算力"])
    hits = ks.search(kb.kb_id, "深度学习", k=3, user_id="u1")
    assert hits and hits[0].get("score") is not None        # hybrid 生效


def test_upload_unsupported_file_goes_failed():
    """不支持的格式 → 后台任务失败 → 状态机走到 failed（而不是永久 indexing）。"""
    with TestClient(app) as c:
        kb = c.request("POST", "/api/v1/kbs",
                       json={"name": "坏文件库", "scope": "public", "user_id": "u1"}).json()
        r = c.post(f"/api/v1/kbs/{kb['kb_id']}/documents",
                   files={"files": ("data.xlsx", b"binary", "application/octet-stream")})
        assert r.status_code == 200          # 上传立即成功
        for _ in range(30):
            status = c.get(f"/api/v1/kbs/{kb['kb_id']}").json()["status"]
            if status in ("ready", "failed"):
                break
            time.sleep(0.5)
        assert status == "failed"            # 状态机正确落位


def test_clean_text_drops_lone_surrogates():
    """孤立代理字符（pypdf 提取残缺数学符号产生）→ 清洗掉，不再编码崩溃。"""
    from app.services.kb_service import _clean_text
    assert _clean_text("abc\ud835def") == "abcdef"
    # 完整代理对（合法数学字母 U+1D435）保留
    assert _clean_text("x\U0001d435y") == "x\U0001d435y"
    assert _clean_text("普通中文 text") == "普通中文 text"


def test_batch_skips_bad_file_and_ingests_rest():
    """批量上传：一个文件解析失败只跳过它，其余文件照常入库，状态仍 ready。"""
    ks = _make_ctx()
    kb = ks.create_kb("批量容错库", "public", None)
    n = ks.ingest_files(kb.kb_id, [
        ("ok.txt", "量子比特可以处于叠加态。".encode()),
        ("bad.xlsx", b"binary"),                       # 不支持的格式 → 跳过
    ])
    assert n > 0
    cur = ks.get_kb(kb.kb_id)
    assert cur.status == "ready"
    assert len(cur.source_doc_ids or []) == 1          # 只有 ok.txt 入库

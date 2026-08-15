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

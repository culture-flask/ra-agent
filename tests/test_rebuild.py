"""Day 8 测试：每库选模型、知识库重建、双库并存、维度错配防护。"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.abstractions.embedding import EmbeddingModel, EmbeddingMeta
from app.main import app
from app.services.kb_service import KBService
from app.settings import Settings


def _make_ks():
    settings = Settings.load().model_copy(update={
        "chroma_persist_dir": Path(tempfile.mkdtemp()),
        "data_dir": Path(tempfile.mkdtemp()),
        "embedding_default_provider": "local",
    })
    return KBService(settings)


def test_create_kb_with_custom_provider():
    """建库时显式指定模型 → 标注进 KB 元数据（每库标注模型，§11.2）。"""
    ks = _make_ks()
    kb = ks.create_kb("定制库", "public", None,
                      ["量子比特可以处于叠加态"],
                      provider="local", model_id="mini-b")
    assert kb.embedding_provider == "local"
    assert kb.embedding_model_id == "mini-b"
    assert kb.status == "ready"


def test_unknown_provider_rejected():
    """未知 provider → 明确报错（fail fast，而不是悄悄用默认）。"""
    ks = _make_ks()
    with pytest.raises(ValueError, match="unknown embedding provider"):
        ks.create_kb("坏库", "public", None, provider="nonexistent")


def test_rebuild_keeps_old_kb_and_new_searchable():
    """重建：旧 KB 保留可检索 + 新 KB（新 collection）可检索，互不影响。"""
    ks = _make_ks()
    old = ks.create_kb("原库", "public", None,
                       ["量子比特是量子计算的基本单元，可以处于叠加态。",
                        "Shor算法可以分解大整数。"])
    new = ks.rebuild(old.kb_id, provider="local", model_id="mini-b")

    assert new.kb_id != old.kb_id                  # 新 KB 独立 id
    assert new.name.endswith("(重建)")
    assert new.embedding_model_id == "mini-b"      # 新模型标注
    assert new.status == "ready"

    # 旧库：仍可检索（未被覆盖）
    hits_old = ks.search(old.kb_id, "叠加态", k=3, user_id="u1")
    assert any("叠加态" in h["text"] for h in hits_old)
    # 新库：同样可检索（独立 collection）
    hits_new = ks.search(new.kb_id, "叠加态", k=3, user_id="u1")
    assert any("叠加态" in h["text"] for h in hits_new)
    assert hits_old[0]["kb_id"] != hits_new[0]["kb_id"]


def test_dual_model_kbs_coexist():
    """双模型并存：两个库用不同模型标注，各自检索独立。"""
    ks = _make_ks()
    kb_a = ks.create_kb("库A", "public", None, ["量子计算综述"],
                        provider="local", model_id="mini")
    kb_b = ks.create_kb("库B", "public", None, ["深度学习综述"],
                        provider="local", model_id="mini-b")
    hits_a = ks.search(kb_a.kb_id, "量子", k=3, user_id="u1")
    hits_b = ks.search(kb_b.kb_id, "量子", k=3, user_id="u1")
    assert len(hits_a) == 1 and len(hits_b) == 1


def test_dimension_mismatch_rejected():
    """维度错配防护：往已定维度的 collection 写不同维度向量 → 明确报错。"""
    ks = _make_ks()

    class Fake500Dim(EmbeddingModel):
        def embed_texts(self, texts):
            return [[0.0] * 500 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 500

    kb = ks.create_kb("384库", "public", None, ["正常向量"], provider="local")
    vs = ks._vector_store(kb)
    bad = Fake500Dim(EmbeddingMeta(provider="fake", model_id="x", dim=500))
    from app.abstractions.vectorstore import ChunkRecord
    with pytest.raises(Exception):                 # Chroma 拒绝维度不一致的写入
        bad_vs = __import__("app.abstractions.vectorstore",
                            fromlist=["ChromaVectorStore"]).ChromaVectorStore(
            str(ks._settings.chroma_persist_dir), kb.kb_id, bad)
        bad_vs.add([ChunkRecord(id="x1", text="错维", payload={"scope": "public"})])


def test_rebuild_api():
    """API：重建端点返回新 KB（模型标注更新）。"""
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "API重建", "scope": "public", "user_id": "u1",
            "texts": ["量子比特是基本单元"],
            "embedding_provider": "local"}).json()
        r = c.post(f"/api/v1/kbs/{kb['kb_id']}/rebuild",
                   json={"embedding_provider": "local",
                         "embedding_model_id": "mini-b"})
        assert r.status_code == 200
        new = r.json()
        assert new["kb_id"] != kb["kb_id"]
        assert new["embedding_model_id"] == "mini-b"
        assert new["status"] == "ready"

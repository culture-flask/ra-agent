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
    # 自定义 provider 也必须同时给全 base_url / model_id / dim，缺一即拒绝
    with pytest.raises(ValueError, match="unknown embedding provider"):
        ks.create_kb("坏库2", "public", None, provider="llama", model_id="x", dim=768)
    with pytest.raises(ValueError, match="unknown embedding provider"):
        ks.create_kb("坏库3", "public", None, provider="llama",
                     dim=768, base_url="http://h:1/v1")      # 缺 model_id → 拒绝


def test_custom_openai_compatible_provider():
    """任意 OpenAI 兼容端点（llama.cpp / vLLM）：显式 base_url + model + dim 即可接入。"""
    ks = _make_ks()
    kb = ks.create_kb("llama库", "public", None,
                      provider="llama", model_id="nomic-embed-text-v1.5",
                      dim=768, base_url="http://100.85.4.71:9999/v1")
    assert kb.embedding_provider == "llama"
    assert kb.embedding_model_id == "nomic-embed-text-v1.5"
    assert kb.embedding_dim == 768
    assert kb.embedding_base_url == "http://100.85.4.71:9999/v1"
    assert kb.status == "ready"


def test_self_hosted_provider_needs_no_api_key():
    """自建端点免鉴权：未配置 provider 无 key 也能构建（不再抛 missing api_key）。"""
    from app.abstractions.embedding import EmbeddingFactory
    meta = EmbeddingMeta("llama", "nomic-embed-text-v1.5", 768,
                         "http://100.85.4.71:9999/v1")
    model = EmbeddingFactory.build(meta, {})
    assert model.meta.provider == "llama"
    # 配置过的云端 provider 无 key 仍要 fail fast（secrets 字典含该 provider 键）
    with pytest.raises(RuntimeError, match="missing api_key"):
        EmbeddingFactory.build(EmbeddingMeta("doubao", "m", 2048),
                               {"doubao": None, "system": None})


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


def test_update_embedding_config_and_mismatch():
    """嵌入配置创建后可改：改模型后 embedding_mismatch 提醒，只换端点不提醒。"""
    ks = _make_ks()
    kb = ks.create_kb("改配置库", "public", None)
    assert ks.embedding_mismatch(ks.get_kb(kb.kb_id)) is None     # 空库无提醒

    ks.ingest_file(kb.kb_id, "a.txt", "量子比特可以处于叠加态。".encode())
    kb = ks.get_kb(kb.kb_id)
    assert kb.embedded_model is not None                          # 记录了写入模型
    assert ks.embedding_mismatch(kb) is None                      # 配置未变 → 无提醒

    # 只换端点（同模型）→ 不提醒
    kb1 = ks.update_embedding(kb.kb_id, base_url="http://127.0.0.1:1/v1")
    assert ks.embedding_mismatch(kb1) is None

    # 换模型（同 provider 不同 model_id）→ 提醒
    kb2 = ks.update_embedding(kb.kb_id, provider="local", model_id="mini-b", dim=384)
    assert kb2.embedding_model_id == "mini-b"
    warn = ks.embedding_mismatch(kb2)
    assert warn is not None and "mini" in warn and "重建" in warn

    # 再入库一次 → embedded_model 更新为当前配置 → 提醒消失
    ks.ingest_file(kb.kb_id, "b.txt", "Shor算法可以分解大整数。".encode())
    kb4 = ks.get_kb(kb.kb_id)
    assert kb4.embedded_model["model_id"] == "mini-b"
    assert ks.embedding_mismatch(kb4) is None


def test_api_update_embedding():
    """API：PATCH /kbs/{id}/embedding 修改配置；非法 provider → 400。"""
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "API改库", "scope": "public", "user_id": "u1",
            "embedding_provider": "local"}).json()
        # 换到自定义端点（llama.cpp 风格）→ 200
        r = c.patch(f"/api/v1/kbs/{kb['kb_id']}/embedding", json={
            "embedding_provider": "llama.cpp", "embedding_model_id": "qwen3-embedding:8b",
            "embedding_dim": 4096, "embedding_base_url": "http://127.0.0.1:18080/v1"})
        assert r.status_code == 200
        d = r.json()
        assert d["embedding_provider"] == "llama.cpp"
        assert d["embedding_dim"] == 4096
        # 非法 provider 且无可用端点（base_url 被清空）→ 400 且配置不变
        r2 = c.patch(f"/api/v1/kbs/{kb['kb_id']}/embedding",
                     json={"embedding_provider": "nonexistent",
                           "embedding_base_url": ""})
        assert r2.status_code == 400
        d2 = c.get(f"/api/v1/kbs/{kb['kb_id']}").json()
        assert d2["embedding_provider"] == "llama.cpp"

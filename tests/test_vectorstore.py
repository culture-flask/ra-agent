# 两级隔离（离线）
import tempfile

from app.abstractions.embedding import EmbeddingMeta, LocalEmbeddingModel
from app.abstractions.vectorstore import ChromaVectorStore, ChunkRecord, kb_filter


def _vs():
    tmp = tempfile.mkdtemp()
    emb = LocalEmbeddingModel(EmbeddingMeta(provider="local", model_id="mini", dim=384))
    vs = ChromaVectorStore(tmp, "kb1", emb)
    vs.add([
        ChunkRecord(id="p1", text="量子比特可以处于叠加态", payload={"scope": "public", "user_id": ""}),
        ChunkRecord(id="r1", text="张三的实验记录：pH=7.2", payload={"scope": "private", "user_id": "u1"}),
        ChunkRecord(id="r2", text="李四的私人笔记：温度25度", payload={"scope": "private", "user_id": "u2"}),
    ])
    return vs


def test_kb_filter_shapes():
    assert kb_filter("public", None) == {"scope": "public"}
    assert kb_filter("private", "u1") == {"$and": [{"scope": "private"}, {"user_id": "u1"}]}
    assert "$or" in kb_filter("all", "u1")


def test_user_can_see_own_private_only():
    vs = _vs()
    hits = vs.search("张三的实验", k=5, scope="private", user_id="u1")
    assert [h["id"] for h in hits] == ["r1"]


def test_cross_user_isolation():
    vs = _vs()
    hits = vs.search("张三的实验", k=5, scope="private", user_id="u2")
    assert "r1" not in [h["id"] for h in hits]


def test_public_visible_to_all():
    vs = _vs()
    hits = vs.search("量子", k=5, scope="public", user_id=None)
    assert all(h["metadata"]["scope"] == "public" for h in hits)


def test_merge_search_public_plus_own_private():
    vs = _vs()
    hits = vs.search("记录", k=5, scope="all", user_id="u1")
    ids = [h["id"] for h in hits]
    assert "r1" in ids
    assert "r2" not in ids
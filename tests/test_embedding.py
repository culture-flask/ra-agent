# 本地嵌入（离线，不依赖外部 API）
from app.abstractions.embedding import EmbeddingMeta, EmbeddingFactory, LocalEmbeddingModel


def test_local_embedding_dim():
    meta = EmbeddingMeta(provider="local", model_id="mini", dim=384)
    m = LocalEmbeddingModel(meta)
    v = m.embed_texts(["你好", "世界"])
    assert len(v) == 2
    assert len(v[0]) == 384


def test_embedding_factory_missing_key():
    meta = EmbeddingMeta(provider="doubao", model_id="m", dim=2048, base_url="http://x")
    # "已配置"= secrets 字典里有该 provider 键（KBService._embedding_secrets 按
    # settings.embedding_cloud 生成）；配置过的云端 provider 无 key 必须 fail fast
    try:
        EmbeddingFactory.build(meta, {"doubao": None, "system": None})
    except RuntimeError as e:
        assert "api_key" in str(e)
    else:
        raise AssertionError("缺少密钥必须报错")


def test_local_dim_matches_meta():
    from app.settings import Settings
    meta = EmbeddingMeta(provider="local", model_id="mini",
                         dim=Settings.load().embedding_local_default_dim)
    m = LocalEmbeddingModel(meta)
    assert m.dim == len(m.embed_texts(["x"])[0])
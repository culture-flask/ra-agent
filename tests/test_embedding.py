# 本地嵌入（离线，不依赖外部 API）
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from app.abstractions.embedding import EmbeddingMeta, EmbeddingFactory, LocalEmbeddingModel
from app.abstractions.embedding import CloudEmbeddingModel


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


# ---------- 云端嵌入：瞬时故障重试 / 批量 400 单条隔离 ----------
def _cloud_model() -> CloudEmbeddingModel:
    meta = EmbeddingMeta(provider="llama.cpp", model_id="e", dim=4,
                         base_url="http://x/v1")
    return CloudEmbeddingModel(meta, "sk-x")


def _resp(vecs: list[list[float]]):
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vecs])


def _bad_request():
    req = httpx.Request("POST", "http://x/v1/embeddings")
    return BadRequestError("Error code: 400",
                           response=httpx.Response(400, request=req),
                           body={"error": "bad"})


def test_json_corruption_retried(monkeypatch):
    """响应流损坏（"Extra data: ..." JSON 解析错）是瞬时故障 → 退避重试后成功。"""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    m = _cloud_model()
    calls = {"n": 0}

    def create(model, input):
        calls["n"] += 1
        if calls["n"] == 1:      # 第一次响应损坏（llama.cpp 高负载下偶发截断/串包）
            raise json.JSONDecodeError("Extra data", "line 1", 55030)
        return _resp([[0.1] * 4 for _ in input])

    m._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    out = m.embed_texts(["a", "b"])
    assert len(out) == 2 and len(out[0]) == 4
    assert calls["n"] == 2       # 重试吸收，不再打挂整个文件


def test_batch_400_falls_back_to_single(monkeypatch):
    """整批 400（端点负载抖动）→ 拆单条重试：单条更轻，大概率成功。"""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    m = _cloud_model()

    def create(model, input):
        if len(input) > 1:
            raise _bad_request()          # 批量请求被拒
        return _resp([[0.2] * 4])         # 单条成功

    m._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    out = m.embed_texts(["a", "b"])
    assert len(out) == 2 and len(out[0]) == 4


def test_single_item_400_raises():
    """单条也 400 = 输入确实被端点拒绝（非瞬时故障）→ 抛出，由上层标记该文件失败。"""
    m = _cloud_model()

    def create(model, input):
        raise _bad_request()

    m._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(BadRequestError):
        m.embed_texts(["bad input"])
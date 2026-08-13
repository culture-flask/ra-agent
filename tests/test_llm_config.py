"""Day 9 API 测试：provider 目录、配置 CRUD（掩码）、一键模型列表（mock）、隔离。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _register(c, username):
    reg = c.post("/api/v1/auth/register",
                 json={"username": username, "password": "pass1234"}).json()
    return c.get("/api/v1/auth/me",
                 headers={"Authorization": f"Bearer {reg['access_token']}"}).json()["id"]


def test_providers_catalog():
    with TestClient(app) as c:
        p = c.get("/api/v1/llm/providers").json()
        assert "sensenova" in p and "openai" in p


def test_config_save_and_masked_list():
    with TestClient(app) as c:
        uid = _register(c, "cfgtest1")
        r = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "sensenova",
            "base_url": "https://token.sensenova.cn/v1",
            "model_id": "deepseek-v4-flash",
            "api_key": "sk-abcdefgh123456", "is_default": True})
        assert r.status_code == 200
        assert r.json()["api_key_masked"] == "sk-a...3456"

        cfgs = c.get("/api/v1/llm/configs", params={"user_id": uid}).json()
        assert len(cfgs) == 1
        assert "abcdefgh" not in str(cfgs)            # 永不明文


def test_configs_user_isolated():
    with TestClient(app) as c:
        uid1 = _register(c, "cfgtest2")
        c.post("/api/v1/llm/configs", json={
            "user_id": uid1, "provider": "sensenova",
            "base_url": "https://x/v1", "model_id": "m1", "api_key": "k1"})
        cfgs = c.get("/api/v1/llm/configs", params={"user_id": "u2"}).json()
        assert cfgs == []                             # 他人配置不可见


def test_custom_provider_mocked():
    """目录外 provider 按 custom 处理：mock httpx，验证不再 400。"""

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "local-model"}]}

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        return FakeResp()

    import app.api.llm_config as m
    from app.main import app as a
    from fastapi.testclient import TestClient
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(m.httpx, "get", fake_get)
    with TestClient(a) as c:
        r = c.post("/api/v1/llm/models", json={
            "provider": "ollama", "base_url": "http://x:11434/v1", "api_key": "k"})
        assert r.status_code == 200
        assert r.json() == ["local-model"]
    monkeypatch.undo()


def test_list_models_mocked(monkeypatch):
    """一键模型列表：mock httpx，验证解析逻辑（不依赖网络）。"""

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "m1"}, {"id": "m2"}]}

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        assert "/models" in url
        return FakeResp()

    import app.api.llm_config as m
    monkeypatch.setattr(m.httpx, "get", fake_get)
    with TestClient(app) as c:
        r = c.post("/api/v1/llm/models", json={
            "provider": "sensenova", "base_url": "https://token.sensenova.cn/v1",
            "api_key": "k"})
        assert r.status_code == 200
        assert r.json() == ["m1", "m2"]

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


def test_set_default_switches_between_configs():
    """已保存的配置之间可随时切换默认（互斥，且不影响他人）。"""
    with TestClient(app) as c:
        uid = _register(c, "cfgswitch1")
        r1 = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "sensenova",
            "base_url": "https://a/v1", "model_id": "m-a",
            "api_key": "sk-aaaaaaaa1111", "is_default": True}).json()
        r2 = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "openai",
            "base_url": "https://b/v1", "model_id": "m-b",
            "api_key": "sk-bbbbbbbb2222", "is_default": False}).json()

        def defaults():
            return [x["id"] for x in c.get("/api/v1/llm/configs",
                                           params={"user_id": uid}).json()
                    if x["is_default"]]

        assert defaults() == [r1["id"]]

        # 把 m-b 切换为默认 → 互斥：m-a 不再是默认
        assert c.patch(f"/api/v1/llm/configs/{r2['id']}/default",
                       params={"user_id": uid}).status_code == 200
        assert defaults() == [r2["id"]]

        # 再切回 m-a
        assert c.patch(f"/api/v1/llm/configs/{r1['id']}/default",
                       params={"user_id": uid}).status_code == 200
        assert defaults() == [r1["id"]]


def test_set_default_rejects_other_users_config():
    """不能把别人的配置设为默认（越权防护）。"""
    with TestClient(app) as c:
        uid = _register(c, "cfgswitch2")
        r = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "sensenova",
            "base_url": "https://a/v1", "model_id": "m-a",
            "api_key": "sk-aaaaaaaa1111"}).json()
        resp = c.patch(f"/api/v1/llm/configs/{r['id']}/default",
                       params={"user_id": "someone-else"})
        assert resp.status_code == 404
        # 原用户的默认状态不受影响（本就没设默认，仍无默认）
        cfgs = c.get("/api/v1/llm/configs", params={"user_id": uid}).json()
        assert not any(x["is_default"] for x in cfgs)


def test_update_config_switches_model():
    """PATCH 只改 model_id：同一条配置内切换模型，密钥掩码与默认状态不变。"""
    with TestClient(app) as c:
        uid = _register(c, "cfgpatch1")
        r = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "sensenova",
            "base_url": "https://a/v1", "model_id": "m-a",
            "api_key": "sk-aaaaaaaa1111", "is_default": True}).json()

        resp = c.patch(f"/api/v1/llm/configs/{r['id']}",
                       params={"user_id": uid},
                       json={"model_id": "m-b"})
        assert resp.status_code == 200

        cfgs = c.get("/api/v1/llm/configs", params={"user_id": uid}).json()
        assert len(cfgs) == 1                       # 仍是同一条配置
        assert cfgs[0]["model_id"] == "m-b"
        assert cfgs[0]["api_key_masked"] == "sk-a...1111"   # 密钥没被改
        assert cfgs[0]["is_default"] is True        # 切换后仍是默认


def test_update_config_rejects_other_user():
    """他人 user_id 不能改我的配置。"""
    with TestClient(app) as c:
        uid = _register(c, "cfgpatch2")
        r = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "sensenova",
            "base_url": "https://a/v1", "model_id": "m-a",
            "api_key": "sk-aaaaaaaa1111"}).json()
        resp = c.patch(f"/api/v1/llm/configs/{r['id']}",
                       params={"user_id": "someone-else"},
                       json={"model_id": "m-b"})
        assert resp.status_code == 404


def test_config_models_uses_stored_key(monkeypatch):
    """按配置拉模型列表：后端解密该配置的 api_key 调 /models（key 不出服务端）。"""
    seen = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-v4"}]}

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        seen["auth"] = headers.get("Authorization")
        assert "/models" in url
        return FakeResp()

    import app.api.llm_config as m
    monkeypatch.setattr(m.httpx, "get", fake_get)
    with TestClient(app) as c:
        uid = _register(c, "cfgpatch3")
        r = c.post("/api/v1/llm/configs", json={
            "user_id": uid, "provider": "sensenova",
            "base_url": "https://a/v1", "model_id": "m-a",
            "api_key": "sk-abcdefgh123456"}).json()
        resp = c.get(f"/api/v1/llm/configs/{r['id']}/models",
                     params={"user_id": uid})
        assert resp.status_code == 200
        assert resp.json() == ["deepseek-v4-flash", "deepseek-v4"]
        assert seen.get("auth") == "Bearer sk-abcdefgh123456"   # 明文 key 只在服务端内部用


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

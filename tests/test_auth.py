from fastapi.testclient import TestClient

from app.main import app


def test_register_and_me():
    with TestClient(app) as client:
        # 注册
        r = client.post("/api/v1/auth/register",
                        json={"username": "alice", "password": "pass1234"})
        assert r.status_code == 201
        token = r.json()["access_token"]

        # /me 带 token
        r = client.get("/api/v1/auth/me",
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["username"] == "alice"


def test_duplicate_register():
    with TestClient(app) as client:
        client.post("/api/v1/auth/register",
                    json={"username": "bob", "password": "bob12345"})
        r = client.post("/api/v1/auth/register",
                        json={"username": "bob", "password": "bob12345"})
        assert r.status_code == 409


def test_bad_token():
    with TestClient(app) as client:
        r = client.get("/api/v1/auth/me",
                       headers={"Authorization": "Bearer fake.token.here"})
        assert r.status_code == 401


def test_wrong_password():
    with TestClient(app) as client:
        client.post("/api/v1/auth/register",
                    json={"username": "carol", "password": "carol123"})
        r = client.post("/api/v1/auth/login",
                        json={"username": "carol", "password": "wrong"})
        assert r.status_code == 401
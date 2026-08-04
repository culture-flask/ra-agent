from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:        # 起一个进程内 HTTP 客户端
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
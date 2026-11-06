from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.settings import Settings


@pytest.mark.parametrize("directory_type", [str, Path], ids=["yaml-string", "env-path"])
def test_health(monkeypatch, tmp_path, directory_type):
    # YAML 赋值可能保留字符串，环境变量经 Pydantic 验证后则是 Path。
    # 两种配置都须走完整 lifespan，避免测试 DATA_DIR 掩盖实际启动错误。
    loaded = Settings.load().model_copy(update={"data_dir": directory_type(tmp_path / "data")})
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: loaded))
    with TestClient(app) as client:        # 起一个进程内 HTTP 客户端
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert app.state.multimodal_service.root_dir == tmp_path / "data" / "chat_media"

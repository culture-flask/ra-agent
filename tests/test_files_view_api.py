"""「笔记和文件」视图 API 测试：产出目录全量列表 + 知识库源文档下载。

产出目录隔离（只能看自己的）、下载路径穿越防护、
源文档下载的可见性闸门（私有库仅属主，与 delete_kb 同语义）。"""

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.mcp.adapter import _user_dir_name


def test_outputs_all_lists_own_files_only(auth_factory):
    """产出列表：最近修改优先、只见本人目录；下载按名取件防穿越。"""
    h1, h2 = auth_factory("u1"), auth_factory("u2")
    with TestClient(app) as c:
        c.app.state.settings.output_dir = Path(tempfile.mkdtemp())
        d1 = c.app.state.settings.output_dir / _user_dir_name("u1")
        d1.mkdir(parents=True)
        (d1 / "b-调研报告.md").write_text("# hi", encoding="utf-8")
        (d1 / "a-refs.bib").write_text("@x{}", encoding="utf-8")
        os.utime(d1 / "a-refs.bib", (1000, 1000))          # 旧时间 → 排序在后
        d2 = c.app.state.settings.output_dir / _user_dir_name("u2")
        d2.mkdir(parents=True)
        (d2 / "secret.md").write_text("s", encoding="utf-8")

        r = c.get("/api/v1/outputs/all", headers=h1)
        assert r.status_code == 200
        assert [f["name"] for f in r.json()] == ["b-调研报告.md", "a-refs.bib"]
        assert r.json()[0]["size"] == 4

        # 下载：命中本人文件（中文文件名原样）
        r = c.get("/api/v1/outputs/download", headers=h1, params={"name": "b-调研报告.md"})
        assert r.status_code == 200 and r.content == "# hi".encode()
        # 穿越被白名单拦截
        r = c.get("/api/v1/outputs/download", headers=h1,
                  params={"name": "../" + _user_dir_name("u2") + "/secret.md"})
        assert r.status_code == 400
        # u2 下载 u1 的文件名：本人目录无此文件 → 404（天然按目录隔离）
        r = c.get("/api/v1/outputs/download", headers=h2, params={"name": "b-调研报告.md"})
        assert r.status_code == 404
        # u2 的列表只含自己的
        r = c.get("/api/v1/outputs/all", headers=h2)
        assert [f["name"] for f in r.json()] == ["secret.md"]


def test_outputs_all_empty_dir(auth_factory):
    auth_factory("u1")
    with TestClient(app) as c:
        c.app.state.settings.output_dir = Path(tempfile.mkdtemp())
        r = c.get("/api/v1/outputs/all", headers=auth_factory("u1"))
        assert r.status_code == 200 and r.json() == []


def test_kb_doc_download_visibility(auth_factory):
    """私有库源文档：属主可下、他人 403（列表与下载同闸）；doc_id 白名单。"""
    h1, h2 = auth_factory("u1"), auth_factory("u2")
    with TestClient(app) as c:
        settings = c.app.state.settings
        kb = c.app.state.kb_service.create_kb("源文档库", "private", "u1",
                                              ["量子比特可以处于叠加态"])
        docs = Path(settings.data_dir) / "docs" / kb.kb_id
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "3cf7839a__笔记与材料.md").write_text("# 来源", encoding="utf-8")

        # 属主：列表 + 下载还原原始文件名（剥离 doc_id__ 前缀）
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/files", headers=h1)
        assert r.status_code == 200
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/docs/3cf7839a/download", headers=h1)
        assert r.status_code == 200 and "# 来源".encode() in r.content
        assert "attachment" in r.headers.get("content-disposition", "")

        # 非属主：列表与下载都 403
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/files", headers=h2)
        assert r.status_code == 403
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/docs/3cf7839a/download", headers=h2)
        assert r.status_code == 403

        # 非法 doc_id（非十六进制）→ 400；合法但不存在 → 404
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/docs/ABC123/download", headers=h1)
        assert r.status_code == 400
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/docs/deadbeef/download", headers=h1)
        assert r.status_code == 404


def test_kb_doc_download_public_kb_any_user(auth_factory):
    """公共库源文档：登录用户都可下载。"""
    auth_factory("u1")
    h2 = auth_factory("u2")
    with TestClient(app) as c:
        settings = c.app.state.settings
        kb = c.app.state.kb_service.create_kb("公共源文档库", "public", None,
                                              ["混合检索 RRF 融合排序"])
        docs = Path(settings.data_dir) / "docs" / kb.kb_id
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "abcdef12__source.txt").write_text("rrf", encoding="utf-8")
        r = c.get(f"/api/v1/kbs/{kb.kb_id}/docs/abcdef12/download", headers=h2)
        assert r.status_code == 200 and r.content == b"rrf"

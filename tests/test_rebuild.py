"""Day 8 测试：每库选模型、知识库重建、双库并存、维度错配防护。"""

import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.abstractions.embedding import EmbeddingModel, EmbeddingMeta
from app.core.db import SessionLocal
from app.main import app
from app.models import KnowledgeBase, User
from app.services.kb_service import KBService
from app.settings import Settings


def _ensure_user(user_id: str):
    """kbs.owner_user_id 有外键指向 users，私人库（含重建库）测试前先建用户。"""
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()


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
    # 白名单外的自定义 provider 必须同时给全 base_url / model_id / dim，缺一即拒绝
    # （不用 "llama"：本地 config.cloud 可能已将其加入白名单，测不了"未知"分支）
    with pytest.raises(ValueError, match="unknown embedding provider"):
        ks.create_kb("坏库2", "public", None, provider="ghost_endpoint",
                     model_id="x", dim=768)                 # 缺 base_url → 拒绝
    with pytest.raises(ValueError, match="unknown embedding provider"):
        ks.create_kb("坏库3", "public", None, provider="ghost_endpoint",
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
    _ensure_user("u1")
    old = ks.create_kb("原库", "public", None,
                       ["量子比特是量子计算的基本单元，可以处于叠加态。",
                        "Shor算法可以分解大整数。"])
    new = ks.rebuild(old.kb_id, provider="local", model_id="mini-b", user_id="u1")

    assert new.kb_id != old.kb_id                  # 新 KB 独立 id
    assert new.name.endswith("(重建)")
    assert new.embedding_model_id == "mini-b"      # 新模型标注
    assert new.status == "ready"
    assert new.scope == "private"                  # 重建库强制私人（归属发起者）
    assert new.owner_user_id == "u1"

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
    """API：重建改为异步——POST 立即返回 new_kb_id，轮询进度到 ready。"""
    _ensure_user("u1")
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "API重建", "scope": "public", "user_id": "u1",
            "texts": ["量子比特是基本单元"],
            "description": "量子计算资料",
            "embedding_provider": "local"}).json()
        r = c.post(f"/api/v1/kbs/{kb['kb_id']}/rebuild",
                   json={"embedding_provider": "local",
                         "embedding_model_id": "mini-b", "user_id": "u1"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "reembedding" and body["new_kb_id"]
        new_id = body["new_kb_id"]
        # 轮询进度端点 → ready
        for _ in range(40):
            p = c.get(f"/api/v1/kbs/{new_id}/rebuild-progress").json()
            if p and p["status"] in ("ready", "failed"):
                break
            time.sleep(0.2)
        assert p and p["status"] == "ready" and p["pct"] == 100
        new = c.get(f"/api/v1/kbs/{new_id}").json()
        assert new["kb_id"] == new_id and new["kb_id"] != kb["kb_id"]
        assert new["embedding_model_id"] == "mini-b"
        assert new["scope"] == "private"           # 重建库默认私人
        assert new["description"] == "量子计算资料"  # 介绍继承原库


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

    # 再入库一次 -> embedded_model 更新为当前配置 -> 提醒消失
    ks.ingest_file(kb.kb_id, "b.txt", "Shor算法可以分解大整数。".encode())
    kb4 = ks.get_kb(kb.kb_id)
    assert kb4.embedded_model["model_id"] == "mini-b"
    assert ks.embedding_mismatch(kb4) is None


def test_ingest_failure_after_config_change_keeps_old_files():
    """回归：改嵌入配置后入库失败，历史文件绝不能被牵连删除。

    旧 bug：_ingest 失败路径 rmtree 整个 chunk 目录--改配置再上传，
    向量化失败（维度不兼容/端点不可达）-> 库内之前所有文件被清空。
    修复后：配置不一致只提醒不拦截；失败只清本批残留，旧文件保留。
    """
    ks = _make_ks()
    kb = ks.create_kb("失败保护库", "public", None)
    ks.ingest_file(kb.kb_id, "a.txt", "量子比特可以处于叠加态。".encode())
    assert [d["filename"] for d in ks.list_documents(kb.kb_id)] == ["a.txt"]

    # 换成不可达的自定义端点：与已入库向量模型不一致 -> 有提醒，但不拦截入库
    ks.update_embedding(kb.kb_id, provider="llama.cpp", model_id="qwen3-embedding",
                        dim=4096, base_url="http://127.0.0.1:1/v1")
    assert ks.embedding_mismatch(ks.get_kb(kb.kb_id)) is not None

    n = ks.ingest_file(kb.kb_id, "b.txt", "Shor算法可以分解大整数。".encode())
    assert n == 0                                       # 向量化失败（连接拒绝）
    kb2 = ks.get_kb(kb.kb_id)
    assert kb2.status == "failed"
    assert kb2.embedded_model["provider"] == "local"    # 已入库向量仍标注旧模型
    # 关键：历史文件完好（旧 bug 这里是空列表）
    assert [d["filename"] for d in ks.list_documents(kb.kb_id)] == ["a.txt"]


def test_api_update_embedding():
    """API：PATCH /kbs/{id}/embedding 修改配置；非法 provider → 400。"""
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "API改库", "scope": "public", "user_id": "u1",
            "description": "改配置测试库",
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


def test_set_retrieval_per_user_isolated():
    """检索开关 per-user：用户 u1 禁用不影响 u2；可随时恢复。

    旧实现 retrieval_enabled 是库级全局字段——u1 禁用后所有人（u2）都被禁。
    """
    ks = _make_ks()
    kb_a = ks.create_kb("可检索库", "public", None)
    kb_b = ks.create_kb("被禁库", "public", None)

    assert {k.kb_id for k in ks.list_queryable_kbs("u1")} == {kb_a.kb_id, kb_b.kb_id}

    ks.set_retrieval(kb_b.kb_id, "u1", enabled=False)     # u1 禁用
    assert kb_b.kb_id not in {k.kb_id for k in ks.list_queryable_kbs("u1")}
    assert kb_b.kb_id in {k.kb_id for k in ks.list_queryable_kbs("u2")}   # u2 不受影响
    visible = {k.kb_id for k in ks.list_kbs("u1")}         # 列表仍可见（可管理）
    assert kb_b.kb_id in visible

    ks.set_retrieval(kb_b.kb_id, "u1", enabled=True)       # 恢复
    assert kb_b.kb_id in {k.kb_id for k in ks.list_queryable_kbs("u1")}
    assert kb_b.kb_id in {k.kb_id for k in ks.list_queryable_kbs("u2")}


def test_set_retrieval_recovers_legacy_global_disable():
    """回归：per-user 改造前的旧全局禁用（retrieval_enabled=False）不再死锁。

    旧 bug：开启只清个人禁用列表，库级总开关仍卡 False → 永远开不回来。
    修复：开启时一并把库级开关拉回 True。
    """
    ks = _make_ks()
    kb = ks.create_kb("旧禁用库", "public", None)
    with SessionLocal() as db:               # 模拟旧版写入的全局禁用
        row = db.get(KnowledgeBase, kb.kb_id)
        row.retrieval_enabled = False
        db.commit()
    assert not ks.list_queryable_kbs("u1")   # 旧状态：全员禁用

    ks.set_retrieval(kb.kb_id, "u1", enabled=True)          # 用户点「允许检索」
    assert kb.kb_id in {k.kb_id for k in ks.list_queryable_kbs("u1")}
    assert kb.kb_id in {k.kb_id for k in ks.list_queryable_kbs("u2")}


def test_api_retrieval_toggle_per_user():
    """API：PATCH /kbs/{id}/retrieval?user_id= per-user 开关；缺 enabled → 422。"""
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "API开关库", "scope": "public", "user_id": "u1",
            "description": "检索开关测试库"}).json()
        assert kb["retrieval_enabled_for_user"] is True

        # u1 禁用：只影响 u1，u2 视角仍是可用
        r = c.patch(f"/api/v1/kbs/{kb['kb_id']}/retrieval?user_id=u1",
                    json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["retrieval_enabled_for_user"] is False
        assert c.get(f"/api/v1/kbs/{kb['kb_id']}?user_id=u2"
                     ).json()["retrieval_enabled_for_user"] is True

        r2 = c.patch(f"/api/v1/kbs/{kb['kb_id']}/retrieval?user_id=u1",
                     json={"enabled": True})
        assert r2.json()["retrieval_enabled_for_user"] is True

        r3 = c.patch(f"/api/v1/kbs/{kb['kb_id']}/retrieval?user_id=u1", json={})
        assert r3.status_code == 422


def test_copy_kb_full_clone_without_embedding():
    """完全复制：向量原样搬运（不调嵌入 API），数据独立、可直接检索、强制私人。"""
    ks = _make_ks()
    _ensure_user("u1")
    src = ks.create_kb("源库", "public", None,
                       ["量子比特可以处于叠加态。", "Shor算法可以分解大整数。"],
                       provider="local", description="复制源")
    # 完全复制：嵌入配置继承，不重新向量化
    new = ks.copy_kb(src.kb_id, "u1")
    assert new.kb_id != src.kb_id
    assert new.name == "源库(复制)"
    assert new.scope == "private" and new.owner_user_id == "u1"   # 强制私人
    assert new.status == "ready"
    assert new.embedding_model_id == src.embedding_model_id
    assert new.embedded_model == src.embedded_model          # 向量写入标注继承
    assert new.source_doc_ids == src.source_doc_ids
    assert new.description == "复制源"
    # 复制库独立可检索（BM25 + 向量都就位）
    hits = ks.search(new.kb_id, "量子比特", k=2, user_id="u1")
    assert hits and any("量子" in h["text"] for h in hits)
    # 文件管理与源库一致（chunk 目录完整复制）
    assert [d["filename"] for d in ks.list_documents(new.kb_id)] == \
           [d["filename"] for d in ks.list_documents(src.kb_id)]
    # 磁盘数据独立：删源库不影响复制库
    ks.delete_kb(src.kb_id)
    assert ks.search(new.kb_id, "量子比特", k=2, user_id="u1")


def test_copy_kb_via_api():
    """API：POST /kbs/{id}/rebuild mode=copy → 异步复制，轮询进度到 ready。"""
    _ensure_user("u1")
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "API复制源", "scope": "public", "user_id": "u1",
            "texts": ["量子纠错需要冗余量子比特"],
            "description": "API 复制测试", "embedding_provider": "local"}).json()
        r = c.post(f"/api/v1/kbs/{kb['kb_id']}/rebuild",
                   json={"mode": "copy", "user_id": "u1"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "copying" and body["new_kb_id"]
        new_id = body["new_kb_id"]
        for _ in range(40):
            p = c.get(f"/api/v1/kbs/{new_id}/rebuild-progress").json()
            if p and p["status"] in ("ready", "failed"):
                break
            time.sleep(0.2)
        assert p and p["status"] == "ready" and p["pct"] == 100
        d = c.get(f"/api/v1/kbs/{new_id}").json()
        assert d["scope"] == "private"
        assert d["kb_id"] != kb["kb_id"]
        assert d["embedding_model_id"] == kb["embedding_model_id"]


def test_update_kb_name_and_description():
    """名称/介绍创建后随时可改：PATCH /kbs/{id}；空名 400；不存在的库 404。"""
    with TestClient(app) as c:
        kb = c.post("/api/v1/kbs", json={
            "name": "旧名字", "scope": "public", "user_id": "u1",
            "description": "旧介绍", "embedding_provider": "local"}).json()
        # 只改名字
        r = c.patch(f"/api/v1/kbs/{kb['kb_id']}",
                    json={"name": "新名字"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "新名字" and d["description"] == "旧介绍"
        # 名字 + 介绍一起改
        r2 = c.patch(f"/api/v1/kbs/{kb['kb_id']}",
                     json={"name": "更好名字", "description": "新介绍"})
        assert r2.json()["description"] == "新介绍"
        # 空名 → 400；空介绍允许（仅清空）
        assert c.patch(f"/api/v1/kbs/{kb['kb_id']}",
                       json={"name": "  "}).status_code == 400
        assert c.patch(f"/api/v1/kbs/{kb['kb_id']}",
                       json={"description": ""}).json()["description"] == ""
        assert c.patch("/api/v1/kbs/nonexistent",
                       json={"name": "x"}).status_code == 404


# ---------- 取消：入库 / 重建可被用户随时终止 ----------
def test_ingest_cancel():
    """request_cancel 后入库 → 状态 cancelled（非 failed），已成功文件保留。"""
    from app.core.cancel import acknowledge_cancel, request_cancel
    ks = _make_ks()
    kb = ks.create_kb("取消入库库", "public", None)
    acknowledge_cancel(kb.kb_id)                     # 清残留
    request_cancel(kb.kb_id)                         # 模拟用户点终止入库
    n = ks.ingest_files(kb.kb_id, [
        ("a.txt", "量子比特内容".encode()),
        ("b.txt", "Shor算法内容".encode())])
    assert n == 0                                    # 未成功入库任何文件
    prog = ks.ingest_progress(kb.kb_id)
    assert prog["status"] == "cancelled"             # 取消不是失败
    assert all(f["status"] in ("pending", "cancelled") for f in prog["files"])
    # 库状态回到 ready（未被标记 failed）
    assert ks.get_kb(kb.kb_id).status == "ready"
    assert not any("a.txt" in (d.get("filename") or "") for d in ks.list_documents(kb.kb_id))


def test_ingest_cancel_keeps_finished_files():
    """入库中途取消：已成功文件保留并入库，其余取消。"""
    import app.services.kb_service as kbmod
    from app.core.cancel import acknowledge_cancel, request_cancel
    ks = _make_ks()
    kb = ks.create_kb("取消保留库", "public", None)
    # 把模块级 _check_cancel 包一层：首次（第一个文件边界）通过，
    # 第二次（第二个文件边界）先 set 取消标记再检查 → 命中
    orig = kbmod._check_cancel
    calls = {"n": 0}

    def flaky(key):
        calls["n"] += 1
        if calls["n"] >= 2:
            request_cancel(kb.kb_id)
        return orig(key)

    kbmod._check_cancel = flaky
    try:
        ks.ingest_files(kb.kb_id, [
            ("ok.txt", "第一个文件".encode()),
            ("later.txt", "第二个文件".encode())])
    finally:
        kbmod._check_cancel = orig
        acknowledge_cancel(kb.kb_id)
    prog = ks.ingest_progress(kb.kb_id)
    assert prog["status"] == "cancelled"
    by_name = {f["filename"]: f["status"] for f in prog["files"]}
    assert by_name["ok.txt"] == "ok"                 # 已完成的保留
    assert by_name["later.txt"] == "cancelled"       # 未完成的取消
    assert [d["filename"] for d in ks.list_documents(kb.kb_id)] == ["ok.txt"]


def test_rebuild_cancel_deletes_partial_kb():
    """重建中途取消 → 抛 OperationCancelled，且半成品库被删除。"""
    import app.services.kb_service as kbmod
    from app.core.cancel import OperationCancelled, acknowledge_cancel, request_cancel
    ks = _make_ks()
    _ensure_user("u1")
    src = ks.create_kb("取消重建源", "public", None,
                       ["量子比特是基本单元，可以处于叠加态。"] * 20,
                       provider="local")
    acknowledge_cancel("partial-new-id")
    # 包一层 _rebuild_progress_cb：首个批次回调时 set 取消标记再检查 → 中断向量化
    orig_cb = kbmod.KBService._rebuild_progress_cb
    orig_check = kbmod._check_cancel
    fired = {"n": 0}

    def cancel_first_cb(self, new_kb_id, prog, chunks, done):
        fired["n"] += 1
        if fired["n"] >= 1:
            request_cancel(new_kb_id)
        return orig_cb(self, new_kb_id, prog, chunks, done)

    kbmod.KBService._rebuild_progress_cb = cancel_first_cb
    try:
        try:
            ks.rebuild(src.kb_id, provider="local", user_id="u1",
                       new_kb_id="partial-new-id")
            raise AssertionError("should have cancelled")
        except OperationCancelled:
            pass                                   # 期望取消
    finally:
        kbmod.KBService._rebuild_progress_cb = orig_cb
        kbmod._check_cancel = orig_check
        acknowledge_cancel("partial-new-id")
    # 半成品库已被删除
    with pytest.raises(KeyError):
        ks.get_kb("partial-new-id")

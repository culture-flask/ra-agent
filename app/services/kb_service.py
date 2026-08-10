"""知识库服务：

- KB 元数据 → Postgres
- 文档解析 → 多格式（parsing.py）
- chunk → 本地盘"对象存储"（data/chunks/{kb_id}/，便于重建复用免解析）
- 状态机：indexing → ready（文件上传走后台异步入库）
"""

import hashlib
import uuid
from pathlib import Path

from sqlalchemy import select

from app.abstractions.embedding import EmbeddingFactory, EmbeddingMeta
from app.abstractions.vectorstore import ChunkRecord, VectorStoreFactory
from app.core.db import SessionLocal
from app.core.events import emit
from app.core.logging import get_logger
from app.models import KnowledgeBase
from app.services.parsing import parse_file

logger = get_logger("kb_service")


def split_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """分块：固定窗口 + 重叠（正式解析流程的第 2 步）。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


class KBService:
    """两级知识库管理：元数据在 Postgres，chunk 在本地盘，向量在 Chroma。"""

    def __init__(self, settings):
        self._settings = settings
        self._chunk_dir = Path(settings.data_dir) / "chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_secrets = {
            **{p: settings.embedding_api_key for p in settings.embedding_cloud},
            "system": settings.embedding_api_key,
        }

    # ---------- 模型解析（与 Day 4 相同） ----------
    def resolve_embedding_meta(self) -> EmbeddingMeta:
        provider = self._settings.embedding_default_provider
        if provider == "local":
            return EmbeddingMeta(provider="local", model_id="mini",
                                 dim=self._settings.embedding_local_default_dim)
        cloud = self._settings.embedding_cloud[provider]
        return EmbeddingMeta(provider=provider, model_id=cloud["model_id"],
                             dim=cloud["dim"], base_url=cloud["base_url"])

    def _vector_store(self, kb: KnowledgeBase):
        meta = EmbeddingMeta(
            kb.embedding_provider, kb.embedding_model_id, kb.embedding_dim,
            self._settings.embedding_cloud.get(kb.embedding_provider, {}).get("base_url"))
        emb = EmbeddingFactory.build(meta, self._embedding_secrets)
        return VectorStoreFactory.build(self._settings, kb.kb_id, emb)

    # ---------- CRUD（Postgres） ----------
    def create_kb(self, name: str, scope: str, user_id: str | None,
                  texts: list[str] | None = None) -> KnowledgeBase:
        """建库：写 Postgres（固化嵌入模型标注），有文本则同步入库。"""
        meta = self.resolve_embedding_meta()
        with SessionLocal() as db:
            kb = KnowledgeBase(
                kb_id=uuid.uuid4().hex[:12], name=name, scope=scope,
                owner_user_id=user_id if scope == "private" else None,
                category_id="default",
                embedding_provider=meta.provider,
                embedding_model_id=meta.model_id,
                embedding_dim=meta.dim,
                status="ready",                     # 空库即 ready；有文本走入库后也是 ready
                source_doc_ids=[],
            )
            db.add(kb)
            db.commit()
            kb_id = kb.kb_id
        if texts:
            self.add_documents(kb_id, texts)
        return self.get_kb(kb_id)

    def get_kb(self, kb_id: str) -> KnowledgeBase:
        with SessionLocal() as db:
            kb = db.get(KnowledgeBase, kb_id)
            if not kb:
                raise KeyError(f"kb not found: {kb_id}")
            return kb

    def list_kbs(self, user_id: str | None) -> list[KnowledgeBase]:
        """可见性：public 全员 + private 仅本人。"""
        with SessionLocal() as db:
            stmt = select(KnowledgeBase).where(
                (KnowledgeBase.scope == "public")
                | ((KnowledgeBase.scope == "private")
                   & (KnowledgeBase.owner_user_id == user_id)))
            return list(db.scalars(stmt))

    # ---------- 入库流水线 ----------
    def add_documents(self, kb_id: str, texts: list[str]) -> int:
        """文本入库：分块 → 存 chunk → 向量化 → ready。"""
        return self._ingest(kb_id, [("text.txt", t.encode("utf-8")) for t in texts])

    def ingest_file(self, kb_id: str, filename: str, content: bytes) -> int:
        """单文件入库：置 indexing → 解析 → 分块 → 存 chunk → 向量化 → ready。"""
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.status = "indexing"
            db.commit()
        return self._ingest(kb_id, [(filename, content)])

    def _ingest(self, kb_id: str, files: list[tuple[str, bytes]]) -> int:
        """入库核心：解析全部文件 → 分块 → 落盘 → 向量化 → 更新状态。

        状态机：indexing → ready | failed（架构文档 §11.2 status 四态之一）。
        """
        kb = self.get_kb(kb_id)
        try:
            return self._ingest_inner(kb_id, kb, files)
        except Exception as e:
            logger.error("ingest failed kb=%s: %s", kb_id, e)   # 失败原因进日志
            with SessionLocal() as db:
                row = db.get(KnowledgeBase, kb_id)
                row.status = "failed"
                db.commit()
            emit("ingest", {"kb_id": kb_id, "status": "failed", "error": str(e)[:200]})
            return 0      # 错误已落在 status=failed，不再抛出（后台任务不能把异常带回响应）

    def _ingest_inner(self, kb_id: str, kb: KnowledgeBase,
                      files: list[tuple[str, bytes]]) -> int:
        kb_dir = self._chunk_dir / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[ChunkRecord] = []
        doc_ids: list[str] = []
        for filename, content in files:
            text = parse_file(filename, content)
            # 内容哈希：同一文件重复上传 → 相同 chunk id → Chroma 覆盖（幂等）
            digest = hashlib.sha256(content).hexdigest()[:8]
            doc_ids.append(digest)
            for i, piece in enumerate(split_chunks(text)):
                chunk_id = f"{kb_id}_{digest}_{i}"
                chunks.append(ChunkRecord(
                    id=chunk_id, text=piece,
                    payload={"scope": kb.scope, "user_id": kb.owner_user_id or "",
                             "doc_id": digest, "category_id": kb.category_id}))
                (kb_dir / f"{chunk_id}.txt").write_text(piece, encoding="utf-8")

        self._vector_store(kb).add(chunks)
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.status = "ready"
            existing = row.source_doc_ids or []
            for doc_id in doc_ids:
                if doc_id not in existing:
                    existing.append(doc_id)
            row.source_doc_ids = existing
            db.commit()
        emit("ingest", {"kb_id": kb_id, "chunks": len(chunks), "status": "ready"})
        return len(chunks)

    # ---------- 检索 ----------
    def search(self, kb_id: str, query: str, k: int = 3,
               user_id: str | None = None) -> list[dict]:
        kb = self.get_kb(kb_id)
        hits = self._vector_store(kb).search(query, k=k, scope=kb.scope, user_id=user_id)
        for h in hits:
            h["scope"] = kb.scope
            h["kb_id"] = kb.kb_id
            h["kb_name"] = kb.name
        emit("retrieve", {"kb_id": kb_id, "hits": len(hits), "scope": kb.scope})
        return hits

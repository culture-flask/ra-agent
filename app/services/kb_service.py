"""知识库服务：

- KB 元数据 → Postgres
- 文档解析 → 多格式（parsing.py）
- chunk → 本地盘"对象存储"（data/chunks/{kb_id}/，便于重建复用免解析）
- 状态机：indexing → ready（文件上传走后台异步入库）
"""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import chromadb
from sqlalchemy import select

from app.abstractions.bm25 import Bm25Index, rrf_fuse
from app.abstractions.embedding import EmbeddingFactory, EmbeddingMeta
from app.abstractions.vectorstore import ChunkRecord, VectorStoreFactory
from app.core.crypto import SecretCrypto
from app.core.db import SessionLocal
from app.core.events import emit
from app.core.logging import get_logger
from app.models import KnowledgeBase
from app.services.parsing import parse_file_pages

logger = get_logger("kb_service")


def split_chunks(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
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


def _chunk_index(chunk_id: str) -> int:
    """chunk_id 形如 {kb_id}_{digest}_{i} → 返回文件内序号 i（解析失败返回 0）。"""
    try:
        return int(chunk_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _clean_text(text: str) -> str:
    """剔除文本中的孤立代理字符（lone surrogate）。

    部分 PDF 提取器（如 pypdf 处理含数学符号字体的 PDF）会产出残缺的
    UTF-16 代理半区（如 \ud835），这类字符无法编码为 UTF-8，直接导致
    写 chunk 落盘 / 向量化时 UnicodeEncodeError、整批入库失败。
    完整代理对（合法数学字母等）经 surrogatepass 保留转换，只丢弃孤立半区。
    """
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


class KBService:
    """两级知识库管理：元数据在 Postgres，chunk 在本地盘，向量在 Chroma。"""

    def __init__(self, settings):
        self._settings = settings
        self._chunk_dir = Path(settings.data_dir) / "chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._crypto = SecretCrypto(settings.jwt_secret)      # 库级嵌入密钥加解密
        self._embedding_secrets = {
            **{p: settings.embedding_api_key for p in settings.embedding_cloud},
            "system": settings.embedding_api_key,
        }
        self._bm25_cache: dict[str, Bm25Index] = {}   # kb_id -> BM25 索引（懒构建）
        self._chunks_cache: dict[str, list[ChunkRecord]] = {}  # kb_id -> 全量 chunk（聚合父块用）

    # ---------- 模型解析 ----------
    def resolve_embedding_meta(self, provider: str | None = None,
                               model_id: str | None = None,
                               dim: int | None = None,
                               base_url: str | None = None) -> EmbeddingMeta:
        """解析嵌入模型：缺省用配置默认；可显式指定 provider/model_id/dim（每库标注）。

        任意 OpenAI 兼容端点（llama.cpp / vLLM 等）无需进配置白名单——
        只要显式给出 base_url + model_id + dim，按云端协议直接接入。
        """
        provider = provider or self._settings.embedding_default_provider
        if provider == "local":
            return EmbeddingMeta(provider="local", model_id=model_id or "mini",
                                 dim=dim or self._settings.embedding_local_default_dim)
        cloud = self._settings.embedding_cloud.get(provider)
        if not cloud:
            if not (base_url and model_id and dim):
                raise ValueError(
                    f"unknown embedding provider: {provider} "
                    f"(可用: {', '.join(self._settings.embedding_cloud)}, local；"
                    f"或自定义 OpenAI 兼容端点——需同时提供 base_url / model_id / dim)")
            return EmbeddingMeta(provider=provider, model_id=model_id, dim=dim,
                                 base_url=base_url)
        return EmbeddingMeta(provider=provider,
                             model_id=model_id or cloud["model_id"],
                             dim=dim or cloud["dim"],
                             base_url=base_url or cloud["base_url"])

    def _vector_store(self, kb: KnowledgeBase):
        cloud_cfg = self._settings.embedding_cloud.get(kb.embedding_provider, {})
        rate_limited = bool(cloud_cfg.get("rate_limited"))
        meta = EmbeddingMeta(
            kb.embedding_provider, kb.embedding_model_id, kb.embedding_dim,
            # 库级自定义端点优先，缺省回退该 provider 在配置里的默认 base_url
            kb.embedding_base_url or cloud_cfg.get("base_url"),
            # 自建端点（ollama/llama.cpp 等，配置未标 rate_limited）：零间隔 + 小批次，
            # 降低单请求内存峰值（8B 模型配大 n_ctx 的服务器，大批次容易 OOM/断连）
            batch_delay=None if rate_limited else 0.0,
            batch_size=None if rate_limited else 4)
        emb = EmbeddingFactory.build(meta, self._secrets_for(kb))
        return VectorStoreFactory.build(self._settings, kb.kb_id, emb)

    def _secrets_for(self, kb: KnowledgeBase) -> dict:
        """该库的嵌入密钥集：库级专用 key 优先（解密），缺省回退系统/Provider 默认。"""
        secrets = dict(self._embedding_secrets)
        if kb.embedding_api_key:
            try:
                secrets[kb.embedding_provider] = self._crypto.decrypt(kb.embedding_api_key)
            except Exception as e:
                logger.warning("decrypt embedding key failed kb=%s: %s", kb.kb_id, e)
        return secrets

    # ---------- CRUD（Postgres） ----------
    def create_kb(self, name: str, scope: str, user_id: str | None,
                  texts: list[str] | None = None,
                  provider: str | None = None,
                  model_id: str | None = None,
                  dim: int | None = None,
                  api_key: str | None = None,
                  base_url: str | None = None) -> KnowledgeBase:
        """建库：写 Postgres（固化嵌入模型标注），有文本则同步入库。
        可显式指定嵌入模型（provider/model_id/dim）与专用 base_url/api_key，缺省用配置默认。"""
        meta = self.resolve_embedding_meta(provider, model_id, dim, base_url)
        with SessionLocal() as db:
            kb = KnowledgeBase(
                kb_id=uuid.uuid4().hex[:12], name=name, scope=scope,
                owner_user_id=user_id if scope == "private" else None,
                category_id="default",
                embedding_provider=meta.provider,
                embedding_model_id=meta.model_id,
                embedding_dim=meta.dim,
                embedding_base_url=base_url or None,
                embedding_api_key=self._crypto.encrypt(api_key) if api_key else None,
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

    def list_queryable_kbs(self, user_id: str | None) -> list[KnowledgeBase]:
        """对话可检索的库：可见性基础上排除用户主动禁用的（retrieval_enabled=False）。"""
        return [kb for kb in self.list_kbs(user_id) if kb.retrieval_enabled]

    def set_retrieval(self, kb_id: str, enabled: bool) -> KnowledgeBase:
        """允许/禁止该库被对话检索（库本身保留，随时可恢复）。"""
        self.get_kb(kb_id)                        # 不存在抛 KeyError
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.retrieval_enabled = bool(enabled)
            db.commit()
        logger.info("kb=%s retrieval_enabled -> %s", kb_id, enabled)
        return self.get_kb(kb_id)

    def delete_kb(self, kb_id: str) -> None:
        """删除知识库：Chroma collection + 磁盘 chunk + Postgres 元数据，三段都清理。"""
        kb = self.get_kb(kb_id)                          # 不存在抛 KeyError
        # 1) 向量 collection（可能从未写入过，容错）
        try:
            client = chromadb.PersistentClient(
                path=str(self._settings.chroma_persist_dir))
            client.delete_collection(f"kb_{kb_id}")
        except Exception as e:
            logger.warning("delete collection failed kb=%s: %s", kb_id, e)
        # 2) 磁盘 chunk 目录
        shutil.rmtree(self._chunk_dir / kb_id, ignore_errors=True)
        self._bm25_cache.pop(kb_id, None)              # 3.5) BM25/父块 chunk 缓存失效
        self._chunks_cache.pop(kb_id, None)
        # 3) Postgres 元数据
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            if row:
                db.delete(row)
                db.commit()

    # ---------- 入库流水线 ----------
    def add_documents(self, kb_id: str, texts: list[str]) -> int:
        """文本入库：分块 → 存 chunk → 向量化 → ready。"""
        return self._ingest(kb_id, [("text.txt", t.encode("utf-8")) for t in texts])

    def ingest_files(self, kb_id: str, files: list[tuple[str, bytes]]) -> int:
        """批量文件入库：一次置 indexing → 解析全部文件 → 分块 → 存 chunk → 向量化 → ready。

        一次请求的多个文件走同一条后台任务，状态机只流转一次
        （indexing → ready | failed），避免并发任务互相覆盖 status。
        """
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.status = "indexing"
            db.commit()
        return self._ingest(kb_id, files)

    def ingest_file(self, kb_id: str, filename: str, content: bytes) -> int:
        """单文件入库（批量接口的单文件特例）。"""
        return self.ingest_files(kb_id, [(filename, content)])

    def _ingest(self, kb_id: str, files: list[tuple[str, bytes]]) -> int:
        """入库核心：解析全部文件 → 分块 → 落盘 → 向量化 → 更新状态。

        状态机：indexing → ready | failed（架构文档 §11.2 status 四态之一）。
        """
        kb = self.get_kb(kb_id)
        try:
            n = self._ingest_inner(kb_id, kb, files)
            self._bm25_cache.pop(kb_id, None)          # chunk 变了，BM25/父块缓存失效
            self._chunks_cache.pop(kb_id, None)
            return n
        except Exception as e:
            logger.error("ingest failed kb=%s: %s", kb_id, e)   # 失败原因进日志
            if e.__cause__:
                # Chroma 等会把底层异常包一层（如 "Connection error. in add."），
                # 记下真实原因（连接谁、什么错），不然排查只能看到包装消息
                logger.error("ingest failed kb=%s (cause %s): %s",
                             kb_id, type(e.__cause__).__name__, e.__cause__)
            # 清掉本次失败留下的半截 chunk，避免脏文件残留（重新上传会重建）
            shutil.rmtree(self._chunk_dir / kb_id, ignore_errors=True)
            with SessionLocal() as db:
                row = db.get(KnowledgeBase, kb_id)
                if row is not None:            # 入库途中库被删 → 不再写失败状态
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
            try:
                # 逐页解析（PDF 带页码）：_clean_text 剔除 pypdf 等提取出的
                # 孤立代理字符，避免编码崩溃
                pages = parse_file_pages(filename, content)
                # 内容哈希：同一文件重复上传 → 相同 chunk id → Chroma 覆盖（幂等）
                digest = hashlib.sha256(content).hexdigest()[:8]
                file_chunks: list[ChunkRecord] = []
                i = 0
                for page_no, page_text in pages:
                    for piece in split_chunks(_clean_text(page_text)):
                        chunk_id = f"{kb_id}_{digest}_{i}"
                        payload = {"scope": kb.scope,
                                   "user_id": kb.owner_user_id or "",
                                   "doc_id": digest,
                                   "category_id": kb.category_id,
                                   "source": filename}     # 源文件名（引用溯源）
                        if page_no is not None:
                            payload["page"] = page_no      # 页码（Chroma 不接受 None 值）
                        file_chunks.append(ChunkRecord(
                            id=chunk_id, text=piece, payload=payload))
                        (kb_dir / f"{chunk_id}.txt").write_text(piece, encoding="utf-8")
                        # 伴生元数据文件：重建/BM25 从磁盘读 chunk 时恢复 source/page
                        (kb_dir / f"{chunk_id}.meta.json").write_text(
                            json.dumps({"source": filename, "page": page_no},
                                       ensure_ascii=False), encoding="utf-8")
                        i += 1
                # 整个文件成功后才登记 doc_id + chunks，避免半截文件污染批次
                doc_ids.append(digest)
                chunks.extend(file_chunks)
            except Exception as e:
                # 单文件容错：坏文件只跳过并记日志，不影响同批其他文件入库
                logger.warning("skip file %s (kb=%s): %s", filename, kb_id, e)
        if not doc_ids:
            raise ValueError("所有文件都解析失败，无文档可入库")

        self._vector_store(kb).add(chunks)
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.status = "ready"
            # 关键：list(...) 拷贝成新对象再改。若直接对加载出的 list 就地 append，
            # SQLAlchemy JSON 列检测不到"原地突变"，第二次入库的 doc_id 会丢。
            existing = list(row.source_doc_ids or [])
            for doc_id in doc_ids:
                if doc_id not in existing:
                    existing.append(doc_id)
            row.source_doc_ids = existing
            # 记录这批向量实际由哪个嵌入模型写入（查询时用来比对，模型不一致则提醒）
            row.embedded_model = {
                "provider": kb.embedding_provider,
                "model_id": kb.embedding_model_id,
                "dim": kb.embedding_dim,
                "base_url": kb.embedding_base_url,
            }
            db.commit()
        emit("ingest", {"kb_id": kb_id, "chunks": len(chunks), "status": "ready"})
        return len(chunks)

    # ---------- 检索 ----------
    def embedding_mismatch(self, kb: KnowledgeBase) -> str | None:
        """比对"向量实际由谁嵌入"与"当前查询配置"：模型不一致时返回提醒文案。

        只比对 provider/model_id/dim —— 仅换端点（base_url）不影响向量语义，不提醒。
        """
        em = kb.embedded_model or {}
        same = (em.get("provider") == kb.embedding_provider
                and em.get("model_id") == kb.embedding_model_id
                and em.get("dim") == kb.embedding_dim)
        if same:
            return None
        if not em:
            return None        # 还没入库过任何向量，无从比对
        return (f"向量库由 {em.get('provider')}/{em.get('model_id')}（dim {em.get('dim')}）"
                f"嵌入，当前查询使用 {kb.embedding_provider}/{kb.embedding_model_id}"
                f"（dim {kb.embedding_dim}），检索结果可能不准确；"
                f"如需让向量匹配新模型，请重新上传文档或使用「重建」生成新库")

    def _bm25_index(self, kb: KnowledgeBase) -> Bm25Index:
        """该库的 BM25 索引：进程内缓存 + 懒构建（chunk 从磁盘读，与嵌入模型解耦）。

        入库/重建/删除后由调用方清除缓存，重启后自动重建。
        """
        idx = self._bm25_cache.get(kb.kb_id)
        if idx is None:
            idx = Bm25Index()
            idx.build(self._read_chunks(kb.kb_id, kb))
            self._bm25_cache[kb.kb_id] = idx
        return idx

    def search(self, kb_id: str, query: str, k: int = 3,
               user_id: str | None = None,
               mode: str | None = None) -> list[dict]:
        """检索一个知识库：mode=vector 纯向量；mode=hybrid 向量 + BM25 融合。

        默认取全局配置 retrieval_mode；结果统一带 scope/kb_id/kb_name/method：
        - vector：distance（越小越近）
        - hybrid：distance（向量分）+ bm25_score（词频分）+ score（RRF 融合分，越大越好）
        """
        kb = self.get_kb(kb_id)
        mode = mode or self._settings.retrieval_mode
        warn = self.embedding_mismatch(kb)
        if warn:
            logger.warning("embedding mismatch kb=%s: %s", kb_id, warn)

        if mode == "hybrid":
            vec_hits = self._vector_store(kb).search(
                query, k=max(k, 8), scope=kb.scope, user_id=user_id)
            bm_hits = self._bm25_index(kb).search(query, k=max(k, 8))
            hits = rrf_fuse([self._annotate(kb, h) for h in vec_hits],
                            [self._annotate(kb, h) for h in bm_hits],
                            top=k)
        else:
            hits = [self._annotate(kb, h)
                    for h in self._vector_store(kb).search(
                        query, k=k, scope=kb.scope, user_id=user_id)]
        emit("retrieve", {"kb_id": kb_id, "hits": len(hits), "scope": kb.scope,
                          "mode": mode})
        return hits

    @staticmethod
    def _annotate(kb: KnowledgeBase, h: dict) -> dict:
        """补知识库归属字段 + 命中方式标记（vector/bm25/hybrid 两路都命中）。"""
        out = dict(h)
        out.setdefault("scope", kb.scope)
        out.setdefault("kb_id", kb.kb_id)
        out.setdefault("kb_name", kb.name)
        has_vec = out.get("distance") is not None
        has_bm = out.get("bm25_score") is not None
        out["method"] = "hybrid" if (has_vec and has_bm) else (
            "bm25" if has_bm else "vector")
        return out

    # ---------- 嵌入配置修改 ----------
    def update_embedding(self, kb_id: str, provider: str | None = None,
                         model_id: str | None = None, dim: int | None = None,
                         base_url: str | None = None,
                         api_key: str | None = None) -> KnowledgeBase:
        """修改知识库的嵌入配置（创建后随时可改，不再固化）。

        - 传了才改：provider/model_id/dim 仅在有值时更新
        - base_url：None=保持不变；空串=清空（回退 provider 默认端点）
        - api_key：None=保持不变；空串=清除专用密钥；其他=加密更新
        - 已入库的向量不会重新嵌入，查询时由 embedding_mismatch 提醒
        """
        kb = self.get_kb(kb_id)
        if kb.status == "indexing":
            raise ValueError("knowledge base is indexing: 入库中请勿修改嵌入配置")
        new_provider = provider or kb.embedding_provider
        new_model = model_id or kb.embedding_model_id
        new_dim = dim or kb.embedding_dim
        new_base = kb.embedding_base_url if base_url is None else (base_url.strip() or None)
        meta = self.resolve_embedding_meta(new_provider, new_model, new_dim, new_base)
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.embedding_provider = meta.provider
            row.embedding_model_id = meta.model_id
            row.embedding_dim = meta.dim
            row.embedding_base_url = meta.base_url
            if api_key is not None:
                row.embedding_api_key = self._crypto.encrypt(api_key) if api_key else None
            db.commit()
        logger.info("kb=%s embedding config updated: %s/%s dim=%s base=%s",
                    kb_id, meta.provider, meta.model_id, meta.dim, meta.base_url)
        return self.get_kb(kb_id)

    # ---------- 知识库重建 ----------
    def rebuild(self, src_kb_id: str, provider: str | None = None,
                model_id: str | None = None, dim: int | None = None,
                api_key: str | None = None,
                base_url: str | None = None) -> KnowledgeBase:
        """复制原 chunk + 新嵌入模型重新向量化 → 新 KB（新 collection），旧库保留。

        免重新解析：chunk 已落盘，直接读盘重向量化。
        api_key / base_url 缺省继承原库；但 base_url 仅在 provider 未变时才继承
        （换了 provider 还沿用旧端点是错的，此时回退新 provider 的默认端点）。
        状态机：reembedding → ready | failed。
        """
        src = self.get_kb(src_kb_id)
        meta = self.resolve_embedding_meta(provider, model_id, dim, base_url)
        new_base_url = base_url if base_url is not None else (
            src.embedding_base_url if meta.provider == src.embedding_provider else None)
        new_kb = KnowledgeBase(
            kb_id=uuid.uuid4().hex[:12], name=f"{src.name}{model_id or ''}(重建)",
            scope=src.scope, owner_user_id=src.owner_user_id,
            category_id=src.category_id,
            embedding_provider=meta.provider, embedding_model_id=meta.model_id,
            embedding_dim=meta.dim, status="reembedding",
            embedding_base_url=new_base_url,
            embedding_api_key=self._crypto.encrypt(api_key) if api_key else src.embedding_api_key,
            source_doc_ids=list(src.source_doc_ids or []),
        )
        with SessionLocal() as db:
            db.add(new_kb)
            db.commit()
        try:
            chunks = self._read_chunks(src_kb_id, new_kb)     # 复用已存 chunk
            self._vector_store(new_kb).add(chunks)            # 新 collection 重向量化
            # 重建库的 chunk 物理上在 src 目录：BM25 索引与父块 chunk 直接缓存，
            # 否则懒构建会读到空目录
            bm = Bm25Index()
            bm.build(chunks)
            self._bm25_cache[new_kb.kb_id] = bm
            self._chunks_cache[new_kb.kb_id] = list(chunks)
            with SessionLocal() as db:
                row = db.get(KnowledgeBase, new_kb.kb_id)
                row.status = "ready"
                row.embedded_model = {                       # 新库向量由重建配置写入
                    "provider": new_kb.embedding_provider,
                    "model_id": new_kb.embedding_model_id,
                    "dim": new_kb.embedding_dim,
                    "base_url": new_kb.embedding_base_url,
                }
                db.commit()
            emit("rebuild", {"src_kb_id": src_kb_id, "new_kb_id": new_kb.kb_id,
                             "chunks": len(chunks), "status": "ready"})
        except Exception as e:
            logger.error("rebuild failed src=%s: %s", src_kb_id, e)
            with SessionLocal() as db:
                row = db.get(KnowledgeBase, new_kb.kb_id)
                row.status = "failed"
                db.commit()
            raise
        return self.get_kb(new_kb.kb_id)      # 返回 DB 最新状态（含 ready）

    def _doc_chunks(self, kb_id: str, kb: KnowledgeBase) -> list[ChunkRecord]:
        """该库全部 chunk（磁盘读取），进程内缓存（与 BM25 同步失效）。"""
        chunks = self._chunks_cache.get(kb_id)
        if chunks is None:
            chunks = self._read_chunks(kb_id, kb)
            self._chunks_cache[kb_id] = chunks
        return chunks

    def get_parent_block(self, kb_id: str, doc_id: str, group_no: int,
                         group_size: int = 3,
                         max_chars: int = 4000) -> dict | None:
        """聚合父块：同一 doc 内序号 i//group_size==group_no 的 chunk 按序拼接。

        父块内容运行时从磁盘 chunk 拼接——旧数据/重建库天然兼容，无需重新入库。
        返回 {text, source, pages, chunk_ids, doc_id, group}；组不存在返回 None。
        """
        kb = self.get_kb(kb_id)
        members = [c for c in self._doc_chunks(kb_id, kb)
                   if c.payload.get("doc_id") == doc_id
                   and _chunk_index(c.id) // group_size == group_no]
        if not members:
            return None
        members.sort(key=lambda c: _chunk_index(c.id))
        parts: list[str] = []
        pages: list[int] = []
        total = 0
        for c in members:
            page = c.payload.get("page")
            if isinstance(page, int) and page not in pages:
                pages.append(page)
            if total >= max_chars:
                break
            text = c.text
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
            parts.append(text)
            total += len(text)
        return {
            "text": "".join(parts),
            "source": members[0].payload.get("source"),
            "pages": sorted(pages),
            "chunk_ids": [c.id for c in members],
            "doc_id": doc_id,
            "group": group_no,
        }

    def _read_chunks(self, kb_id: str, kb: KnowledgeBase) -> list[ChunkRecord]:
        """从磁盘读回该库的 chunk（含伴生元数据：source 源文件名 / page 页码）。

        旧格式没有 .meta.json → 回退（source/page 为 None），兼容历史数据。
        """
        kb_dir = self._chunk_dir / kb_id
        chunks: list[ChunkRecord] = []
        for f in sorted(kb_dir.glob("*.txt")):
            parts = f.stem.split("_")
            digest = parts[1] if len(parts) > 2 else "doc"
            meta: dict = {}
            meta_path = kb_dir / f"{f.stem}.meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("read chunk meta failed %s: %s", f.stem, e)
            payload = {"scope": kb.scope, "user_id": kb.owner_user_id or "",
                       "doc_id": digest, "category_id": kb.category_id,
                       "source": meta.get("source")}
            if meta.get("page") is not None:
                payload["page"] = meta.get("page")
            chunks.append(ChunkRecord(
                id=f.stem, text=f.read_text(encoding="utf-8"), payload=payload))
        return chunks

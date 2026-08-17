"""知识库服务：

- KB 元数据 → Postgres
- 文档解析 → 多格式（parsing.py）
- chunk → 本地盘"对象存储"（data/chunks/{kb_id}/，便于重建复用免解析）
- 源文件归档 → data/docs/{kb_id}/{doc_id}__文件名（原始上传件留底）
- 状态机：indexing → ready（文件上传走后台异步入库）
"""

import hashlib
import json
import re
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


def _safe_filename(name: str) -> str:
    """上传文件名 → 可安全落盘的名字：去路径成分（防目录穿越）、
    替换文件系统特殊字符、限长。\w 默认含中文等 unicode 字母，中文名保留。
    """
    base = Path(name.replace("\\", "/")).name.strip()
    if base in ("", ".", ".."):               # Path('..').name 仍是 '..'，需显式兜底
        base = "upload.bin"
    safe = re.sub(r"[^\w.\-]+", "_", base)
    return safe[:150] or "upload.bin"


class KBService:
    """两级知识库管理：元数据在 Postgres，chunk 在本地盘，向量在 Chroma。"""

    def __init__(self, settings):
        self._settings = settings
        self._chunk_dir = Path(settings.data_dir) / "chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._docs_dir = Path(settings.data_dir) / "docs"   # 上传源文件归档目录
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._crypto = SecretCrypto(settings.jwt_secret)      # 库级嵌入密钥加解密
        self._embedding_secrets = {
            **{p: settings.embedding_api_key for p in settings.embedding_cloud},
            "system": settings.embedding_api_key,
        }
        self._bm25_cache: dict[str, Bm25Index] = {}   # kb_id -> BM25 索引（懒构建）
        self._chunks_cache: dict[str, list[ChunkRecord]] = {}  # kb_id -> 全量 chunk（聚合父块用）
        self._ingest_progress: dict[str, dict] = {}   # kb_id -> 最近一批入库进度（前端轮询）

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
                  base_url: str | None = None,
                  description: str = "") -> KnowledgeBase:
        """建库：写 Postgres（固化嵌入模型标注），有文本则同步入库。

        可显式指定嵌入模型（provider/model_id/dim）与专用 base_url/api_key，
        缺省用配置默认。description 为知识库介绍（LLM 选库参考，API 层必填）。
        """
        meta = self.resolve_embedding_meta(provider, model_id, dim, base_url)
        with SessionLocal() as db:
            kb = KnowledgeBase(
                kb_id=uuid.uuid4().hex[:12], name=name, scope=scope,
                description=description.strip(),
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

    def update_info(self, kb_id: str, name: str | None = None,
                    description: str | None = None) -> KnowledgeBase:
        """修改知识库名称/介绍（创建后随时可改；传 None 表示该项不改）。"""
        self.get_kb(kb_id)                        # 不存在抛 KeyError
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            if name is not None:
                name = name.strip()
                if not name:
                    raise ValueError("knowledge base name cannot be empty")
                row.name = name[:128]
            if description is not None:
                row.description = description.strip()[:512]
            db.commit()
        return self.get_kb(kb_id)

    def list_kbs(self, user_id: str | None) -> list[KnowledgeBase]:
        """可见性：public 全员 + private 仅本人。"""
        with SessionLocal() as db:
            stmt = select(KnowledgeBase).where(
                (KnowledgeBase.scope == "public")
                | ((KnowledgeBase.scope == "private")
                   & (KnowledgeBase.owner_user_id == user_id)))
            return list(db.scalars(stmt))

    @staticmethod
    def kb_queryable(kb: KnowledgeBase, user_id: str | None) -> bool:
        """该用户能否检索此库：库主全局开关 AND 该用户未在个人禁用列表。

        禁用列表按用户隔离——用户 A 禁用不影响用户 B（检索偏好是个人行为）。
        """
        return (kb.retrieval_enabled
                and user_id not in (kb.retrieval_disabled_users or []))

    def list_queryable_kbs(self, user_id: str | None) -> list[KnowledgeBase]:
        """对话可检索的库：可见性基础上排除该用户自己禁用的（per-user）。"""
        return [kb for kb in self.list_kbs(user_id)
                if self.kb_queryable(kb, user_id)]

    def set_retrieval(self, kb_id: str, user_id: str,
                      enabled: bool) -> KnowledgeBase:
        """允许/禁止【该用户】的对话检索此库（库本身保留，随时可恢复）。

        per-user：写入 retrieval_disabled_users 列表，其他用户的开关不受影响。
        开启时若库级总开关 retrieval_enabled 为 False（per-user 改造前留下的
        旧全局禁用）一并拉回 True——否则该库永远无法恢复检索（死锁）。
        """
        self.get_kb(kb_id)                        # 不存在抛 KeyError
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            # 拷贝新 list 再改（JSON 列检测不到原地突变）
            disabled = list(row.retrieval_disabled_users or [])
            if enabled:
                disabled = [u for u in disabled if u != user_id]
                row.retrieval_enabled = True      # 修复旧的全局禁用卡死
            elif user_id not in disabled:
                disabled.append(user_id)
            row.retrieval_disabled_users = disabled
            db.commit()
        logger.info("kb=%s retrieval for user=%s -> %s", kb_id, user_id, enabled)
        return self.get_kb(kb_id)

# ---------- 文件管理 ----------
    def list_documents(self, kb_id: str) -> list[dict]:
        """该库的源文件列表：doc_id（内容哈希）、原始文件名、chunk 数、页码范围。

        文件名来自 chunk 的 source 元数据（伴生 meta 文件）；旧数据没有
        则回退显示 doc_id。
        """
        kb = self.get_kb(kb_id)
        docs: dict[str, dict] = {}
        for c in self._doc_chunks(kb_id, kb):
            doc_id = c.payload.get("doc_id") or "doc"
            entry = docs.setdefault(doc_id, {"doc_id": doc_id, "filename": None,
                                             "chunks": 0, "pages": []})
            entry["chunks"] += 1
            page = c.payload.get("page")
            if isinstance(page, int) and page not in entry["pages"]:
                entry["pages"].append(page)
            if entry["filename"] is None and c.payload.get("source"):
                entry["filename"] = c.payload["source"]
        return sorted(docs.values(),
                      key=lambda d: (d["filename"] or d["doc_id"]).lower())

    def delete_documents(self, kb_id: str, doc_ids: list[str]) -> dict:
        """删除该库的若干源文件（按 doc_id 内容哈希）：Chroma chunk + 磁盘 + 元数据。

        返回 {"files": 删除的文件数, "chunks": 删除的片段数}；不存在的 doc_id 无害。
        """
        doc_set = set(doc_ids or [])
        if not doc_set:
            return {"files": 0, "chunks": 0}
        kb_dir = self._chunk_dir / kb_id
        removed_chunks = 0
        # 1) 磁盘 chunk 文本 + 伴生 meta（文件名里第二段是 doc_id）
        for f in list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.meta.json")):
            parts = f.stem.split("_")
            if len(parts) > 2 and parts[1] in doc_set:
                try:
                    f.unlink()
                    removed_chunks += 1 if f.suffix == ".txt" else 0
                except OSError as e:
                    logger.warning("remove file failed %s: %s", f, e)
        # 2) Chroma：按 metadata doc_id 过滤删除该文件的所有向量
        try:
            client = chromadb.PersistentClient(
                path=str(self._settings.chroma_persist_dir))
            col = client.get_collection(f"kb_{kb_id}")
            for doc_id in doc_set:
                col.delete(where={"doc_id": doc_id})
        except Exception as e:
            logger.warning("delete chroma chunks failed kb=%s: %s", kb_id, e)
        # 2.5) 归档的源文件（data/docs/{kb_id}/{doc_id}__*，与 chunk 同生命周期）
        kb_docs = self._docs_dir / kb_id
        if kb_docs.exists():
            for doc_id in doc_set:
                for f in kb_docs.glob(f"{doc_id}__*"):
                    try:
                        f.unlink()
                    except OSError as e:
                        logger.warning("remove source file failed %s: %s", f, e)
        # 3) Postgres 元数据：source_doc_ids 移除
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            if row is not None:
                row.source_doc_ids = [d for d in (row.source_doc_ids or [])
                                      if d not in doc_set]
                db.commit()
        # 4) 缓存失效：BM25 / 父块 chunk（下次检索懒重建）
        self._bm25_cache.pop(kb_id, None)
        self._chunks_cache.pop(kb_id, None)
        files = len(doc_set)
        emit("doc_delete", {"kb_id": kb_id, "files": files,
                            "chunks": removed_chunks})
        return {"files": files, "chunks": removed_chunks}


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
        # 2) 磁盘 chunk 目录 + 源文件归档目录
        shutil.rmtree(self._chunk_dir / kb_id, ignore_errors=True)
        shutil.rmtree(self._docs_dir / kb_id, ignore_errors=True)
        self._bm25_cache.pop(kb_id, None)              # 3.5) BM25/父块 chunk 缓存失效
        self._chunks_cache.pop(kb_id, None)
        # 3) Postgres 元数据
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            if row:
                db.delete(row)
                db.commit()

    # ---------- 入库流水线 ----------
    def _save_source_file(self, kb_id: str, doc_id: str,
                          filename: str, content: bytes) -> None:
        """源文件归档：data/docs/{kb_id}/{doc_id}__文件名（原始上传件留底）。

        doc_id（内容哈希）做前缀保证唯一：同内容换名重传时先清旧档再写新名，
        一个 doc 恒占一个文件。归档失败只记日志，不影响入库主流程。
        """
        try:
            kb_docs = self._docs_dir / kb_id
            kb_docs.mkdir(parents=True, exist_ok=True)
            for old in kb_docs.glob(f"{doc_id}__*"):
                old.unlink()
            (kb_docs / f"{doc_id}__{_safe_filename(filename)}").write_bytes(content)
        except OSError as e:
            logger.warning("save source file failed kb=%s doc=%s: %s",
                           kb_id, doc_id, e)

    def add_documents(self, kb_id: str, texts: list[str]) -> int:
        """文本入库：分块 → 存 chunk → 向量化 → ready。"""
        return self._ingest(kb_id, [("text.txt", t.encode("utf-8")) for t in texts])

    def ingest_files(self, kb_id: str, files: list[tuple[str, bytes]]) -> int:
        """批量文件入库：逐文件独立走 解析 -> 分块 -> 落盘 -> 向量化 -> ready。

        每个文件自成一个小事务：单个文件失败只记录错误并继续下一个，
        不再整批失败；进度与逐文件错误明细存 _ingest_progress 供前端轮询。
        """
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            row.status = "indexing"
            db.commit()
        return self._ingest(kb_id, files)

    def ingest_file(self, kb_id: str, filename: str, content: bytes) -> int:
        """单文件入库（批量接口的单文件特例）。"""
        return self.ingest_files(kb_id, [(filename, content)])

    def ingest_progress(self, kb_id: str) -> dict | None:
        """最近一批入库进度：{total, done, succeeded, failed, status, files:[...]}。

        files 每项 {filename, status: pending|ok|failed, chunks, error}；
        入库结束后保留（终态含错误明细），下一批上传时覆盖。
        """
        return self._ingest_progress.get(kb_id)

    def _ingest(self, kb_id: str, files: list[tuple[str, bytes]]) -> int:
        """逐文件入库编排：失败隔离（单文件不影响其他）+ 实时进度跟踪。

        终态：>=1 个文件成功 -> ready（失败的留在进度明细里）；全部失败 -> failed。
        """
        kb = self.get_kb(kb_id)
        # 嵌入配置与已入库向量的模型不一致：只提醒不拦截——两个 provider
        # 可能服务同一个模型（向量依然兼容）；真不兼容（维度不同/端点不可达）
        # 由各文件向量化失败落到进度明细，绝不牵连删除库内已有文件
        warn = self.embedding_mismatch(kb)
        if warn:
            logger.warning("ingest kb=%s under changed embedding config: %s",
                           kb_id, warn)
        prog = {
            "total": len(files), "done": 0, "succeeded": 0, "failed": 0,
            "current": None, "status": "indexing", "warning": warn,
            "files": [{"filename": name, "status": "pending",
                       "chunks": 0, "error": None} for name, _ in files],
        }
        self._ingest_progress[kb_id] = prog
        ok_ids: list[str] = []
        ok_chunks = 0
        try:
            vs = self._vector_store(kb)   # 配置级错误（缺密钥等）在这里 fail fast
        except Exception as e:
            logger.error("build vector store failed kb=%s: %s", kb_id, e)
            for entry in prog["files"]:
                entry.update(status="failed", error=f"嵌入配置不可用: {e}")
            prog["failed"], prog["done"] = prog["total"], prog["total"]
            self._finish_ingest(kb_id, kb, prog, ok_ids, ok_chunks)
            return 0
        for i, (filename, content) in enumerate(files):
            entry = prog["files"][i]
            prog["current"] = filename
            try:
                digest, n = self._ingest_one(kb, kb_id, vs, filename, content)
                entry.update(status="ok", chunks=n)
                prog["succeeded"] += 1
                ok_chunks += n
                if digest not in ok_ids:
                    ok_ids.append(digest)
            except Exception as e:
                logger.warning("ingest file failed %s (kb=%s): %s",
                               filename, kb_id, e)
                if e.__cause__:
                    # Chroma/嵌入 API 会把底层异常包一层，记下真实原因便于排查
                    logger.warning("ingest file failed %s (kb=%s) cause %s: %s",
                                   filename, kb_id,
                                   type(e.__cause__).__name__, e.__cause__)
                # 只清该文件的残留 chunk，库内已有文件与其他文件不受影响
                self._cleanup_failed_batch(kb_id, [(filename, content)])
                entry.update(status="failed", error=str(e)[:300])
                prog["failed"] += 1
            prog["done"] += 1
            emit("ingest_progress", {"kb_id": kb_id, "done": prog["done"],
                                     "total": prog["total"], "file": filename})
        self._finish_ingest(kb_id, kb, prog, ok_ids, ok_chunks)
        return ok_chunks

    def _ingest_one(self, kb: KnowledgeBase, kb_id: str, vs,
                    filename: str, content: bytes) -> tuple[str, int]:
        """单文件入库：解析 -> 分块 -> 落盘 -> 向量化 -> 源文件归档。

        任一步失败抛异常，由 _ingest 记录错误、清理残留后继续下一个文件。
        返回 (doc_id 内容哈希, chunk 数)。
        """
        kb_dir = self._chunk_dir / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        pages = parse_file_pages(filename, content)   # 不支持的格式等在这里抛
        digest = hashlib.sha256(content).hexdigest()[:8]
        chunks: list[ChunkRecord] = []
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
                chunks.append(ChunkRecord(
                    id=chunk_id, text=piece, payload=payload))
                (kb_dir / f"{chunk_id}.txt").write_text(piece, encoding="utf-8")
                # 伴生元数据文件：重建/BM25 从磁盘读 chunk 时恢复 source/page
                (kb_dir / f"{chunk_id}.meta.json").write_text(
                    json.dumps({"source": filename, "page": page_no},
                               ensure_ascii=False), encoding="utf-8")
                i += 1
        vs.add(chunks)              # 向量化失败 -> 调用方清该文件残留后继续
        self._save_source_file(kb_id, digest, filename, content)
        return digest, len(chunks)

    def _finish_ingest(self, kb_id: str, kb: KnowledgeBase, prog: dict,
                       ok_ids: list[str], ok_chunks: int) -> None:
        """收尾：终态判定 + 元数据落库（source_doc_ids/embedded_model）+ 缓存失效。"""
        status = "ready" if prog["succeeded"] else "failed"
        prog["status"], prog["current"] = status, None
        with SessionLocal() as db:
            row = db.get(KnowledgeBase, kb_id)
            if row is not None:            # 入库途中库被删 -> 不再写状态
                row.status = status
                if ok_ids:
                    # 关键：list(...) 拷贝成新对象再改。若直接对加载出的 list
                    # 就地 append，SQLAlchemy JSON 列检测不到"原地突变"，
                    # 第二次入库的 doc_id 会丢。
                    existing = list(row.source_doc_ids or [])
                    for doc_id in ok_ids:
                        if doc_id not in existing:
                            existing.append(doc_id)
                    row.source_doc_ids = existing
                    # 记录这批向量实际由哪个嵌入模型写入（查询时比对提醒）
                    row.embedded_model = {
                        "provider": kb.embedding_provider,
                        "model_id": kb.embedding_model_id,
                        "dim": kb.embedding_dim,
                        "base_url": kb.embedding_base_url,
                    }
                db.commit()
        self._bm25_cache.pop(kb_id, None)          # chunk 变了，BM25/父块缓存失效
        self._chunks_cache.pop(kb_id, None)
        emit("ingest", {"kb_id": kb_id, "chunks": ok_chunks, "status": status,
                        "succeeded": prog["succeeded"], "failed": prog["failed"]})

    def _cleanup_failed_batch(self, kb_id: str,
                              files: list[tuple[str, bytes]]) -> None:
        """删除指定文件集的残留 chunk（按内容哈希定位），保留库内已有文件。

        换嵌入模型后维度不匹配、解析失败等场景，错误只落在进度明细，
        历史文件的 chunk 绝不能被牵连删除（重新上传会重建）。
        """
        digests = {hashlib.sha256(content).hexdigest()[:8] for _, content in files}
        kb_dir = self._chunk_dir / kb_id
        for f in list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.meta.json")):
            parts = f.stem.split("_")
            if len(parts) > 2 and parts[1] in digests:
                try:
                    f.unlink()
                except OSError as e:
                    logger.warning("remove chunk file failed %s: %s", f, e)

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
            # 每腿取 max(k, 15)：候选池略深于输出 k，避免一路深位（第 9~15 名）
            # 的好命中在 RRF 融合时被另一路浅位挤出（26 查询网格实验最优值，
            # 见 rag_test/hparam_eval_report_v4.md）
            vec_hits = self._vector_store(kb).search(
                query, k=max(k, 15), scope=kb.scope, user_id=user_id)
            bm_hits = self._bm25_index(kb).search(query, k=max(k, 15))
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
                base_url: str | None = None,
                user_id: str | None = None) -> KnowledgeBase:
        """复制原 chunk + 新嵌入模型重新向量化 → 新 KB（新 collection），旧库保留。

        免重新解析：chunk 已落盘，直接读盘重向量化。
        api_key / base_url 缺省继承原库；但 base_url 仅在 provider 未变时才继承
        （换了 provider 还沿用旧端点是错的，此时回退新 provider 的默认端点）。
        状态机：reembedding → ready | failed。

        ⚠️ 重建库强制为「私人」库（归属发起重建的用户）：公共库被任何用户重建
        都会变成共有库，影响其他用户——重建只影响发起者自己，不污染公共库。
        """
        src = self.get_kb(src_kb_id)
        meta = self.resolve_embedding_meta(provider, model_id, dim, base_url)
        new_base_url = base_url if base_url is not None else (
            src.embedding_base_url if meta.provider == src.embedding_provider else None)
        new_kb = KnowledgeBase(
            kb_id=uuid.uuid4().hex[:12], name=f"{src.name}{model_id or ''}(重建)",
            scope="private", owner_user_id=user_id or src.owner_user_id,
            description=src.description,          # 介绍继承原库
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
            # 先把源库 chunk 目录完整复制一份：新库拥有独立磁盘数据，
            # 之后删源库/删源文件都不会波及重建库（向量本来就在独立 collection）
            src_dir = self._chunk_dir / src_kb_id
            if src_dir.exists():
                shutil.copytree(src_dir, self._chunk_dir / new_kb.kb_id,
                                dirs_exist_ok=True)
            # 源文件归档目录同样复制一份（新库文件管理/留底不受源库删除影响）
            src_docs = self._docs_dir / src_kb_id
            if src_docs.exists():
                shutil.copytree(src_docs, self._docs_dir / new_kb.kb_id,
                                dirs_exist_ok=True)
            chunks = self._read_chunks(new_kb.kb_id, new_kb)   # 读自己的目录
            self._vector_store(new_kb).add(chunks)            # 新 collection 重向量化
            # BM25 索引与父块 chunk 直接缓存（省一次磁盘重读；目录已复制，
            # 重启后懒构建同样能从新库目录读到）
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

    # ---------- 知识库完全复制 ----------
    def copy_kb(self, src_kb_id: str, user_id: str | None) -> KnowledgeBase:
        """完全复制：chunk 目录 + 源文件归档 + 向量数据原样搬运，不过嵌入模型。

        与 rebuild 的区别：重建复用 chunk 文本但必须重新向量化（可换模型）；
        复制连向量一起拷（embedding API 零调用，秒级完成），嵌入配置原样继承。
        复制库强制为「私人」库（同重建），归属发起复制的用户。
        """
        src = self.get_kb(src_kb_id)
        new_kb = KnowledgeBase(
            kb_id=uuid.uuid4().hex[:12], name=f"{src.name}(复制)",
            scope="private", owner_user_id=user_id or src.owner_user_id,
            description=src.description,
            category_id=src.category_id,
            embedding_provider=src.embedding_provider,
            embedding_model_id=src.embedding_model_id,
            embedding_dim=src.embedding_dim,
            embedding_base_url=src.embedding_base_url,
            embedding_api_key=src.embedding_api_key,   # 密文直接继承
            embedded_model=src.embedded_model,          # 向量是拷来的，写入标注一并继承
            status="copying", source_doc_ids=list(src.source_doc_ids or []),
        )
        with SessionLocal() as db:
            db.add(new_kb)
            db.commit()
        try:
            # 磁盘数据独立成套：chunk 目录 + 源文件归档目录完整复制
            src_dir = self._chunk_dir / src_kb_id
            if src_dir.exists():
                shutil.copytree(src_dir, self._chunk_dir / new_kb.kb_id,
                                dirs_exist_ok=True)
            src_docs = self._docs_dir / src_kb_id
            if src_docs.exists():
                shutil.copytree(src_docs, self._docs_dir / new_kb.kb_id,
                                dirs_exist_ok=True)
            # 向量原样搬运（不调嵌入 API），BM25/父块缓存直接构建
            moved = self._vector_store(new_kb).copy_from(
                self._vector_store(src))
            chunks = self._read_chunks(new_kb.kb_id, new_kb)
            bm = Bm25Index()
            bm.build(chunks)
            self._bm25_cache[new_kb.kb_id] = bm
            self._chunks_cache[new_kb.kb_id] = list(chunks)
            with SessionLocal() as db:
                row = db.get(KnowledgeBase, new_kb.kb_id)
                row.status = "ready"
                db.commit()
            emit("kb_copy", {"src_kb_id": src_kb_id, "new_kb_id": new_kb.kb_id,
                             "chunks": moved, "status": "ready"})
        except Exception as e:
            logger.error("copy kb failed src=%s: %s", src_kb_id, e)
            with SessionLocal() as db:
                row = db.get(KnowledgeBase, new_kb.kb_id)
                row.status = "failed"
                db.commit()
            raise
        return self.get_kb(new_kb.kb_id)

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

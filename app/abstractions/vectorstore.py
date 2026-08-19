# 向量存储抽象
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import chromadb

from app.abstractions.embedding import EmbeddingModel


@dataclass
class ChunkRecord:
    id: str
    text: str
    payload: dict = field(default_factory=dict)


def kb_filter(scope: str, user_id: str | None) -> dict:
    """两级知识库过滤：public 全员 / private 仅本人 / 合并检索。"""
    if scope == "public":
        return {"scope": "public"}
    if scope == "private":
        return {"$and": [{"scope": "private"}, {"user_id": user_id}]}
    return {
        "$or": [
            {"scope": "public"},
            {"$and": [{"scope": "private"}, {"user_id": user_id}]},
        ]
    }


class VectorStore(ABC):
    """向量存储抽象：每个 KB 独立 collection；嵌入由上层模型显式计算后传入。"""

    def __init__(self, kb_id: str, embedding_model: EmbeddingModel):
        self.kb_id = kb_id
        self.embedding_model = embedding_model

    @property
    def dim(self) -> int:
        return self.embedding_model.dim

    @abstractmethod
    def add(self, chunks: list[ChunkRecord]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 5, scope: str = "all",
               user_id: str | None = None) -> list[dict]: ...


class ChromaVectorStore(VectorStore):
    """Chroma 实现：PersistentClient 本地持久化，collection 命名 kb_{kb_id}。

    刻意不向 Chroma 注册 embedding_function：向量全部由 self.embedding_model
    显式计算后通过 embeddings=/query_embeddings= 传入。这样集合与嵌入模型
    彻底解耦——知识库随时改嵌入配置都不会触发 Chroma 的 embedding function
    冲突校验（模型不一致由上层 embedding_mismatch 提醒）。
    """

    def __init__(self, persist_dir: str, kb_id: str, embedding_model: EmbeddingModel):
        super().__init__(kb_id, embedding_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(name=f"kb_{kb_id}")

    # Chroma 服务端单次 add 硬上限 5461 条（超出抛
    # "Batch size of N is greater than max batch size of 5461"）。
    # 大库（>5461 chunk）一次全量提交必被拒——入库/重建都走这里，
    MAX_ADD_BATCH = 16

    def add(self, chunks: list[ChunkRecord],
            on_batch: Callable[[int], None] | None = None) -> None:
        """批量写入向量；on_batch(done_count) 每批写完后回调（进度上报用）。"""
        if not chunks:
            return
        for i in range(0, len(chunks), self.MAX_ADD_BATCH):
            batch = chunks[i:i + self.MAX_ADD_BATCH]
            self._col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.payload for c in batch],
                embeddings=self.embedding_model.embed_texts([c.text for c in batch]),
            )
            if on_batch:
                on_batch(i + len(batch))

    def copy_from(self, src_store: "ChromaVectorStore") -> int:
        """把源 collection 的向量数据原样搬到本 collection（不调嵌入 API）。

        「完全复制」知识库用：分批 get(ids/documents/metadatas/embeddings)
        再 add——纯数据搬运，几万 chunk 秒级完成，远端嵌入模型零调用。
        返回搬运的 chunk 数。
        """
        total = src_store._col.count()
        if total == 0:
            return 0
        moved = 0
        for off in range(0, total, self.MAX_ADD_BATCH):
            res = src_store._col.get(
                limit=self.MAX_ADD_BATCH, offset=off,
                include=["documents", "metadatas", "embeddings"])
            if not res.get("ids"):
                break
            self._col.add(
                ids=res["ids"],
                documents=res["documents"],
                metadatas=[m or {} for m in res["metadatas"]],
                embeddings=res["embeddings"])
            moved += len(res["ids"])
        return moved

    def search(self, query: str, k: int = 5, scope: str = "all",
               user_id: str | None = None) -> list[dict]:
        n = self._col.count()
        if n == 0:
            return []
        query_vec = self.embedding_model.embed_query(query)
        res = self._col.query(
            query_embeddings=[query_vec],
            n_results=min(k, n),
            where=kb_filter(scope, user_id),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        for i, doc in enumerate(docs):
            out.append({
                "id": ids[i],
                "text": doc,
                "metadata": metas[i] or {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return out


class VectorStoreFactory:
    """按配置选择向量库实现：当前只有 chroma，预留 Milvus/Qdrant。"""

    @staticmethod
    def build(settings, kb_id: str, embedding_model: EmbeddingModel) -> VectorStore:
        if settings.vector_store_provider == "chroma":
            return ChromaVectorStore(str(settings.chroma_persist_dir), kb_id, embedding_model)
        raise RuntimeError(f"unsupported vector store provider: {settings.vector_store_provider}")
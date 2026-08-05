# 嵌入模型抽象
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class EmbeddingMeta:
    provider: str        # "local" | "openai" | "qwen" ...
    model_id: str
    dim: int
    base_url: str | None = None


class EmbeddingModel(ABC):
    """嵌入模型抽象接口：本地/云端两类实现，业务不感知来源。"""

    def __init__(self, meta: EmbeddingMeta):
        self.meta = meta

    @property
    def dim(self) -> int:
        return self.meta.dim

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    def as_chroma_function(self):
        """适配 Chroma 的 embedding_function 接口（闭包持有 self）。"""
        from chromadb.api.types import EmbeddingFunction as ChromaFn

        outer = self

        class _Fn(ChromaFn):
            def __call__(self, inputs):
                return outer.embed_texts(list(inputs))

        return _Fn()


class CloudEmbeddingModel(EmbeddingModel):
    """云端 API：api_key + base_url + model_id（OpenAI 兼容协议）。"""

    def __init__(self, meta: EmbeddingMeta, api_key: str):
        super().__init__(meta)
        self._client = OpenAI(api_key=api_key, base_url=meta.base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.meta.model_id, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class LocalEmbeddingModel(EmbeddingModel):
    """本地推理（数据不出域）：优先 sentence-transformers；未安装时回退
    chromadb 内置 ONNX MiniLM。"""

    def __init__(self, meta: EmbeddingMeta, model_name: str | None = None):
        super().__init__(meta)
        self._backend = self._load_backend(model_name)

    @staticmethod
    def _load_backend(model_name):
        try:
            from sentence_transformers import SentenceTransformer
            return ("st", SentenceTransformer(model_name or "BAAI/bge-small-zh-v1.5"))
        except Exception:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            return ("chroma", DefaultEmbeddingFunction())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        kind, backend = self._backend
        if kind == "st":
            return backend.encode(texts).tolist()
        return backend(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class EmbeddingFactory:
    """按 KB 标注的 EmbeddingMeta 构建模型。"""

    @staticmethod
    def build(meta: EmbeddingMeta, secrets: dict[str, str]) -> EmbeddingModel:
        if meta.provider == "local":
            return LocalEmbeddingModel(meta)
        api_key = secrets.get(meta.provider) or secrets.get("system")
        if not api_key:
            raise RuntimeError(f"missing api_key for embedding provider: {meta.provider}")
        return CloudEmbeddingModel(meta, api_key)
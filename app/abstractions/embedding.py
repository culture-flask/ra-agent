from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class EmbeddingMeta:
    provider: str        # "local" | "doubao" | "openai" | "qwen" ...
    model_id: str
    dim: int
    base_url: str | None = None


class EmbeddingModel(ABC):
    """嵌入模型抽象接口（架构文档 §11.1）：本地/云端两类实现，业务不感知来源。"""

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
            def __init__(self):
                pass

            def __call__(self, inputs):
                return outer.embed_texts(list(inputs))

            @staticmethod
            def name():
                """Chroma 在类级别调用 name()（官方实现同款签名）。"""
                return f"ra_{outer.meta.provider}_{outer.meta.model_id}"

            def get_config(self):
                """Chroma 序列化配置：记录所用模型，便于回溯 collection 绑定的模型。"""
                return {
                    "provider": outer.meta.provider,
                    "model_id": outer.meta.model_id,
                    "dim": outer.meta.dim,
                }

            @classmethod
            def build_from_config(cls, config):
                """Chroma 反序列化（未来版本从配置重建嵌入函数）。"""
                return cls()

        return _Fn()


class CloudEmbeddingModel(EmbeddingModel):
    """云端 API：api_key + base_url + model_id（OpenAI 兼容协议）。"""

    BATCH_SIZE = 10      # doubao 单次 input 上限 10 条（实测；各家不同，保守取 10）
    BATCH_DELAY = 5.0    # 批次间隔秒数：防 429 限流（测试密钥 RPM 低，实测 0.5s 仍触发）
    MAX_RETRIES = 10      # 429 重试次数（指数退避）

    def __init__(self, meta: EmbeddingMeta, api_key: str):
        super().__init__(meta)
        self._client = OpenAI(api_key=api_key, base_url=meta.base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """分批嵌入：一次最多 BATCH_SIZE 条，超出自动切片并间隔发送（防限流）。"""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            out.extend(self._embed_batch_with_retry(batch))
            if i + self.BATCH_SIZE < len(texts):
                import time
                time.sleep(self.BATCH_DELAY)
        return out

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """单批嵌入：429 限流时按指数退避重试，其他错误直接抛。"""
        import time

        from openai import RateLimitError

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.embeddings.create(
                    model=self.meta.model_id, input=batch)
                return [d.embedding for d in resp.data]
            except RateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)      # 退避：1s → 2s
        raise RuntimeError("unreachable")

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
    """按 KB 标注的 EmbeddingMeta 构建模型（架构文档 §11.1）。"""

    @staticmethod
    def build(meta: EmbeddingMeta, secrets: dict[str, str]) -> EmbeddingModel:
        if meta.provider == "local":
            return LocalEmbeddingModel(meta)
        api_key = secrets.get(meta.provider) or secrets.get("system")
        if not api_key:
            raise RuntimeError(f"missing api_key for embedding provider: {meta.provider}")
        return CloudEmbeddingModel(meta, api_key)

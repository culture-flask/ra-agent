from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class EmbeddingMeta:
    provider: str        # "local" | "doubao" | "openai" | "qwen" | 任意自建端点名（llama.cpp 等）
    model_id: str
    dim: int
    base_url: str | None = None
    batch_delay: float | None = None   # 批次间隔秒数；None=用实现默认（云端限流需要），自建端点置 0
    batch_size: int | None = None      # 单批条数；None=用实现默认 10，自建端点减小防打崩脆弱服务器


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

    BATCH_SIZE = 100      # doubao 单次 input 上限 10 条（实测；各家不同，保守取 10）
    BATCH_DELAY = 0    # 批次间隔秒数：防 429 限流（测试密钥 RPM 低，实测 0.5s 仍触发）
    MAX_RETRIES = 10      # 429 重试次数（指数退避）
    MAX_CONN_RETRIES = 4  # 连接抖动重试次数（0.5+1+2+4≈7.5s；服务器真挂了就快速失败，别空等 8 分钟）

    def __init__(self, meta: EmbeddingMeta, api_key: str):
        super().__init__(meta)
        self._client = OpenAI(api_key=api_key, base_url=meta.base_url)
        # 自建端点（ollama / llama.cpp / vLLM）无限流，跳过批次间隔，避免大批量入库空等
        self.batch_delay = meta.batch_delay if meta.batch_delay is not None else self.BATCH_DELAY
        # 自建端点用小批次（内存峰值低，8B 模型 + 大 n_ctx 的服务器扛不住 10 条×1000 字）
        self.batch_size = meta.batch_size or self.BATCH_SIZE

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """分批嵌入：一次最多 batch_size 条，超出自动切片并间隔发送（防限流）。"""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            out.extend(self._embed_batch_with_retry(batch))
            if self.batch_delay and i + self.batch_size < len(texts):
                import time
                time.sleep(self.batch_delay)
        return out

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """单批嵌入：429 限流 / 连接抖动（远端模型冷启动、网络瞬断）指数退避重试。"""
        import time

        from openai import APIConnectionError, APITimeoutError, RateLimitError

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.embeddings.create(
                    model=self.meta.model_id, input=batch)
                return [d.embedding for d in resp.data]
            except RateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)            # 退避：1s → 2s → 4s
            except (APIConnectionError, APITimeoutError):
                if attempt == self.MAX_CONN_RETRIES - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))    # 连接抖动：0.5s → 1s → 2s → 4s
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
            if meta.provider in secrets:
                # 配置过的云端 provider 必须有 key（fail fast，避免入库时才 401）
                raise RuntimeError(f"missing api_key for embedding provider: {meta.provider}")
            api_key = "sk-self-hosted"   # 自建端点（llama.cpp 等）免鉴权，占位即可
        return CloudEmbeddingModel(meta, api_key)

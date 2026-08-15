# 将yaml/环境变量统一映射成带类型的Python对象

import yaml
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # 从 ./env读配置，未知字段忽略
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ra-agent"
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = Path("./data")
    database_url: str = "postgresql+psycopg://ra:ra@localhost:5432/ra_agent"
    redis_url: str = "redis://localhost:6379/0"
    vector_store_provider: str = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma")
    max_logs: int = 1000
    jwt_secret: str = "dev-secret-change-me-to-32+bytes!!"
    llm_system_default: dict = Field(default_factory=dict)
    llm_api_key: str = ""
    llm_retry_max_retries: int = 3          # LLM 调用失败重试次数（指数退避）
    llm_retry_base_delay: float = 1.0       # 首次重试基础等待秒数，之后 2 倍递增
    llm_context_window: int = 32768         # 生效模型上下文窗口（token），80% 触发自动压缩
    retrieval_mode: str = "hybrid"          # 检索模式：vector（纯向量）| hybrid（向量+BM25）
    retrieval_per_kb_k: int = 3             # 对话 RAG：每个知识库检索几条
    retrieval_total_k: int = 5              # 对话 RAG：所有库合并后总共取几条
    embedding_default_provider: str = "doubao"
    embedding_default_model: str = "doubao-embedding-vision"
    embedding_cloud: dict = Field(default_factory=dict)
    embedding_local_default_dim: int = 384
    embedding_api_key: str = ""
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = "localhost,127.0.0.1"
    mcp_servers: dict = Field(default_factory=dict)
    llm_providers: dict = Field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        """加载配置。优先级： 环境变量/.env > settings.ymal > 类默认值

        命名约定：yaml按主题嵌套、代码字段扁平化，靠下面这个映射表桥接。
        规则——新增配置项必须三处同步：yaml键、Settings字段、映射表。
        且字段名与yaml的”叶子键”要完全一致。
        """
        merged = cls()                   # 先让 pydantic-settings 应用环境变量与 .env
        yaml_path = BASE_DIR / "config" / "settings.yaml"
        if yaml_path.exists():
            raw = yaml.safe_load(yaml_path.read_text())
            yaml_values = {
                "app_name" : raw["app"]["name"],
                "host" : raw["app"]["host"],
                "port" : raw["app"]["port"],
                "data_dir" : raw["app"]["data_dir"],
                "database_url" : raw["database"]["url"],
                "redis_url": raw["redis"]["url"],
                "vector_store_provider": raw["vector_store"]["provider"],
                "chroma_persist_dir": raw["vector_store"]["chroma"]["persist_dir"],
                "max_logs": raw["tracing"]["max_logs"],
                "llm_system_default": raw["llm"]["system_default"],
                "llm_retry_max_retries": raw.get("llm", {}).get("retry", {}).get("max_retries", 3),
                "llm_retry_base_delay": raw.get("llm", {}).get("retry", {}).get("base_delay", 1.0),
                "llm_context_window": raw.get("llm", {}).get("context_window", 32768),
                "retrieval_mode": raw.get("retrieval", {}).get("mode", "hybrid"),
                "retrieval_per_kb_k": raw.get("retrieval", {}).get("per_kb_k", 3),
                "retrieval_total_k": raw.get("retrieval", {}).get("total_k", 5),
                "embedding_default_provider": raw["embedding"]["default_provider"],
                "embedding_default_model": raw["embedding"]["default_model"],
                "embedding_cloud": raw["embedding"]["cloud"],
                "embedding_local_default_dim": raw["embedding"]["local"]["default_dim"],
                "mcp_servers": raw.get("mcp_servers", {}),
                "llm_providers": raw.get("llm", {}).get("providers", {}),   # 容错:缺段用空目录
            }
            for key, value in yaml_values.items():
                if key not in merged.model_fields_set:
                    setattr(merged, key, value)        # 环境已设置的字段，yaml不再覆盖
        return merged

settings = Settings.load()
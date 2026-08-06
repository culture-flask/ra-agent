# LLM 工厂与服务
from dataclasses import dataclass

from langchain_openai import ChatOpenAI


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model_id: str
    api_key: str | None = None


class LLMFactory:
    """OpenAI 协议兼容多家厂商：base_url 接入 qwen/deepseek/ollama...。"""

    @staticmethod
    def build(cfg: LLMConfig) -> ChatOpenAI:
        return ChatOpenAI(
            model=cfg.model_id,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=0.3,
            streaming=True,
            request_timeout=120,
        )


class LLMService:
    """按用户级配置构建 ChatModel，未配置回退系统默认。"""

    def __init__(self, system_default: dict, system_api_key: str,
                 user_configs: dict[str, LLMConfig] | None = None):
        self._system = LLMConfig(
            provider=system_default.get("provider", "sensenova"),
            base_url=system_default.get("base_url", "https://token.sensenova.cn/v1"),
            model_id=system_default.get("model_id", "deepseek-v4-flash"),
            api_key=system_api_key,
        )
        self._user_configs = user_configs or {}

    def get_chat_model(self, user_id: str) -> ChatOpenAI:
        cfg = self._user_configs.get(user_id) or self._system
        return LLMFactory.build(cfg)

    def set_user_config(self, user_id: str, cfg: LLMConfig) -> None:
        self._user_configs[user_id] = cfg
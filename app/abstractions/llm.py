"""LLM 服务（Day 9 升级）：用户级配置落库（user_llm_config 表）+ 加密 + 掩码。

第 3 天的内存 dict 换成第 1 天建好的 user_llm_config 表：
- get_chat_model：该用户默认配置（解密）→ 工厂构建；未配置回退系统默认；
- set_user_config：api_key 加密落库，is_default 互斥；
- list_configs：只回显掩码（永不下发明文）。
"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from sqlalchemy import select

from app.core.crypto import SecretCrypto, mask_secret
from app.core.db import SessionLocal
from app.models import UserLLMConfig


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model_id: str
    api_key: str | None = None


class LLMFactory:
    """OpenAI 协议兼容多家厂商：base_url 可定制即接入 qwen/deepseek/ollama...（§12.2）。"""

    @staticmethod
    def build(cfg: LLMConfig) -> ChatOpenAI:
        return ChatOpenAI(
            model=cfg.model_id,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=0.3,
            streaming=True,
            request_timeout=60,   # 连续 60s 无数据视为卡死，抛错而不是无限等
        )


class LLMService:
    """按用户级配置构建 ChatModel，未配置回退系统默认（§12.2）。"""

    def __init__(self, system_default: dict, system_api_key: str,
                 crypto: SecretCrypto):
        self._system = LLMConfig(
            provider=system_default.get("provider", "sensenova"),
            base_url=system_default.get("base_url", "https://token.sensenova.cn/v1"),
            model_id=system_default.get("model_id", "deepseek-v4-flash"),
            api_key=system_api_key,
        )
        self._crypto = crypto

    # ---------- 查询 ----------
    def get_user_config(self, user_id: str) -> LLMConfig | None:
        """该用户的生效配置：默认标记优先，否则取最新一条（api_key 解密）。"""
        with SessionLocal() as db:
            row = db.scalar(select(UserLLMConfig).where(
                UserLLMConfig.user_id == user_id,
                UserLLMConfig.is_default == True))          # noqa: E712
            if row is None:
                row = db.scalar(select(UserLLMConfig).where(
                    UserLLMConfig.user_id == user_id)
                    .order_by(UserLLMConfig.created_at.desc()))
        if row is None:
            return None
        return LLMConfig(provider=row.provider, base_url=row.base_url,
                         model_id=row.model_id,
                         api_key=self._crypto.decrypt(row.api_key))

    def get_chat_model(self, user_id: str) -> ChatOpenAI:
        cfg = self.get_user_config(user_id) or self._system
        return LLMFactory.build(cfg)

    # ---------- 写入 ----------
    def set_user_config(self, user_id: str, provider: str, base_url: str,
                        model_id: str, api_key: str,
                        is_default: bool = False) -> str:
        """保存用户配置：api_key 加密落库；设默认时先清掉其他默认。"""
        if is_default:
            with SessionLocal() as db:
                rows = db.scalars(select(UserLLMConfig).where(
                    UserLLMConfig.user_id == user_id,
                    UserLLMConfig.is_default == True)).all()   # noqa: E712
                for r in rows:
                    r.is_default = False
                db.commit()
        with SessionLocal() as db:
            row = UserLLMConfig(
                user_id=user_id, provider=provider,
                api_key=self._crypto.encrypt(api_key),   # 加密落库
                base_url=base_url, model_id=model_id,
                is_default=is_default,
            )
            db.add(row)
            db.commit()
            return row.id

    def list_configs(self, user_id: str) -> list[dict]:
        """该用户的配置列表：api_key 只回显掩码（用户级隔离，永不明文）。"""
        with SessionLocal() as db:
            rows = db.scalars(select(UserLLMConfig).where(
                UserLLMConfig.user_id == user_id)
                .order_by(UserLLMConfig.created_at.desc())).all()
        return [
            {"id": r.id, "provider": r.provider, "base_url": r.base_url,
             "model_id": r.model_id, "is_default": r.is_default,
             "api_key_masked": mask_secret(self._crypto.decrypt(r.api_key))}
            for r in rows
        ]

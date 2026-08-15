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

    def get_config(self, user_id: str, config_id: str) -> LLMConfig | None:
        """读取某条已保存配置（api_key 解密），仅本人可见，不存在返回 None。"""
        with SessionLocal() as db:
            row = db.get(UserLLMConfig, config_id)
            if row is None or row.user_id != user_id:
                return None
        return LLMConfig(provider=row.provider, base_url=row.base_url,
                         model_id=row.model_id,
                         api_key=self._crypto.decrypt(row.api_key))

    # ---------- 写入 ----------
    def _clear_defaults(self, db, user_id: str,
                         exclude_id: str | None = None) -> None:
        """把该用户所有默认标记清掉（可排除某条，用于把它切成默认）。"""
        rows = db.scalars(select(UserLLMConfig).where(
            UserLLMConfig.user_id == user_id,
            UserLLMConfig.is_default == True)).all()   # noqa: E712
        for r in rows:
            if r.id != exclude_id:
                r.is_default = False

    def set_user_config(self, user_id: str, provider: str, base_url: str,
                        model_id: str, api_key: str,
                        is_default: bool = False) -> str:
        """保存用户配置：api_key 加密落库；设默认时先清掉其他默认。"""
        with SessionLocal() as db:
            if is_default:
                self._clear_defaults(db, user_id)
            row = UserLLMConfig(
                user_id=user_id, provider=provider,
                api_key=self._crypto.encrypt(api_key),   # 加密落库
                base_url=base_url, model_id=model_id,
                is_default=is_default,
            )
            db.add(row)
            db.commit()
            return row.id

    def set_default_config(self, user_id: str, config_id: str) -> bool:
        """把某条已保存配置切换为默认模型（互斥：先清该用户其他默认）。

        仅允许操作本人的配置；config 不存在或属于他人 → 返回 False。
        """
        with SessionLocal() as db:
            target = db.get(UserLLMConfig, config_id)
            if target is None or target.user_id != user_id:
                return False
            self._clear_defaults(db, user_id, exclude_id=config_id)
            target.is_default = True
            db.commit()
            return True

    def update_config(self, user_id: str, config_id: str,
                      provider: str | None = None,
                      base_url: str | None = None,
                      model_id: str | None = None,
                      api_key: str | None = None) -> bool:
        """更新已保存配置（切换模型用）：只传要改的字段，None 表示保持原值。

        仅本人配置可改；is_default 不受影响（切换后仍是默认/非默认）。
        """
        with SessionLocal() as db:
            row = db.get(UserLLMConfig, config_id)
            if row is None or row.user_id != user_id:
                return False
            if provider is not None:
                row.provider = provider
            if base_url is not None:
                row.base_url = base_url
            if model_id is not None:
                row.model_id = model_id
            if api_key is not None:
                row.api_key = self._crypto.encrypt(api_key)
            db.commit()
            return True

    def delete_config(self, user_id: str, config_id: str) -> bool:
        """删除一条配置：仅允许删本人的，返回是否删除成功。"""
        with SessionLocal() as db:
            row = db.get(UserLLMConfig, config_id)
            if row is None or row.user_id != user_id:
                return False
            db.delete(row)
            db.commit()
            return True

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

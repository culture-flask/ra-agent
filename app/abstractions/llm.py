"""LLM 服务（Day 9 升级）：用户级配置落库（user_llm_config 表）+ 加密 + 掩码。

第 3 天的内存 dict 换成第 1 天建好的 user_llm_config 表：
- get_chat_model：该用户默认配置（解密）→ 工厂构建；未配置回退系统默认；
- set_user_config：api_key 加密落库，is_default 互斥；
- list_configs：只回显掩码（永不下发明文）。
- RetryableChatModel：统一给 LLM 调用包一层指数退避重试（限流/5xx/网络错误）。
"""

import asyncio
from dataclasses import dataclass

import httpx
from langchain_openai import ChatOpenAI
from sqlalchemy import select

from app.core.crypto import SecretCrypto, mask_secret
from app.core.db import SessionLocal
from app.core.logging import get_logger
from app.models import UserLLMConfig

logger = get_logger("llm_retry")


_QUOTA_MARKERS = (
    "insufficient_quota", "quota exceeded", "quota exhausted",
    "workspace allocated quota", "out of quota", "no quota",
)


def _is_quota_exhausted(exc: BaseException) -> bool:
    """429 里区分"账户/工作区额度用尽"（重试无意义）与"限流"（可重试）。

    glm/doubao 等平台额度用尽时返回的 message 形如
    "Workspace allocated quota exceeded, please increase your quota limit"，
    属于计费问题，退避重试多少次都一样，必须让用户去平台提升额度。
    """
    text = str(exc)
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            text += " " + (resp.text or "")
        except Exception:
            pass
    lowered = text.lower()
    return any(m in lowered for m in _QUOTA_MARKERS)


def _is_rate_limited(exc: BaseException) -> bool:
    """真正的限流 429（rpm/tpm 超限），区别于额度用尽。"""
    return getattr(exc, "status_code", None) == 429 and not _is_quota_exhausted(exc)


def _is_retryable(exc: BaseException) -> bool:
    """是否值得重试：限流(429)/服务端错误(5xx)/传输层(超时、断连、读写出错)。

    - 429 限流（rpm/tpm）→ 重试
    - 429 额度用尽（insufficient_quota）→ 重试无意义，直接抛
    - 参数错误、鉴权失败等 4xx 重试无意义，直接抛
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return not _is_quota_exhausted(exc)
        return status >= 500
    if isinstance(exc, httpx.TransportError):
        return True
    # 兜底：异常名含超时/连接/网络/协议字样（部分 SDK 会包一层自定义异常）
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connect", "network", "protocol"))


class RetryableChatModel:
    """给 ChatModel 包一层指数退避重试（每次重试记日志，可观测）。

    - 重试条件：429 限流 / 5xx / 网络超时与断连；其余异常直接抛
      （429 额度用尽 insufficient_quota 除外——重试无意义，直接抛）
    - 退避：1s → 2s → 4s → 8s → 16s → 32s → 64s → 128s → 256s（共 9 次）
    - astream：整轮重试——流中断后从开头重新完整生成
      （已推给前端的半截内容会短暂重复，属预期行为）
    """

    def __init__(self, model, max_retries: int = 9, base_delay: float = 1.0,
                 label: str = ""):
        self._model = model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._label = label or getattr(model, "model_name", None) or "chat"

    @property
    def model_name(self) -> str:
        """透传底层模型名（追踪/日志用）。"""
        return self._label

    def bind_tools(self, schemas):
        """绑定 MCP 工具后仍保留重试能力（generate ⇄ tool 循环的第二次生成）。"""
        bound = self._model.bind_tools(schemas)
        return RetryableChatModel(bound, self._max_retries, self._base_delay,
                                  self._label)

    def _backoff(self, attempt: int) -> float:
        """指数退避：1s → 2s → 4s → 8s → 16s → 32s → 64s → 128s → 256s。"""
        return self._base_delay * (2 ** (attempt - 1))

    async def _retry(self, call):
        attempt = 0
        while True:
            try:
                return await call()
            except Exception as e:
                attempt += 1
                if attempt > self._max_retries or not _is_retryable(e):
                    raise
                delay = self._backoff(attempt)
                logger.warning(
                    "llm call failed (attempt %d/%d) model=%s: %s; retry in %.1fs",
                    attempt, self._max_retries, self._label, e, delay)
                await asyncio.sleep(delay)

    async def ainvoke(self, messages, **kwargs):
        return await self._retry(lambda: self._model.ainvoke(messages, **kwargs))

    async def astream(self, messages, **kwargs):
        attempt = 0
        while True:
            try:
                async for chunk in self._model.astream(messages, **kwargs):
                    yield chunk
                return
            except Exception as e:
                attempt += 1
                if attempt > self._max_retries or not _is_retryable(e):
                    raise
                delay = self._backoff(attempt)
                logger.warning(
                    "llm stream broken (attempt %d/%d) model=%s: %s; retry in %.1fs",
                    attempt, self._max_retries, self._label, e, delay)
                await asyncio.sleep(delay)


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
            max_retries=0,        # 关闭 SDK 内部静默重试，统一走外层 RetryableChatModel（可观测）
        )

class LLMService:
    """按用户级配置构建 ChatModel，未配置回退系统默认（§12.2）。"""

    def __init__(self, system_default: dict, system_api_key: str,
                 crypto: SecretCrypto, retry_max_retries: int = 3,
                 retry_base_delay: float = 1.0):
        self._system = LLMConfig(
            provider=system_default.get("provider", "sensenova"),
            base_url=system_default.get("base_url", "https://token.sensenova.cn/v1"),
            model_id=system_default.get("model_id", "deepseek-v4-flash"),
            api_key=system_api_key,
        )
        self._crypto = crypto
        self._retry_max = retry_max_retries
        self._retry_delay = retry_base_delay

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

    def get_chat_model(self, user_id: str) -> RetryableChatModel:
        cfg = self.get_user_config(user_id) or self._system
        model = LLMFactory.build(cfg)
        return RetryableChatModel(model, max_retries=self._retry_max,
                                  base_delay=self._retry_delay,
                                  label=cfg.model_id)

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

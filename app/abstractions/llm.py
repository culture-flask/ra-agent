"""LLM 服务：用户级配置落库（user_llm_config 表）+ 加密 + 掩码。

- get_chat_model：该用户默认配置（解密）→ 工厂构建；未配置回退系统默认；
- set_user_config：api_key 加密落库，is_default 互斥；
- list_configs：只回显掩码（永不下发明文）。
- RetryableChatModel：统一给 LLM 调用包一层指数退避重试（限流/5xx/网络错误）。
"""

import asyncio
from dataclasses import dataclass, replace

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
    属于计费问题，退避重试多少次都一样。
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

    def __init__(self, model, max_retries: int = 10, base_delay: float = 1.0,
                 label: str = ""):
        self._model = model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._label = label or getattr(model, "model_name", None) or "chat"

    @property
    def model_name(self) -> str:
        """透传底层模型名（追踪/日志用）。"""
        return self._label

    @property
    def temperature(self) -> float:
        """透传底层生成温度（调试/测试用）。"""
        return getattr(self._model, "temperature", DEFAULT_TEMPERATURE)

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


# 各平台 /models 响应里常见的上下文窗口字段名（按优先级尝试）
_CONTEXT_WINDOW_FIELDS = (
    "context_length",        # OpenRouter 等
    "context_window",
    "max_model_len",         # vLLM / llama.cpp 风格
    "max_context_length",
    "max_sequence_length",
    "model_max_length",
    "context",
)


def extract_context_window(item: dict) -> int | None:
    """从模型元数据里提取上下文窗口（token），没有则返回 None。"""
    for field in _CONTEXT_WINDOW_FIELDS:
        v = item.get(field)
        if isinstance(v, int) and v > 0:
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return None


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model_id: str
    api_key: str | None = None
    context_window: int | None = None   # 显式指定时优先于探测/默认
    temperature: float | None = None    # None = 使用默认温度 0.3（按轮次可由用户覆盖）


# 默认生成温度
DEFAULT_TEMPERATURE = 0.3


class LLMFactory:
    """OpenAI 协议兼容多家厂商：base_url 可定制即接入 qwen/deepseek/ollama...。"""

    @staticmethod
    def build(cfg: LLMConfig) -> ChatOpenAI:
        return ChatOpenAI(
            model=cfg.model_id,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=(cfg.temperature if cfg.temperature is not None
                         else DEFAULT_TEMPERATURE),
            streaming=True,
            stream_usage=True,  # 流式响应末块携带真实 token 用量（上下文占用展示用）
            request_timeout=60,   # 连续 60s 无数据视为卡死，抛错而不是无限等
            max_retries=0,        # 关闭 SDK 内部静默重试，统一走外层 RetryableChatModel（可观测）
        )

class LLMService:
    """按用户级配置构建 ChatModel，未配置回退系统默认。"""

    def __init__(self, system_default: dict, system_api_key: str,
                 crypto: SecretCrypto, retry_max_retries: int = 10,
                 retry_base_delay: float = 1.0,
                 context_window_default: int = 256000):
        self._system = LLMConfig(
            provider=system_default.get("provider", "sensenova"),
            base_url=system_default.get("base_url", "https://token.sensenova.cn/v1"),
            model_id=system_default.get("model_id", "deepseek-v4-flash"),
            api_key=system_api_key,
        )
        self._crypto = crypto
        self._retry_max = retry_max_retries
        self._retry_delay = retry_base_delay
        self._window_default = context_window_default
        self._probe_cache: dict = {}      # (provider, base_url, model_id) -> int | None

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
                         context_window=row.context_window,
                         api_key=self._crypto.decrypt(row.api_key))

    def get_chat_model(self, user_id: str,
                       temperature: float | None = None) -> RetryableChatModel:
        """构建用户生效模型；temperature 非 None 时覆盖（按轮次可调，0 也有效）。"""
        cfg = self.get_user_config(user_id) or self._system
        if temperature is not None:
            temperature = max(-2.0, min(2.0, float(temperature)))   # 范围约束 -2~2
            cfg = replace(cfg, temperature=temperature)
        model = LLMFactory.build(cfg)
        return RetryableChatModel(model, max_retries=self._retry_max,
                                  base_delay=self._retry_delay,
                                  label=cfg.model_id)

    # ---------- 上下文窗口 ----------
    def context_window_for(self, user_id: str) -> int:
        """当前生效模型的上下文窗口（token）。

        优先级：用户显式设置 > /models 响应探测（缓存） > 默认值。
        """
        cfg = self.get_user_config(user_id) or self._system
        if cfg.context_window:
            return cfg.context_window
        probed = self._probe_context_window(cfg)
        return probed or self._window_default

    def _probe_context_window(self, cfg: LLMConfig) -> int | None:
        """从 {base_url}/models 响应里读当前模型的窗口字段（OpenRouter 等平台提供）。

        请求失败/字段缺失 → None（回退默认值）。结果按配置键缓存，
        避免每次对话都发请求；失败也缓存 None（探测只在配置变化后重试）。
        """
        key = (cfg.provider, cfg.base_url, cfg.model_id)
        if key in self._probe_cache:
            return self._probe_cache[key]
        result = None
        try:
            resp = httpx.get(
                f"{cfg.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {cfg.api_key or ''}"},
                timeout=5)
            resp.raise_for_status()
            for item in resp.json().get("data", []) or []:
                if item.get("id") == cfg.model_id:
                    result = extract_context_window(item)
                    break
        except Exception as e:
            logger.debug("context window probe failed %s: %s", cfg.model_id, e)
        self._probe_cache[key] = result
        return result

    def get_config(self, user_id: str, config_id: str) -> LLMConfig | None:
        """读取某条已保存配置（api_key 解密），仅本人可见，不存在返回 None。"""
        with SessionLocal() as db:
            row = db.get(UserLLMConfig, config_id)
            if row is None or row.user_id != user_id:
                return None
        return LLMConfig(provider=row.provider, base_url=row.base_url,
                         model_id=row.model_id,
                         context_window=row.context_window,
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
                        is_default: bool = False,
                        context_window: int | None = None) -> str:
        """保存用户配置：api_key 加密落库；设默认时先清掉其他默认。

        context_window 为用户显式指定的上下文窗口（token），None = 自动。
        """
        with SessionLocal() as db:
            if is_default:
                self._clear_defaults(db, user_id)
            row = UserLLMConfig(
                user_id=user_id, provider=provider,
                api_key=self._crypto.encrypt(api_key),   # 加密落库
                base_url=base_url, model_id=model_id,
                is_default=is_default,
                context_window=int(context_window) if context_window else None,
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
                      api_key: str | None = None,
                      context_window: int | None = None) -> bool:
        """更新已保存配置（切换模型用）：只传要改的字段，None 表示保持原值。

        仅本人配置可改；is_default 不受影响（切换后仍是默认/非默认）。
        context_window 例外约定：0 表示清除（恢复自动探测/兜底），正数为显式设置。
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
            if context_window is not None:
                row.context_window = int(context_window) or None
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
             "context_window": r.context_window,
             "api_key_masked": mask_secret(self._crypto.decrypt(r.api_key))}
            for r in rows
        ]

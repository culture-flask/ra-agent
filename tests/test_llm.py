from app.abstractions.llm import (
    LLMConfig, LLMService,
    _is_quota_exhausted, _is_retryable, _is_rate_limited,
)
from app.core.crypto import SecretCrypto
from app.settings import Settings


def _make_service():
    s = Settings.load()
    return LLMService(system_default=s.llm_system_default,
                      system_api_key=s.llm_api_key,
                      crypto=SecretCrypto(s.jwt_secret))


def _ensure_user(user_id: str):
    """user_llm_config.user_id 有外键，写配置前先建用户。"""
    from app.core.db import SessionLocal
    from app.models import User
    with SessionLocal() as db:
        if not db.get(User, user_id):
            db.add(User(id=user_id, username=user_id, password_hash="x"))
            db.commit()


def test_system_default_fallback():
    """未配置用户 → 回退系统默认。"""
    svc = _make_service()
    m = svc.get_chat_model("nobody-user")
    assert m.model_name == "deepseek-v4-flash"


def test_set_and_get_config_roundtrip():
    """保存（加密）→ 读取（解密）往返一致，DB 里是密文。"""
    from app.core.db import SessionLocal
    from app.models import User, UserLLMConfig

    with SessionLocal() as db:
        if not db.get(User, "llm-u1"):
            db.add(User(id="llm-u1", username="llm-u1", password_hash="x"))
            db.commit()

    svc = _make_service()
    svc.set_user_config("llm-u1", "openai", "https://api.openai.com/v1",
                        "gpt-4o-mini", "sk-test-abcdef123456")
    cfg = svc.get_user_config("llm-u1")
    assert cfg.api_key == "sk-test-abcdef123456"
    assert cfg.model_id == "gpt-4o-mini"

    with SessionLocal() as db:
        row = db.query(UserLLMConfig).first()
        assert "sk-test" not in row.api_key          # 落库是密文
        assert len(row.api_key) > len("sk-test-abcdef123456")


def test_is_default_exclusive():
    """设新默认 → 旧默认被取消（互斥）。"""
    _ensure_user("llm-u1")
    svc = _make_service()
    svc.set_user_config("llm-u1", "openai", "https://api.openai.com/v1",
                        "gpt-4o-mini", "k1", is_default=True)
    svc.set_user_config("llm-u1", "qwen", "https://qwen/v1",
                        "qwen-plus", "k2", is_default=True)
    cfg = svc.get_user_config("llm-u1")
    assert cfg.provider == "qwen"                    # 新默认生效


def test_masked_listing():
    """列表只回掩码，永不明文。"""
    _ensure_user("llm-u1")
    svc = _make_service()
    svc.set_user_config("llm-u1", "openai", "https://api.openai.com/v1",
                        "gpt-4o-mini", "sk-very-secret-key-9999")
    cfgs = svc.list_configs("llm-u1")
    assert "very-secret" not in str(cfgs)
    assert "sk-v...9999" in str(cfgs)


def test_crypto_roundtrip_and_mask():
    """加解密往返 + 掩码格式。"""
    from app.core.crypto import mask_secret
    c = SecretCrypto("test-master-secret")
    ct = c.encrypt("sk-abcdefgh1234")
    assert ct != "sk-abcdefgh1234"
    assert c.decrypt(ct) == "sk-abcdefgh1234"
    assert mask_secret("sk-abcdefgh1234") == "sk-a...1234"
    assert mask_secret("short") == "***"


class _Fake429:
    """模拟 openai 风格的 429 异常。"""

    def __init__(self, body_text: str, status_code: int = 429):
        self.status_code = status_code
        self.body = body_text
        self.response = type("R", (), {"text": body_text, "headers": {}})()


def test_429_quota_exhausted_not_retryable():
    """额度用尽（insufficient_quota）→ 重试无意义，不重试。"""
    exc = _Fake429('{"error": {"message": "Workspace allocated quota exceeded, '
                   'please increase your quota limit.", "code": "insufficient_quota"}}')
    assert _is_quota_exhausted(exc)
    assert not _is_rate_limited(exc)
    assert not _is_retryable(exc)


def test_429_rpm_exhausted_is_retryable():
    """真正的限流（rpm exhausted）→ 可重试。"""
    exc = _Fake429('{"error": {"message": "rpm exhausted", "code": "8"}}')
    assert not _is_quota_exhausted(exc)
    assert _is_rate_limited(exc)
    assert _is_retryable(exc)


def test_5xx_and_network_retryable_4xx_not():
    """5xx/网络错误可重试；普通 4xx 不重试。"""
    assert _is_retryable(_Fake429("", 500))
    assert _is_retryable(_Fake429("", 502))
    assert not _is_retryable(_Fake429("", 400))
    assert not _is_retryable(_Fake429("", 401))

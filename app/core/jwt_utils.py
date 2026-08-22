# JWT 工具
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from app.settings import settings

ALGORITHM = "HS256"

# 时间声明校验容差（秒）：吸收签发方与验证方时钟的小幅偏差。
# 实测环境中 NTP 回跳会让"刚签发的 token"带着未来的 iat 到达验证端，
# PyJWT 直接抛 Not yet valid (iat) → 偶发 401。分布式系统标准做法是
# 给时间类声明（exp/iat/nbf）留 leeway，而不是要求全网时钟零误差。
CLOCK_LEEWAY_SECONDS = 30


def create_access_token(user_id: str, expires_minutes: int = 60 * 24 * 14) -> str:
    """签发 JWT。payload 含 user_id 与过期时间。"""
    payload = {
        "sub": user_id,                                    # subject = 用户 id
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(timezone.utc),                 # 签发时间
        "jti": uuid.uuid4().hex,                           # token 唯一 id（用于吊销）
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """验签并解出 payload。过期或被篡改会抛 jwt.PyJWTError。

    leeway：对 exp/iat/nbf 统一放宽 CLOCK_LEEWAY_SECONDS——
    过期判定同样受益（时钟前跳时刚过期的 token 不至于立刻拒），
    安全边界由 14 天的长有效期主导，30 秒容差不构成风险。
    """
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM],
                      leeway=CLOCK_LEEWAY_SECONDS)
# JWT 工具
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from app.settings import settings

ALGORITHM = "HS256"


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
    """验签并解出 payload。过期或被篡改会抛 jwt.PyJWTError。"""
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
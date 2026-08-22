# 依赖注入
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.jwt_utils import decode_token
from app.core.logging import get_logger
from app.models import User

logger = get_logger("auth")


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """从 Bearer token 解析当前用户；缺失或无效一律 401。

    （Header(...) 强制形态会把缺头变成 422 参数错误，语义不对——
    鉴权失败就该是 401，故改为可选头 + 显式判定。）
    """
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("401 bad header: %r",
                       authorization[:24] if authorization else None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="invalid auth header")
    token = authorization[len("Bearer "):]
    try:
        payload = decode_token(token)
    except Exception as e:
        logger.warning("401 decode failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="token invalid or expired")
    user = db.get(User, payload["sub"])
    if not user:
        logger.warning("401 user not found: sub=%r", payload.get("sub"))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="user not found")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """要求当前用户是 admin，否则 403。"""
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="admin required")
    return user
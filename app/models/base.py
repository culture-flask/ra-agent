# 所有表的公共基类

from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    """所有 ORM 模型的基类： SQLALchemy 用它收集所有表定义。"""
# 数据库连接与会话
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.settings import Settings

settings = Settings.load()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI 依赖：每个请求一个会话，用毕必关。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
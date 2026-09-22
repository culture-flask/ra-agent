# 数据库连接与会话
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.settings import Settings

settings = Settings.load()

# 全局 Engine：持有连接池与方言，进程内共用（本身不执行 SQL）
# pool_pre_ping：借出连接前先探活，失效连接自动重建，避免"连接已被服务端关闭"的偶发报错
engine = create_engine(settings.database_url, pool_pre_ping=True)
# 会话工厂：SessionLocal() 产出一个数据库会话，是 ORM 操作的入口
# autoflush=False：查询前不自动 flush，写库时机由代码显式控制
# expire_on_commit=False：commit 后对象属性不过期，避免再访问时多查一次库
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI 依赖：每个请求一个会话，用毕必关。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
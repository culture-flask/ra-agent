"""测试公共设施：保证每个测试互不影响、可重复运行。

问题：注册类测试（test_auth.py）共用同一个数据库，第一次跑注册了 alice，
第二次再跑就 409（用户已存在）。解法：每个测试前清空相关表。
"""

import os
import tempfile

# 关键：测试必须离线、确定性——强制本地嵌入 + 临时数据目录。
# 必须在 import app.main 之前设置（Settings.load() 在 lifespan 里读环境变量）。
# 数据库也必须指向专用测试库，绝不能清空开发库（否则跑一次测试就删一次真实知识库）：
#   docker exec ra-postgres psql -U ra -d ra_agent -c "CREATE DATABASE ra_agent_test OWNER ra"
#   DATABASE_URL=postgresql+psycopg://ra:ra@localhost:5432/ra_agent_test .venv/bin/alembic upgrade head
os.environ["DATABASE_URL"] = "postgresql+psycopg://ra:ra@localhost:5432/ra_agent_test"
os.environ["EMBEDDING_DEFAULT_PROVIDER"] = "local"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="ra-test-data-")
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="ra-test-chroma-")

import pytest
from sqlalchemy import text

from app.core.db import engine


@pytest.fixture(autouse=True)
def clean_db():
    """每个测试前清空 users/kbs 表，保证注册/建库类测试可重复运行。

    autouse=True：所有测试自动使用，无需在测试函数里显式声明。
    这是"测试可重复"的最简单方案；更完整的"每个测试事务回滚"技术在第 7 天讲。
    """
    with engine.begin() as conn:          # 事务：DELETE 执行后自动提交
        conn.execute(text("DELETE FROM user_llm_config"))
        conn.execute(text("DELETE FROM memories"))  # 子表链按外键方向删
        conn.execute(text("DELETE FROM kbs"))       # 再删子表(有外键指向 users)
        conn.execute(text("DELETE FROM users"))
        # LangGraph 检查点：不清理会让上一轮测试的 retrievals 等状态
        # 按 thread_id 泄漏进下一轮测试（会话状态必须每轮重置）
        conn.execute(text("TRUNCATE checkpoint_writes, checkpoint_blobs, checkpoints CASCADE"))
    yield

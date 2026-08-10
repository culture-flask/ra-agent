"""测试公共设施：保证每个测试互不影响、可重复运行。

问题：注册类测试（test_auth.py）共用同一个数据库，第一次跑注册了 alice，
第二次再跑就 409（用户已存在）。解法：每个测试前清空相关表。
"""

import pytest
from sqlalchemy import text

from app.core.db import engine


@pytest.fixture(autouse=True)
def clean_users():
    """每个测试前清空 users 表，保证注册/登录类测试可重复运行。

    autouse=True：所有测试自动使用，无需在测试函数里显式声明。
    这是"测试可重复"的最简单方案；更完整的"每个测试事务回滚"技术在第 7 天讲。
    """
    with engine.begin() as conn:          # 事务：DELETE 执行后自动提交
        conn.execute(text("DELETE FROM kbs"))       # 先删子表(有外键指向 users)
        conn.execute(text("DELETE FROM users"))
    yield

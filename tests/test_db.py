from sqlalchemy import text

from app.core.db import engine


def test_db_connect():
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_tables_exist():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )).scalars().all()
        assert {"users", "kbs", "tool_call_log"}.issubset(rows)
        assert "sessions" not in rows      # 死表已随 UserSession 一起清理
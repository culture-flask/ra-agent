"""drop sessions table

UserSession（sessions 表）是登录会话设计的遗留：认证最终走纯 JWT 无状态
方案（见 api/auth.py），该表全仓库零引用（无任何读写），随 UserSession
模型一并清理。downgrade 按 init 迁移原样重建（空表）。

Revision ID: a3b8c1d5e9f2
Revises: f5a6b7c8d9e0
Create Date: 2026-09-17 10:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3b8c1d5e9f2'
down_revision: Union[str, Sequence[str], None] = 'f5a6b7c8d9e0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # IF EXISTS：容忍未跑过 init 迁移的 drifted 开发库（该表已无人使用）
    op.execute("DROP TABLE IF EXISTS sessions")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table('sessions',
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_active_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('session_id')
    )
    op.create_index(op.f('ix_sessions_user_id'), 'sessions', ['user_id'], unique=False)
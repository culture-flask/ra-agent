"""add kbs.description

知识库介绍：新建必填，随 LLM 选库目录一并传入路由提示词，
帮助模型判断某个知识库与提问是否相关。

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-08-15 04:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f5a6b7c8d9e0'
down_revision: Union[str, Sequence[str], None] = 'e4f5a6b7c8d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('kbs', sa.Column('description', sa.String(512), nullable=False,
                                   server_default=''))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('kbs', 'description')

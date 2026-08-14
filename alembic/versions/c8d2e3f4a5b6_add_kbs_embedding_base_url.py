"""add kbs.embedding_base_url

为每个知识库支持可选的自定义嵌入服务端点（base_url），
空则回退该 provider 在配置里的默认端点。

Revision ID: c8d2e3f4a5b6
Revises: b7c1d2e3f4a5
Create Date: 2026-08-14 04:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8d2e3f4a5b6'
down_revision: Union[str, Sequence[str], None] = 'b7c1d2e3f4a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('kbs', sa.Column('embedding_base_url', sa.String(length=256), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('kbs', 'embedding_base_url')

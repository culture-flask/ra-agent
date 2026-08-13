"""add kbs.embedding_api_key

为每个知识库支持可选的专用嵌入 API Key（AES 加密存储），
空则回退系统默认 EMBEDDING_API_KEY。

Revision ID: b7c1d2e3f4a5
Revises: 484a0f86480d
Create Date: 2026-08-14 03:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7c1d2e3f4a5'
down_revision: Union[str, Sequence[str], None] = '484a0f86480d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('kbs', sa.Column('embedding_api_key', sa.String(length=512), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('kbs', 'embedding_api_key')

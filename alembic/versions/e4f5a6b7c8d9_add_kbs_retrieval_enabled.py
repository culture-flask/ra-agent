"""add kbs.retrieval_enabled

用户可自行禁止某个知识库被对话检索（LLM 选库与兜底全查都会排除）；
库本身保留可见可管理，随时可恢复。

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-08-15 03:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4f5a6b7c8d9'
down_revision: Union[str, Sequence[str], None] = 'd3e4f5a6b7c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('kbs', sa.Column('retrieval_enabled', sa.Boolean(), nullable=False,
                                   server_default=sa.text('true')))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('kbs', 'retrieval_enabled')

"""add kbs.embedded_model

记录该库向量实际由哪个嵌入模型写入（最近一次成功入库时的
provider/model_id/dim/base_url），用于查询时检测"嵌入模型 ≠ 查询模型"
并给出提醒。嵌入配置本身改为创建后可随时修改（PATCH /kbs/{id}/embedding）。

Revision ID: d3e4f5a6b7c8
Revises: c8d2e3f4a5b6
Create Date: 2026-08-15 03:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3e4f5a6b7c8'
down_revision: Union[str, Sequence[str], None] = 'c8d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('kbs', sa.Column('embedded_model', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('kbs', 'embedded_model')

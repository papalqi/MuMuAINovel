"""分析任务增加向量索引状态字段

Revision ID: 2475b45e7bee
Revises: d4d253e3f4c6
Create Date: 2026-02-28 13:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2475b45e7bee"
down_revision: Union[str, None] = "d4d253e3f4c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "memory_extracted_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="本次分析提取的记忆数量（关系库）",
        ),
    )
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "vector_expected_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="预期写入向量的数量",
        ),
    )
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "vector_added_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="实际写入向量的数量",
        ),
    )
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "vector_skipped_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="被跳过的向量数量（拦截/超长/空文本等）",
        ),
    )
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "vector_error_message",
            sa.Text(),
            nullable=True,
            comment="向量写入错误/警告信息（不一定导致任务失败）",
        ),
    )
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "vector_embed_id",
            sa.String(length=200),
            nullable=True,
            comment="本次任务使用的embedding配置标识（用于定位collection）",
        ),
    )
    op.add_column(
        "analysis_tasks",
        sa.Column(
            "vector_collection",
            sa.String(length=80),
            nullable=True,
            comment="本次任务写入的Chroma collection名",
        ),
    )


def downgrade() -> None:
    op.drop_column("analysis_tasks", "vector_collection")
    op.drop_column("analysis_tasks", "vector_embed_id")
    op.drop_column("analysis_tasks", "vector_error_message")
    op.drop_column("analysis_tasks", "vector_skipped_count")
    op.drop_column("analysis_tasks", "vector_added_count")
    op.drop_column("analysis_tasks", "vector_expected_count")
    op.drop_column("analysis_tasks", "memory_extracted_count")


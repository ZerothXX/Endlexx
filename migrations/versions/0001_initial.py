"""Initial character workbench schema.

Revision ID: 0001
"""

from alembic import op
from endlexx.db import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)


def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())

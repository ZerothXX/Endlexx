"""Add durable conversation summaries to existing databases."""

from alembic import op

from endlexx.db import ChatContext

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    ChatContext.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade():
    ChatContext.__table__.drop(bind=op.get_bind(), checkfirst=True)

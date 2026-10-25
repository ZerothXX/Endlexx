from alembic import context
from endlexx.db import Base, engine

target_metadata = Base.metadata
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

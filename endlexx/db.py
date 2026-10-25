import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings


def uid() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Character(Base):
    __tablename__ = "characters"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    package: Mapped[str | None] = mapped_column(String(48), nullable=True)
    avatar_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    active_version: Mapped[str | None] = mapped_column(String(48), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ProfileVersion(Base):
    __tablename__ = "profile_versions"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    character_id: Mapped[str] = mapped_column(ForeignKey("characters.id"), index=True)
    label: Mapped[str] = mapped_column(String(80))
    constraints: Mapped[dict] = mapped_column(JSON, default=dict)
    references: Mapped[list] = mapped_column(JSON, default=list)
    provenance: Mapped[str] = mapped_column(Text, default="用户设定")
    confirmed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    character_id: Mapped[str | None] = mapped_column(ForeignKey("characters.id"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(100), default="新会话")
    model_profile: Mapped[str] = mapped_column(String(20), default="local")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    assets: Mapped[list] = mapped_column(JSON, default=list)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    task_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ChatContext(Base):
    __tablename__ = "chat_contexts"
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    through_message_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    compressed_messages: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    filename: Mapped[str] = mapped_column(String(240))
    path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(24), default="image")
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"), nullable=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    character_id: Mapped[str] = mapped_column(ForeignKey("characters.id"), index=True)
    title: Mapped[str] = mapped_column(String(160))
    source: Mapped[str] = mapped_column(Text, default="用户上传")
    category: Mapped[str] = mapped_column(String(24), default="user")
    version: Mapped[str] = mapped_column(String(80), default="默认")
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    character_id: Mapped[str] = mapped_column(ForeignKey("characters.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list | None] = mapped_column(Vector(1024).with_variant(JSON(), "sqlite"), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(160), nullable=True)


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"), index=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    idempotency_key: Mapped[str] = mapped_column(String(100), unique=True)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    phase: Mapped[str] = mapped_column(String(120), default="等待执行")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Trace(Base):
    __tablename__ = "traces"
    id: Mapped[str] = mapped_column(String(48), primary_key=True, default=uid)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    node: Mapped[str] = mapped_column(String(48))
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


engine = create_engine(
    settings().database_url,
    **(
        {"connect_args": {"check_same_thread": False, "timeout": 30}}
        if settings().database_url.startswith("sqlite")
        else {}
    ),
)
Session = sessionmaker(engine, expire_on_commit=False)


if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def sqlite_config(connection, _):
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")


def init_db():
    settings().data_dir.mkdir(parents=True, exist_ok=True)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def record(row) -> dict:
    result = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        result[column.name] = value.isoformat() if isinstance(value, datetime) else value
    return result

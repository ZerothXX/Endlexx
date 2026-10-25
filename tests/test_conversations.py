import io
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image
from sqlalchemy import func, select

from endlexx.config import settings
from endlexx.context import MEMORY_HEADER, build_context, byte_size
from endlexx.db import Character, Chat, ChatContext, Message, Session, Task, Trace


def create_chat(client, character_id):
    response = client.post("/api/chats", json={"character_id": character_id})
    assert response.status_code == 201
    return response.json()


def test_multiple_chats_and_delete_clean_only_selected_conversation(client):
    role = client.post("/api/characters", json={"name": "ayaka"}).json()
    other_role = client.post("/api/characters", json={"name": "other"}).json()
    chats = [create_chat(client, role["id"]) for _ in range(3)]
    foreign = create_chat(client, other_role["id"])
    listed = client.get("/api/chats", params={"character_id": role["id"]}).json()
    assert {row["id"] for row in listed} == {row["id"] for row in chats}

    image = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(image, "PNG")
    asset = client.post("/api/assets", files={"file": ("original.png", image.getvalue(), "image/png")}).json()
    task = client.post(
        f"/api/chats/{chats[0]['id']}/send",
        json={
            "content": "保留图片",
            "assets": [asset["id"]],
            "idempotency_key": "delete-one-chat",
        },
    ).json()
    with Session() as db:
        db.get(Task, task["id"]).status = "completed"
        db.add(Trace(task_id=task["id"], node="plan", data={"test": True}))
        db.add(ChatContext(chat_id=chats[0]["id"], summary="旧对话摘要", compressed_messages=2))
        db.commit()
    with SqliteSaver.from_conn_string(str(settings().data_dir / "checkpoints.sqlite")) as saver:
        config = {"configurable": {"thread_id": task["id"], "checkpoint_ns": ""}}
        saver.put(config, empty_checkpoint(), {}, {})
        assert saver.get_tuple(config)

    assert client.delete(f"/api/chats/{chats[0]['id']}").status_code == 204
    assert client.get(f"/api/chats/{chats[0]['id']}/messages").status_code == 404
    assert client.get(f"/api/chats/{chats[0]['id']}/tasks").status_code == 404
    assert client.get(f"/api/chats/{chats[0]['id']}/context").status_code == 404
    assert client.get(f"/api/tasks/{task['id']}").status_code == 404
    assert client.get(asset["url"]).status_code == 200
    with Session() as db:
        assert db.get(Character, role["id"])
        assert db.get(Chat, foreign["id"])
        assert db.get(ChatContext, chats[0]["id"]) is None
        assert db.scalar(select(func.count()).select_from(Message)) == 0
        assert db.scalar(select(func.count()).select_from(Trace)) == 0
    with SqliteSaver.from_conn_string(str(settings().data_dir / "checkpoints.sqlite")) as saver:
        assert saver.get_tuple(config) is None
    assert len(client.get("/api/chats", params={"character_id": role["id"]}).json()) == 2


@pytest.mark.parametrize("status", ["queued", "running"])
def test_delete_busy_conversation_is_rejected_without_losing_data(client, status):
    role = client.post("/api/characters", json={"name": "busy"}).json()
    chat = create_chat(client, role["id"])
    task = client.post(
        f"/api/chats/{chat['id']}/send",
        json={
            "content": "你好",
            "idempotency_key": "busy-conversation",
        },
    ).json()
    with Session() as db:
        db.get(Task, task["id"]).status = status
        db.commit()
    assert client.delete(f"/api/chats/{chat['id']}").status_code == 409
    assert len(client.get(f"/api/chats/{chat['id']}/messages").json()) == 1
    assert client.get(f"/api/tasks/{task['id']}").json()["status"] == status
    assert client.delete("/api/chats/missing").status_code == 404


def seed_history():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with Session() as db:
        chat, other = Chat(), Chat()
        db.add_all([chat, other])
        db.flush()
        rows = []
        for i in range(14):
            row = Message(
                chat_id=chat.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"历史{i}：" + ("保留青绿色外套与月牙徽记。" * 12),
                created_at=start + timedelta(seconds=i),
            )
            db.add(row)
            rows.append(row)
        current = Message(
            chat_id=chat.id, role="user", content="继续当前需求", created_at=start + timedelta(seconds=20)
        )
        db.add(current)
        db.add(Message(chat_id=other.id, role="user", content="另一个对话的秘密"))
        db.flush()
        task = Task(
            chat_id=chat.id, message_id=current.id, idempotency_key="context-history", status="running"
        )
        db.add(task)
        db.commit()
        return chat.id, current.id, task.id, rows


def test_rolling_summary_is_persistent_scoped_and_keeps_recent_messages_complete(monkeypatch):
    chat_id, current_id, task_id, rows = seed_history()
    calls = []

    def summarize(snapshot, previous, chunk, limit):
        calls.append({"previous_summary": previous, "additional_history": chunk})
        return "用户要求保留青绿色外套和月牙徽记；继续当前角色创作。"

    monkeypatch.setattr("endlexx.context.compact_summary", summarize)
    state = build_context(chat_id, current_id, {"controller": "chosen-model"}, task_id)
    assert state["compressed_messages"] > 0
    assert "青绿色外套" in state["context"][0]["content"]
    assert state["context"][-1]["content"] == rows[-1].content
    assert sum(byte_size(row["content"]) + 32 for row in state["context"]) <= settings().context_history_bytes
    data = "".join(call["additional_history"] for call in calls)
    assert "历史0" in data
    assert "继续当前需求" not in data
    assert "另一个对话的秘密" not in data
    assert calls[-1]["previous_summary"]
    with Session() as db:
        saved = db.get(ChatContext, chat_id)
        assert saved.compressed_messages == state["compressed_messages"]
        assert db.scalar(select(func.count()).select_from(Message).where(Message.chat_id == chat_id)) == 15
    count = len(calls)
    assert build_context(chat_id, current_id, {}, task_id) == state
    assert len(calls) == count  # Restart/reload reuses the durable summary.

    # Actual model overflow can request a smaller context, including shrinking a cached summary.
    tiny = build_context(chat_id, current_id, {}, task_id, budget=512)
    assert tiny["compressed_messages"] >= state["compressed_messages"]
    assert byte_size(tiny["context_summary"]) <= 256


def test_older_retry_never_sees_or_overwrites_future_summary(monkeypatch):
    chat_id, current_id, task_id, rows = seed_history()
    monkeypatch.setattr("endlexx.context.compact_summary", lambda *_args, **_kwargs: "后续决定：未来标记")
    build_context(chat_id, current_id, {}, task_id, budget=512)
    captured = []

    def summarize(_snapshot, previous, chunk, limit):
        captured.append(previous + chunk)
        return "早期历史摘要"

    monkeypatch.setattr("endlexx.context.compact_summary", summarize)
    build_context(chat_id, rows[5].id, {}, task_id, budget=512)
    assert "未来标记" not in "".join(captured)
    assert "历史6" not in "".join(captured)
    with Session() as db:
        assert db.get(ChatContext, chat_id).summary == MEMORY_HEADER + "后续决定：未来标记"


def test_failed_compression_keeps_original_history_and_previous_summary(monkeypatch):
    chat_id, current_id, task_id, _ = seed_history()

    def fail(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("endlexx.context.compact_summary", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        build_context(chat_id, current_id, {}, task_id)
    with Session() as db:
        assert db.get(ChatContext, chat_id) is None
        assert db.scalar(select(func.count()).select_from(Message).where(Message.chat_id == chat_id)) == 15


def test_worker_passes_persistent_summary_to_graph_and_reports_compression(monkeypatch):
    from types import SimpleNamespace

    from endlexx.agent import run_task

    chat_id, current_id, task_id, _ = seed_history()
    with Session() as db:
        task = db.get(Task, task_id)
        task.status = "queued"
        task.snapshot = {"controller_source": "local", "controller": "chosen"}
        db.commit()
    calls = []
    monkeypatch.setattr(
        "endlexx.context.compact_summary",
        lambda snapshot, *_args, **_kwargs: calls.append(snapshot) or "保留青绿色外套和月牙徽记",
    )

    class Graph:
        def get_state(self, _config):
            return SimpleNamespace(next=())

        def invoke(self, initial, _config):
            assert initial["message_id"] == current_id
            assert initial["chat_id"] == chat_id
            assert initial["query"] == "继续当前需求"
            assert "青绿色外套" in initial["context"][0]["content"]
            return initial | {"reply": "已记住要求", "plan": ["chat"]}

    monkeypatch.setattr("endlexx.agent.get_graph", Graph)
    run_task(task_id)
    assert calls and all(call["controller"] == "chosen" for call in calls)
    with Session() as db:
        task = db.get(Task, task_id)
        assert task.status == "completed"
        assert task.result["compressed_messages"] > 0
        reply = db.scalar(select(Message).where(Message.task_id == task_id, Message.role == "assistant"))
        assert reply.content == "已记住要求"
        assert reply.details["compressed_messages"] == task.result["compressed_messages"]


def test_actual_model_overflow_recompresses_and_retries_without_changing_current_request(monkeypatch):
    from endlexx.agent import answer_node
    from endlexx.providers import ModelContextExceeded

    chat_id, current_id, task_id, _ = seed_history()
    monkeypatch.setattr(
        "endlexx.context.compact_summary",
        lambda _snapshot, _previous, _chunk, limit: "记住用户外套要求" * min(25, limit // 24),
    )
    state = build_context(chat_id, current_id, {}, task_id)
    state.update(
        chat_id=chat_id,
        message_id=current_id,
        task_id=task_id,
        snapshot={},
        query="必须完整保留的当前请求",
        plan=["chat"],
    )
    attempts = []

    def answer(_snapshot, messages, **_kwargs):
        attempts.append(messages)
        if len(attempts) == 1:
            raise ModelContextExceeded("context too long")
        return "继续回答"

    monkeypatch.setattr("endlexx.agent.generate", answer)
    result = answer_node(state)
    assert result["reply"] == "继续回答"
    assert len(attempts) == 2
    assert all(attempt[-1]["content"] == "必须完整保留的当前请求" for attempt in attempts)
    assert all(attempt[0]["role"] == "system" for attempt in attempts)
    assert len(str(attempts[1])) < len(str(attempts[0]))
    assert "conversation_summary" in attempts[1][1]["content"]


def test_summary_alone_is_recompressed_when_model_budget_shrinks(monkeypatch):
    chat_id, current_id, task_id, rows = seed_history()
    with Session() as db:
        db.add(
            ChatContext(
                chat_id=chat_id,
                through_message_id=rows[-1].id,
                compressed_messages=14,
                summary=MEMORY_HEADER + "用户：旧摘要中的重要约束。" * 30,
            )
        )
        db.commit()
    calls = []

    def summarize(_snapshot, previous, chunk, limit):
        calls.append({"previous_summary": previous, "additional_history": chunk})
        return "压缩后的约束"

    monkeypatch.setattr("endlexx.context.compact_summary", summarize)
    result = build_context(chat_id, current_id, {}, task_id, budget=512)
    assert result["compressed_messages"] == 14
    assert result["context_summary"] == MEMORY_HEADER + "压缩后的约束"
    assert "旧摘要中的重要约束" in calls[0]["previous_summary"]
    with Session() as db:
        assert db.get(ChatContext, chat_id).through_message_id == rows[-1].id

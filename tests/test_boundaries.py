import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from endlexx.agent import AgentState, answer_node, execute_node, plan_node, run_task
from endlexx.db import Chat, Message, Session, Task
from endlexx.images import NeedsClarification, automatic_mask
from endlexx.knowledge import retrieve
from endlexx.providers import profile_snapshot


def new_chat(client, character_id=None):
    if character_id is None:
        character_id = client.post("/api/characters", json={"name": "测试角色"}).json()["id"]
    return client.post("/api/chats", json={"character_id": character_id}).json()


def test_request_replay_does_not_duplicate_messages_or_tasks(client):
    chat = new_chat(client)
    payload = {"content": "你好", "assets": [], "idempotency_key": "request-stable-123"}
    first = client.post(f"/api/chats/{chat['id']}/send", json=payload)
    replay = client.post(f"/api/chats/{chat['id']}/send", json=payload)
    assert first.status_code == replay.status_code == 202
    assert first.json()["id"] == replay.json()["id"]
    messages = client.get(f"/api/chats/{chat['id']}/messages").json()
    assert len(messages) == 1
    assert (
        client.post(f"/api/chats/{chat['id']}/send", json=payload | {"content": "不同请求"}).status_code
        == 409
    )
    second_chat = new_chat(client)
    assert client.post(f"/api/chats/{second_chat['id']}/send", json=payload).status_code == 409


def test_model_snapshot_stays_local_when_chat_changes(client):
    chat = new_chat(client)
    task = client.post(
        f"/api/chats/{chat['id']}/send", json={"content": "你好", "idempotency_key": "snapshot-test"}
    ).json()
    client.patch(f"/api/chats/{chat['id']}", json={"model_profile": "api"})
    saved = client.get(f"/api/tasks/{task['id']}").json()
    assert saved["snapshot"]["network"] == "local_only"
    assert saved["snapshot"]["api_base_url"] is None
    assert "api_key" not in str(saved)


def test_cross_site_writes_are_rejected(client):
    response = client.post(
        "/api/characters", json={"name": "test"}, headers={"Origin": "https://untrusted.example"}
    )
    assert response.status_code == 403


def test_image_upload_decodes_content_and_rejects_non_images(client):
    assert (
        client.post("/api/assets", files={"file": ("bad.png", b"not an image", "image/png")}).status_code
        == 422
    )
    output = io.BytesIO()
    Image.new("RGB", (48, 32), "green").save(output, format="PNG")
    uploaded = client.post(
        "/api/assets", files={"file": ("../../outside.png", output.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["filename"] == "outside.png"
    assert "path" not in uploaded.json()
    assert client.get(uploaded.json()["url"]).headers["content-type"] == "image/png"


def test_character_scope_and_document_delete_remove_evidence(client, monkeypatch):
    def fake_runtime(path, payload):
        if path == "/embed":
            return {"vectors": [[1.0] + [0.0] * 1023 for _ in payload["texts"]], "model": "BAAI/bge-m3"}
        return {"scores": [1.0] * len(payload["texts"])}

    monkeypatch.setattr("endlexx.knowledge.runtime_call", fake_runtime)
    left = client.post("/api/characters", json={"name": "星遥"}).json()
    right = client.post("/api/characters", json={"name": "霁月"}).json()
    doc = client.post(
        f"/api/characters/{left['id']}/documents",
        json={
            "title": "星遥外观设定",
            "content": "星遥穿青绿色短外套，右肩有月牙徽记。",
            "source": "测试设定",
        },
    ).json()
    result = retrieve(left["id"], "月牙徽记在哪？")
    assert result["citations"][0]["document_id"] == doc["id"]
    assert retrieve(right["id"], "月牙徽记在哪？")["citations"] == []
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert retrieve(left["id"], "月牙徽记在哪？")["citations"] == []


def test_auto_mask_detects_enclosed_hole_and_abstains_on_background():
    image = Image.new("RGB", (160, 160), (120, 80, 180))
    ImageDraw.Draw(image).rectangle((60, 60, 90, 90), fill="white")
    diagnosis = {"regions": [{"kind": "hole", "box": [350, 350, 600, 600]}]}
    mask = np.asarray(automatic_mask(image, diagnosis))
    assert mask[75, 75] == 255
    assert mask[10, 10] == 0
    assert (mask > 0).mean() < 0.1
    with pytest.raises(NeedsClarification):
        automatic_mask(Image.new("RGB", (160, 160), "white"), diagnosis)
    with pytest.raises(NeedsClarification):
        automatic_mask(image, {"regions": []})


def test_plan_requires_image_without_requesting_a_mask(monkeypatch):
    with Session() as db:
        chat = Chat()
        db.add(chat)
        db.flush()
        message = Message(chat_id=chat.id, role="user", content="修复图片")
        db.add(message)
        db.flush()
        task = Task(chat_id=chat.id, message_id=message.id, idempotency_key="plan-image")
        db.add(task)
        db.commit()
        task_id = task.id
    monkeypatch.setattr(
        "endlexx.agent.generate", lambda *_, **__: '{"tools":["repair"],"summary":"修复局部缺损"}'
    )
    with pytest.raises(NeedsClarification, match="上传"):
        plan_node(AgentState(task_id=task_id, query="修复", snapshot=profile_snapshot("local")))


def test_durable_tool_receipt_prevents_repeated_execution(monkeypatch):
    saved_delta = {
        "cursor": 1,
        "results": [{"tool": "repair", "asset": {"id": "already-created"}}],
        "asset_id": "already-created",
    }
    with Session() as db:
        chat = Chat()
        db.add(chat)
        db.flush()
        message = Message(chat_id=chat.id, role="user", content="修复")
        db.add(message)
        db.flush()
        task = Task(
            chat_id=chat.id,
            message_id=message.id,
            idempotency_key="receipt-stable",
            result={"receipts": {"0": saved_delta}},
        )
        db.add(task)
        db.commit()
        task_id = task.id
    monkeypatch.setattr("endlexx.agent.repair_image", lambda *_: pytest.fail("completed tool was repeated"))
    state = AgentState(task_id=task_id, cursor=0, plan=["repair"])
    assert execute_node(state) == saved_delta


def test_long_model_clarification_preserves_error_and_finishes_task(client, monkeypatch):
    chat = new_chat(client)
    task = client.post(
        f"/api/chats/{chat['id']}/send",
        json={"content": "修复", "idempotency_key": "long-clarification"},
    ).json()
    reason = "无法确定缺损位置，请补充位置描述。" * 30

    class FailingGraph:
        def get_state(self, _):
            from types import SimpleNamespace

            return SimpleNamespace(next=())

        def invoke(self, *_):
            raise NeedsClarification(reason)

    monkeypatch.setattr("endlexx.agent.get_graph", lambda: FailingGraph())
    run_task(task["id"])
    saved = client.get(f"/api/tasks/{task['id']}").json()
    assert saved["status"] == "needs_input"
    assert saved["error"] == reason
    assert len(saved["phase"]) <= 120


def test_worker_restart_recovers_pending_work_and_honours_cancellation(client, monkeypatch):
    from endlexx.queue import recover_tasks

    ids = []
    for index in range(2):
        chat = new_chat(client)
        task = client.post(
            f"/api/chats/{chat['id']}/send", json={"content": "修复", "idempotency_key": f"restart-{index}"}
        ).json()
        ids.append(task["id"])
    with Session() as db:
        for index, task_id in enumerate(ids):
            task = db.get(Task, task_id)
            task.status = "running"
            task.result = {"receipts": {"0": {"cursor": 1}}}
            task.cancel_requested = index == 1
        db.commit()
    resubmitted = []
    monkeypatch.setattr("endlexx.queue.enqueue", resubmitted.append)
    assert recover_tasks() == 1
    assert resubmitted == ids[:1]
    with Session() as db:
        assert db.get(Task, ids[0]).status == "queued"
        assert db.get(Task, ids[0]).result["receipts"]
        assert db.get(Task, ids[1]).status == "cancelled"


def test_image_delivery_uses_tool_receipt_even_when_knowledge_is_also_requested(client, monkeypatch):
    chat = new_chat(client)
    task = client.post(
        f"/api/chats/{chat['id']}/send",
        json={"content": "查资料后迁移", "idempotency_key": "truthful-image-delivery"},
    ).json()
    monkeypatch.setattr(
        "endlexx.agent.generate",
        lambda *_args, **_kwargs: pytest.fail("Model must not override a successful tool receipt"),
    )
    reply = answer_node(
        AgentState(
            task_id=task["id"],
            plan=["knowledge", "transfer"],
            results=[{"tool": "transfer"}],
            citations=[{"number": 1, "title": "服装", "text": "青绿色外套"}],
        )
    )["reply"]
    assert "已完成角色视觉迁移" in reply
    assert "[1] 服装：青绿色外套" in reply

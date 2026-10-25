import hashlib
import json
import zipfile

import httpx
import pytest

from endlexx import deployments, model_registry
from endlexx.config import settings
from endlexx.db import Chat, Message, Session, Task


def api_payload(**changes):
    return (
        dict(
            name="私有视觉服务",
            base_url="https://models.example/v1",
            model="text-model",
            vision_model="vision-model",
            api_key="never-return-this-key",
        )
        | changes
    )


def test_connections_are_independent_and_snapshots_hide_credentials(client):
    first = client.post("/api/model-connections/api", json=api_payload()).json()
    second = client.post(
        "/api/model-connections/api", json=api_payload(name="第二服务", api_key="second-secret")
    ).json()
    assert first["id"] != second["id"]
    assert model_registry.credential(second["id"]) == "second-secret"
    role = client.post("/api/characters", json={"name": "角色"}).json()
    chat = client.post("/api/chats", json={"character_id": role["id"], "model_profile": first["id"]}).json()
    task = client.post(
        f"/api/chats/{chat['id']}/send", json={"content": "介绍设定", "idempotency_key": "freeze-connection"}
    ).json()
    changed = client.patch(
        f"/api/model-connections/{first['id']}", json=api_payload(model="new", api_key=None)
    )
    assert changed.status_code == 200
    assert model_registry.credential(first["id"]) == "never-return-this-key"
    saved = client.get(f"/api/tasks/{task['id']}")
    assert saved.json()["snapshot"]["controller"] == "text-model"
    assert saved.json()["snapshot"]["vision"] == "vision-model"
    frozen = saved.json()["snapshot"]
    client.patch(
        f"/api/model-connections/{first['id']}",
        json=api_payload(base_url="https://new.example/v1", api_key="rotated-secret"),
    )
    assert model_registry.credential(first["id"], frozen["credential_ref"]) == "never-return-this-key"
    assert model_registry.credential(first["id"]) == "rotated-secret"
    assert "never-return-this-key" not in saved.text + client.get("/api/model-connections").text
    assert client.delete(f"/api/model-connections/{first['id']}").status_code == 409
    client.post(f"/api/tasks/{task['id']}/cancel")
    assert client.delete(f"/api/model-connections/{first['id']}").status_code == 204
    assert client.get("/api/model-connections").json()["active_id"] == second["id"]


def test_home_is_not_a_general_task_entry(client):
    assert client.post("/api/chats", json={}).status_code == 422
    with Session() as db:
        old = Chat()
        db.add(old)
        db.commit()
        old_id = old.id
        message = Message(chat_id=old_id, role="user", content="修复")
        db.add(message)
        db.flush()
        task = Task(chat_id=old_id, message_id=message.id, idempotency_key="old-retry", status="failed")
        db.add(task)
        db.commit()
        old_task_id = task.id
    assert client.get(f"/api/chats/{old_id}/messages").status_code == 200
    assert (
        client.post(
            f"/api/chats/{old_id}/send", json={"content": "修复", "idempotency_key": "old-home-chat"}
        ).status_code
        == 422
    )

    assert client.post(f"/api/tasks/{old_task_id}/retry").status_code == 422


def test_new_chat_uses_active_model_without_rewriting_explicit_or_existing_choices(client):
    role = client.post("/api/characters", json={"name": "默认模型验收"}).json()
    first = client.post("/api/model-connections/api", json=api_payload()).json()
    chat = client.post("/api/chats", json={"character_id": role["id"]}).json()
    assert chat["model_profile"] == first["id"]
    legacy = client.post(
        "/api/chats", json={"character_id": role["id"], "model_profile": "local"}
    ).json()
    assert legacy["model_profile"] == "local"
    second = client.post(
        "/api/model-connections/api", json=api_payload(name="第二主控")
    ).json()
    assert client.post(f"/api/model-connections/{second['id']}/activate").status_code == 200
    current = client.post("/api/chats", json={"character_id": role["id"]}).json()
    assert current["model_profile"] == second["id"]
    saved = client.get("/api/chats", params={"character_id": role["id"]}).json()
    profiles = {row["id"]: row["model_profile"] for row in saved}
    assert profiles[chat["id"]] == first["id"]
    assert profiles[legacy["id"]] == "local"


def test_pending_deployment_cannot_be_selected_and_failed_job_can_retry(client, monkeypatch):
    def fake_launch(model_id):
        path = deployments.job_path(model_id)
        model_registry.atomic_json(
            path,
            dict(
                status="failed",
                model_id=model_id,
                current=0,
                updated_at=0,
                items=[],
                phase="网络中断",
                error="网络中断",
            ),
        )
        return {"status": "queued"}

    monkeypatch.setattr(deployments, "launch", fake_launch)
    entry = client.post(
        "/api/model-connections/deploy", json={"kind": "local", "model": "qwen3-vl:2b"}
    ).json()
    assert client.post(f"/api/model-connections/{entry['id']}/activate").status_code == 422
    assert client.get(f"/api/model-connections/{entry['id']}/deployment").json()["error"] == "网络中断"
    assert client.post(f"/api/model-connections/{entry['id']}/deployment/retry").status_code == 200
    assert (
        client.post(
            "/api/model-connections/deploy", json={"kind": "local", "model": "../../unsafe"}
        ).status_code
        == 422
    )


def installer_fixture():
    entry = model_registry.add_deployment("local", "qwen3-vl:2b")
    path = deployments.job_path(entry["id"])
    model_registry.atomic_json(
        path,
        dict(
            model_id=entry["id"],
            current=0,
            updated_at=0,
            status="queued",
            phase="准备",
            items=[dict(name="权重", status="queued", completed=0, total=None)],
        ),
    )
    return deployments.Installer(entry["id"])


def test_byte_progress_resume_hash_and_no_partial_publish(monkeypatch, tmp_path):
    installer = installer_fixture()
    data = b"valid-model-bytes"
    target = tmp_path / "weights.bin"
    target.with_suffix(".bin.part").write_bytes(data[:5])
    model_registry.atomic_json(
        target.with_suffix(".bin.part.json"),
        dict(
            url="https://official.example/revision/weights",
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        ),
    )
    requests = []

    def handle(request):
        requests.append(request.headers.get("range"))
        return httpx.Response(
            206, content=data[5:], headers={"content-range": f"bytes 5-{len(data) - 1}/{len(data)}"}
        )

    original = httpx.Client
    monkeypatch.setattr(
        deployments.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    installer.download(
        "https://official.example/revision/weights", target, hashlib.sha256(data).hexdigest(), len(data)
    )
    assert requests == ["bytes=5-"]
    assert target.read_bytes() == data
    assert not target.with_suffix(".bin.part").exists()
    assert installer.job["items"][0]["completed"] == len(data)
    assert installer.job["items"][0]["total"] == len(data)
    target.unlink()
    target.with_suffix(".bin.part").write_bytes(b"xxxxx")
    model_registry.atomic_json(
        target.with_suffix(".bin.part.json"),
        dict(
            url="https://official.example/revision/weights",
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        ),
    )
    with pytest.raises(ValueError, match="校验失败"):
        installer.download(
            "https://official.example/revision/weights", target, hashlib.sha256(data).hexdigest(), len(data)
        )
    assert not target.exists()


def test_cancel_survives_job_runner_and_never_marks_ready(monkeypatch):
    installer = installer_fixture()
    installer.path.with_suffix(".cancel").touch()
    monkeypatch.setattr(installer, "ollama", lambda: pytest.fail("cancelled job must not start downloading"))
    installer.run()
    assert deployments.read_job(installer.entry["id"])["status"] == "cancelled"
    assert model_registry.get_entry(installer.entry["id"])["status"] == "failed"


def test_runtime_zip_rejects_path_escape_and_shard_check_rejects_incomplete_model(tmp_path, monkeypatch):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.exe", "bad")
    with pytest.raises(ValueError, match="不安全路径"):
        deployments.extract_zip(archive, tmp_path / "runtime")
    assert not (tmp_path / "escape.exe").exists()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("runtime.exe", "valid")
    monkeypatch.setattr(deployments.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(ValueError, match="磁盘空间不足"):
        deployments.extract_zip(archive, tmp_path / "runtime")
    assert not (tmp_path / "runtime/runtime.exe").exists()
    for name in ["config.json", "tokenizer_config.json"]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}})
    )
    (tmp_path / "part-1.safetensors").write_bytes(b"present")
    assert not deployments.component_present(tmp_path, None)
    (tmp_path / "part-2.safetensors").write_bytes(b"present")
    assert deployments.component_present(tmp_path, None)


def test_fixed_bundle_manifest_and_steps_are_not_user_selectable(client):
    catalog = client.get("/api/model-catalog").json()
    assert catalog["bundle"] == [row[0] for row in deployments.BUNDLE]
    assert "Anime LaMa" in catalog["bundle"]
    assert (settings().data_dir / "model-secrets.json").exists() is False


def test_multifile_component_download_is_sequential_and_reports_aggregate_bytes(monkeypatch, tmp_path):
    installer = installer_fixture()
    files = {"config.json": b'{"ok":true}', "model.safetensors": b"model-checkpoint"}
    calls, progress = [], []

    def handle(request):
        calls.append(request.url.path)
        if "/api/models/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "sha": "fixed-revision",
                    "siblings": [
                        {
                            "rfilename": name,
                            "size": len(data),
                            "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                        }
                        for name, data in files.items()
                    ],
                },
            )
        return httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[-1]])

    original = httpx.Client
    monkeypatch.setattr(
        deployments.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    monkeypatch.setattr(
        installer, "progress", lambda completed=0, total=None, detail="": progress.append((completed, total))
    )
    installer.hf_component("official/model", tmp_path / "download", list(files))
    assert calls == [
        "/api/models/official/model",
        "/official/model/resolve/fixed-revision/config.json",
        "/official/model/resolve/fixed-revision/model.safetensors",
    ]
    total = sum(map(len, files.values()))
    assert progress[-1] == (total, total)
    assert [row[0] for row in progress] == sorted(row[0] for row in progress)
    assert json.loads((tmp_path / "download/endlexx-source.json").read_text())["revision"] == "fixed-revision"


def test_ollama_vision_protocol_unloads_native_model_and_never_exposes_thinking(monkeypatch, tmp_path):
    from endlexx.providers import ollama_generate

    monkeypatch.setattr(deployments, "ensure_ollama", lambda: None)
    calls, captured = [], {}
    monkeypatch.setattr("endlexx.providers.runtime_call", lambda path, **_: calls.append(path))
    response_content = {"message": {"content": '{"ok":true}', "thinking": "private reasoning"}}

    def handle(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=response_content)

    original = httpx.Client
    monkeypatch.setattr(
        deployments.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    image = tmp_path / "fixture.png"
    image.write_bytes(b"image-fixture")
    messages = [{"role": "user", "content": "analyze"}]
    assert ollama_generate("vision:instruct", messages, [image], 120, True) == '{"ok":true}'
    assert calls == ["/unload"]
    assert captured["keep_alive"] == 0
    assert captured["think"] is False
    assert captured["format"] == "json"
    assert captured["messages"][0]["images"]
    assert "images" not in messages[0]
    response_content["message"]["content"] = ""
    with pytest.raises(RuntimeError, match="Instruct"):
        ollama_generate("thinking-model", messages, [], 120, False)


def test_progress_atomic_replace_retries_transient_windows_reader(monkeypatch, tmp_path):
    original = model_registry.os.replace
    attempts = []

    def windows_reader(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            raise PermissionError("reader still holds handle")
        original(source, target)

    monkeypatch.setattr(model_registry.os, "replace", windows_reader)
    target = tmp_path / "progress.json"
    model_registry.atomic_json(target, {"completed": 1024, "total": 2048})
    assert json.loads(target.read_text()) == {"completed": 1024, "total": 2048}


def test_planner_retries_invalid_schema_once_and_does_not_execute_unknown_tools(client, monkeypatch):
    from endlexx.agent import AgentState, plan_node
    from endlexx.images import NeedsClarification
    from endlexx.providers import profile_snapshot

    role = client.post("/api/characters", json={"name": "规划校验"}).json()
    chat = client.post("/api/chats", json={"character_id": role["id"]}).json()
    task = client.post(
        f"/api/chats/{chat['id']}/send", json={"content": "你好", "idempotency_key": "planner-schema"}
    ).json()
    replies = iter(['{"tools":["arbitrary_command"],"summary":"bad"}', '{"tools":["chat"],"summary":"问候"}'])
    monkeypatch.setattr("endlexx.agent.generate", lambda *_, **__: next(replies))
    state = AgentState(
        task_id=task["id"], character_id=role["id"], query="你好", snapshot=profile_snapshot("local")
    )
    assert plan_node(state)["plan"] == ["chat"]
    monkeypatch.setattr("endlexx.agent.generate", lambda *_, **__: "你好")
    with pytest.raises(NeedsClarification, match="切换模型"):
        plan_node(state)


def test_text_controller_snapshot_keeps_native_vision_and_routes_images_separately(monkeypatch, tmp_path):
    from endlexx.providers import generate

    entry = model_registry.add_deployment("local", "qwen3:4b-instruct-2507-q4_K_M")
    model_registry.update_entry(
        entry["id"], status="ready", capabilities=["completion"], vision_fallback="native"
    )
    snapshot = model_registry.snapshot(entry["id"])
    assert snapshot["controller_source"] == "ollama"
    assert snapshot["vision_source"] == "local"
    assert snapshot["vision"] == "Qwen3-VL-4B-Instruct"
    assert snapshot["network"] == "local_only"
    calls = []
    monkeypatch.setattr("endlexx.providers.ollama_generate", lambda *args: calls.append(args) or "text")
    monkeypatch.setattr(
        "endlexx.providers.runtime_call", lambda path, payload: calls.append(payload) or {"text": "vision"}
    )
    messages = [{"role": "user", "content": "describe"}]
    assert generate(snapshot, messages) == "text"
    assert calls[-1][0] == entry["model"]
    image = tmp_path / "image.png"
    assert generate(snapshot, messages, [image]) == "vision"
    assert calls[-1]["images"] == [str(image.resolve())]
    # Editing a connection after queuing cannot silently reroute an existing task.
    model_registry.update_entry(entry["id"], vision_fallback=None)
    assert snapshot["vision_source"] == "local"

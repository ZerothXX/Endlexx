from endlexx.config import settings


def test_api_settings_keep_secrets_server_side_and_validate_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setattr("endlexx.api.ROOT", tmp_path)
    cfg = settings()
    for name in [
        "api_key",
        "api_base_url",
        "api_model",
        "vision_api_model",
        "hybrid_controller_source",
        "hybrid_vision_source",
    ]:
        monkeypatch.setattr(cfg, name, getattr(cfg, name))
    payload = {
        "base_url": "https://models.example/v1",
        "model": "controller",
        "vision_model": "vision",
        "api_key": "fixture-secret-123",
        "hybrid_controller_source": "local",
        "hybrid_vision_source": "api",
    }
    saved = client.post("/api/settings/api", json=payload)
    assert saved.status_code == 200
    assert saved.json()["configured"]
    assert "fixture-secret" not in saved.text
    assert "fixture-secret" not in client.get("/api/settings/api").text
    assert (
        client.post(
            "/api/settings/api", json=payload | {"base_url": "https://user:password@models.example"}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/settings/api", json=payload | {"base_url": "file:///local/model"}).status_code
        == 422
    )
    assert (tmp_path / ".env").is_file()


def test_new_profile_does_not_change_an_existing_task_snapshot(client):
    role = client.post("/api/characters", json={"name": "版本验收角色"}).json()
    first = client.post(
        f"/api/characters/{role['id']}/profiles", json={"label": "初始", "constraints": {"颜色": "蓝色"}}
    ).json()
    chat = client.post("/api/chats", json={"character_id": role["id"]}).json()
    task = client.post(
        f"/api/chats/{chat['id']}/send",
        json={"content": "查询外观", "idempotency_key": "frozen-profile-version"},
    ).json()
    second = client.post(
        f"/api/characters/{role['id']}/profiles", json={"label": "远航", "constraints": {"颜色": "红色"}}
    ).json()
    assert second["id"] != first["id"]
    assert client.get(f"/api/tasks/{task['id']}").json()["snapshot"]["profile_version_id"] == first["id"]
    versions = client.get(f"/api/characters/{role['id']}/profiles").json()
    assert len(versions) == 2

from pathlib import Path

import pytest

from endlexx.config import settings
from endlexx.providers import generate, profile_snapshot


def test_hybrid_routes_vision_independently_and_freezes_policy(monkeypatch):
    cfg = settings()
    monkeypatch.setattr(cfg, "hybrid_controller_source", "local")
    monkeypatch.setattr(cfg, "hybrid_vision_source", "api")
    snapshot = profile_snapshot("hybrid")
    assert snapshot["network"] == "mixed"
    monkeypatch.setattr(cfg, "hybrid_vision_source", "local")
    assert snapshot["vision_source"] == "api"
    calls = []
    monkeypatch.setattr(
        "endlexx.providers.runtime_call", lambda path, payload: calls.append(path) or {"text": "local reply"}
    )
    assert generate(snapshot, [{"role": "user", "content": "plan"}]) == "local reply"
    assert calls == ["/generate"]


def test_api_vision_protocol_and_error_redaction(monkeypatch, tmp_path):
    cfg = settings()
    monkeypatch.setattr(cfg, "hybrid_controller_source", "local")
    monkeypatch.setattr(cfg, "hybrid_vision_source", "api")
    snapshot = profile_snapshot("hybrid")
    monkeypatch.setattr("dotenv.dotenv_values", lambda *_: {"ENDLEXX_API_KEY": "test-secret"})
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"supported":true}'}}]}

    class Client:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response()

    monkeypatch.setattr("endlexx.providers.httpx.Client", Client)
    image = Path(tmp_path / "input.png")
    image.write_bytes(b"fixture-image")
    original = [{"role": "user", "content": "diagnose"}]
    assert generate(snapshot, original, [image], structured=True) == '{"supported":true}'
    assert captured["json"]["model"] == snapshot["vision"]
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert original[0]["content"] == "diagnose"
    Response.status_code = 401
    with pytest.raises(RuntimeError, match="HTTP 401") as caught:
        generate(snapshot, original, [image])
    assert "test-secret" not in str(caught.value)


def test_api_context_overflow_is_distinguished_without_echoing_upstream_content(monkeypatch):
    import httpx

    from endlexx.providers import ModelContextExceeded

    real_client = httpx.Client
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            400,
            json={
                "error": {"code": "context_length_exceeded", "message": "private upstream material"},
            },
        )
    )
    monkeypatch.setattr("dotenv.dotenv_values", lambda *_: {"ENDLEXX_API_KEY": "test-secret"})
    monkeypatch.setattr(
        "endlexx.providers.httpx.Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    with pytest.raises(ModelContextExceeded) as caught:
        generate(profile_snapshot("api"), [{"role": "user", "content": "请求"}])
    assert "private upstream material" not in str(caught.value)
    assert "test-secret" not in str(caught.value)


def test_old_runtime_dropped_history_triggers_application_compression(monkeypatch):
    from endlexx.providers import ModelContextExceeded

    monkeypatch.setattr("endlexx.providers.runtime_call", lambda *_args, **_kwargs: {
        "text": "answer based on incomplete history", "history_messages_dropped": 2,
    })
    with pytest.raises(ModelContextExceeded, match="压缩"):
        generate(profile_snapshot("local"), [{"role": "user", "content": "继续"}])

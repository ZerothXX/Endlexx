"""Small real inference acceptance against the running workbench, without API secrets."""

import json
import time

import httpx

if __name__ == "__main__":
    with httpx.Client(base_url="http://127.0.0.1:8765/api", timeout=30, trust_env=False) as client:
        client.post("/endless/import").raise_for_status()
        roles = client.get("/characters").json()
        role = next(row for row in roles if row["name"] == "ayaka")
        models = client.get("/model-connections").json()["entries"]
        local = next(row for row in models if row["engine"] == "ollama" and row["status"] == "ready")
        bundle = next(row for row in models if row["kind"] == "hybrid" and row["status"] == "ready")
        response = client.post(
            "/chats", json={"character_id": role["id"], "title": "模型切换验收", "model_profile": local["id"]}
        )
        response.raise_for_status()
        chat = response.json()
        evidence = []
        for index, model in enumerate([local, bundle]):
            client.patch(f"/chats/{chat['id']}", json={"model_profile": model["id"]}).raise_for_status()
            response = client.post(
                f"/chats/{chat['id']}/send",
                json={
                    "content": "你好，请只用一句简短的问候回复。",
                    "idempotency_key": "switch-" + str(time.time_ns()),
                },
            )
            response.raise_for_status()
            task_id = response.json()["id"]
            started = time.monotonic()
            while time.monotonic() - started < 300:
                task = client.get("/tasks/" + task_id).json()
                if task["status"] not in {"queued", "running"}:
                    break
                time.sleep(1)
            assert task["status"] == "completed", task.get("error")
            assert task["snapshot"]["connection_id"] == model["id"]
            evidence.append(
                dict(
                    model=model["model"],
                    engine=model["engine"],
                    task_id=task_id,
                    status=task["status"],
                    reply=task["result"].get("reply"),
                    elapsed_seconds=round(time.monotonic() - started, 1),
                )
            )
            print(json.dumps(evidence[-1], ensure_ascii=True), flush=True)
        from pathlib import Path

        (Path(__file__).resolve().parents[1] / "data/verification/model-switch.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )

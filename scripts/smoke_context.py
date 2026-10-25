"""Real worker/planner/summary/answer roundtrip, on the isolated acceptance DB."""

import argparse
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

if __package__:
    from .smoke_agent import wait
else:
    from smoke_agent import wait

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local")
    parser.add_argument("--base-url", default="http://127.0.0.1:8875")
    parser.add_argument(
        "--database-url",
        default="sqlite:///" + str(ROOT / "data/verification/full-acceptance.db").replace("\\", "/"),
        help="Must match the isolated acceptance API's database",
    )
    args = parser.parse_args()
    os.environ["ENDLEXX_DATABASE_URL"] = args.database_url
    from endlexx.db import ChatContext, Message, Session

    with httpx.Client(base_url=args.base_url, timeout=300, trust_env=False) as client:
        role = client.post("/api/characters", json={"name": "长对话真实验收"})
        role.raise_for_status()
        chat = client.post("/api/chats", json={"character_id": role.json()["id"], "model_profile": args.mode})
        chat.raise_for_status()
        chat_id = chat.json()["id"]
        start = datetime.now(timezone.utc) - timedelta(minutes=10)
        history = [
            (
                "user",
                "最终创作要求：手套黑色，裙子米白色。文件名draft_blue_2026.png，不覆盖原文件。检查失败先汇报，不要自动修改。",
            ),
            ("assistant", "收到上述创作要求。"),
        ] + [("assistant", "讨论画布显示效果，不增加创作要求。" * 25)] * 8
        with Session() as db:
            db.add_all(
                [
                    Message(
                        chat_id=chat_id, role=role, content=content, created_at=start + timedelta(seconds=i)
                    )
                    for i, (role, content) in enumerate(history)
                ]
            )
            db.commit()
        response = client.post(
            f"/api/chats/{chat_id}/send",
            json={
                "content": "回顾此前的创作要求，复述最终颜色、文件名和检查失败后的处理。只复述对话记忆，不查询知识库或操作图片。",
                "idempotency_key": uuid.uuid4().hex,
            },
        )
        response.raise_for_status()
        task = wait(client, response.json()["id"])
        with Session() as db:
            summary = db.get(ChatContext, chat_id)
            context = (
                {"summary": summary.summary, "compressed_messages": summary.compressed_messages}
                if summary
                else {}
            )
        output = ROOT / f"data/verification/acceptance-context-{args.mode}.json"
        output.write_text(
            json.dumps({"task": task, "context": context}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        assert task["status"] == "completed", task.get("error")
        assert task["result"]["plan"] == ["chat"]
        assert context["compressed_messages"] == len(history)
        reply = task["result"]["reply"]
        assert all(term in reply for term in ["黑色", "米白", "draft_blue_2026.png", "汇报"]), reply
        assert any(term in reply for term in ["不要自动", "不自动", "不会自动", "不进行自动"]), reply
        messages = client.get(f"/api/chats/{chat_id}/messages").json()
        assert messages[0]["content"] == history[0][1]
        assert len(messages) == len(history) + 2
        print("ACCEPTANCE PASSED long context", json.dumps(context, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Actual model-backed acceptance run, separate from mocked boundary tests."""

import argparse
import io
import json
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def wait(client, task_id, timeout=1800):
    started, phase = time.monotonic(), None
    while time.monotonic() - started < timeout:
        try:
            response = client.get(f"/api/tasks/{task_id}")
            response.raise_for_status()
            task = response.json()
        except httpx.TransportError:
            # Poll the same durable task after transient connection loss;
            # never submit another image operation just to recover progress.
            time.sleep(2)
            continue
        if task["phase"] != phase:
            phase = task["phase"]
            print(task["status"], phase, flush=True)
        if task["status"] not in {"queued", "running"}:
            return task
        time.sleep(2)
    raise TimeoutError("Acceptance task timed out")


def validate_task(task, feature):
    if task["status"] != "completed":
        raise AssertionError(f"Task did not complete: {task['status']}")
    result = task["result"]
    if feature == "knowledge":
        assert result["citations"], "Expected sourced knowledge answer"
        assert not result["retrieval"]["warnings"], "Real retrieval must not silently degrade"
        assert result["retrieval"]["method"] == "BM25 + 向量 + RRF + Reranker"
        assert "右肩" in result["reply"] and "珐琅" in result["reply"]
    elif feature == "repair":
        repair = next(item for item in result["results"] if item["tool"] == "repair")
        assert repair["metrics"]["protected_pixels_changed"] == 0
        assert repair["asset"]["id"] != repair["before"]
    elif feature == "transfer":
        transfer = next(item for item in result["results"] if item["tool"] == "transfer")
        assert transfer["manifest_sha256"] and transfer["weights"]
        assert transfer["asset"]["id"] != transfer["before"]
        # Successful execution does not assert visual identity or image quality.
    elif feature == "check":
        report = next(item["report"] for item in result["results"] if item["tool"] == "check")
        observed = {}
        for item in report["items"]:
            label = item["feature"] + " " + item["support_region"]
            for feature, word in [("头发", "发"), ("服装", "裙"), ("眼镜", "眼镜"), ("鞋子", "鞋")]:
                if word in label:
                    observed[feature] = item["status"]
        expected = {"头发": "match", "服装": "match", "眼镜": "mismatch", "鞋子": "not_visible"}
        assert observed == expected, f"Visible/missing/cropped fixture mismatch: {observed}"
    elif feature == "image_chat":
        assert result["plan"] == ["chat"]
        assert "白色" in result["reply"] and "吊带" in result["reply"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", choices=["repair", "knowledge", "transfer", "check", "image_chat"])
    parser.add_argument("--mode", default="local")
    parser.add_argument("--package", default="ayaka_f")
    parser.add_argument("--image", type=Path, help="Known acceptance fixture; visual assertions are fixture-specific")
    parser.add_argument("--base-url", default="http://127.0.0.1:8875")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=300, trust_env=False) as client:
        characters = client.get("/api/characters").json()
        # Never add synthetic lore to an existing real role. Only transfer needs
        # the imported visual package; all other acceptance uses its own role.
        role = next((value for value in characters if value["name"] == "验收样例（非正式角色）"), None)
        if args.feature == "transfer":
            imported = client.post("/api/endless/import")
            imported.raise_for_status()
            # Import is idempotent and only returns newly added packages.
            available = client.get("/api/characters")
            available.raise_for_status()
            role = next((value for value in available.json() if value["package"] == args.package), None)
            if role is None:
                raise RuntimeError(f"Acceptance requires installed character package {args.package!r}")
        elif role is None:
            response = client.post("/api/characters", json={"name": "验收样例（非正式角色）"})
            response.raise_for_status()
            role = response.json()
        image_id = None
        if args.feature != "knowledge":
            file = ROOT.parent / "Endless_xxx/input/human2.png"
            if args.feature == "repair":
                file = ROOT.parent / "Endless_xxx/dataset/ayaka/images/003.png"
            file = args.image or file
            if not file.is_file():
                raise SystemExit(
                    "Acceptance image is not bundled. Supply --image PATH using the known fixture; "
                    "visual checks are fixture-specific, not a generic image score."
                )
            image = Image.open(file).convert("RGB")
            image.thumbnail((512, 768))
            if args.feature == "repair":
                width, height = image.size
                ImageDraw.Draw(image).rectangle(
                    (int(width * 0.42), int(height * 0.43), int(width * 0.57), int(height * 0.53)),
                    fill="white",
                )
            output = io.BytesIO()
            image.save(output, format="PNG")
            response = client.post(
                "/api/assets", files={"file": ("验收图片.png", output.getvalue(), "image/png")}
            )
            response.raise_for_status()
            image_id = response.json()["id"]
        if args.feature == "check":
            # A separate test role avoids confirming model-extracted production lore.
            role = next((value for value in characters if value["name"] == "验收样例（非正式角色）"), None)
            if role is None:
                response = client.post(
                    "/api/characters",
                    json={
                        "name": "验收样例（非正式角色）",
                        "description": "仅用于可见外观、故意冲突和不可见特征的验收，不是正式角色设定。",
                    },
                )
                response.raise_for_status()
                role = response.json()
            response = client.post(
                f"/api/characters/{role['id']}/profiles",
                json={
                    "label": "固定验收设定",
                    "confirmed": True,
                    "references": [],
                    "provenance": "开发者编写的验收 fixture，眼镜条目故意与输入冲突，非官方设定。",
                    "constraints": {
                        "头发": "深棕色长发",
                        "服装": "白色吊带连衣裙",
                        "眼镜": "佩戴圆形黑框眼镜",
                        "鞋子": "白色运动鞋",
                    },
                },
            )
            response.raise_for_status()
        if args.feature == "knowledge":
            # Explicitly marked test material; does not invent official character lore.
            previous = client.get(f"/api/characters/{role['id']}/documents").json()
            for doc in previous:
                if (
                    doc["title"] == "验收用用户创作设定"
                    and doc["source"] == "Endlexx 自动验收用例，非官方角色设定"
                ):
                    client.delete(f"/api/documents/{doc['id']}").raise_for_status()
            response = client.post(
                f"/api/characters/{role['id']}/documents",
                json={
                    "title": "验收用用户创作设定",
                    "source": "Endlexx 自动验收用例，非官方角色设定",
                    "category": "user",
                    "content": "本条仅用于 Endlexx 检索验收：测试创作版本的角色在右肩佩戴青色月牙徽记。徽记材质为珐琅，不能放在左肩。",
                },
            )
            response.raise_for_status()
            print("index", response.json(), flush=True)
        query = {
            "repair": "修复这张图片中的局部缺损，保持其他区域不变。",
            "knowledge": "根据验收用用户创作设定，月牙徽记在哪边肩膀，是什么材质？请给出处。",
            "transfer": "把这张图片的主体迁移为当前角色的视觉特征。",
            "check": "检查这张图片是否符合当前角色的外观设定。",
            "image_chat": "只描述这张图片中人物衣服的颜色与款式，不进行修改或设定查询。",
        }[args.feature]
        chat = client.post(
            "/api/chats",
            json={
                "character_id": role["id"],
                "title": "实际模型验收 · " + args.feature,
                "model_profile": args.mode,
            },
        )
        chat.raise_for_status()
        chat = chat.json()
        response = client.post(
            f"/api/chats/{chat['id']}/send",
            json={
                "content": query,
                "assets": [image_id] if image_id else [],
                "idempotency_key": uuid.uuid4().hex,
            },
        )
        response.raise_for_status()
        task = wait(client, response.json()["id"])
        folder = ROOT / "data/verification"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"acceptance-{args.feature}-{args.mode}.json").write_text(
            json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("FINAL", task["status"], "task", task["id"], flush=True)
        validate_task(task, args.feature)
        print(task["result"].get("reply", ""), flush=True)
        print("ACCEPTANCE PASSED", args.feature, flush=True)


if __name__ == "__main__":
    main()

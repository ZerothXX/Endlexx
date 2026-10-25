"""Real deployment acceptance: downloads the small vision model if not installed."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endlexx.deployments import Installer, job_path, read_job  # noqa: E402
from endlexx.model_registry import add_deployment, atomic_json, listing  # noqa: E402

if __name__ == "__main__":
    entry = next(
        (
            row
            for row in listing()["entries"]
            if row["engine"] == "ollama" and row["model"] == "qwen3-vl:2b-instruct"
        ),
        None,
    )
    if entry is None:
        entry = add_deployment("local", "qwen3-vl:2b-instruct")
    names = ["Ollama 运行时", entry["model"], "视觉能力验证"]
    path = job_path(entry["id"])
    path.with_suffix(".cancel").unlink(missing_ok=True)
    atomic_json(
        path,
        dict(
            model_id=entry["id"],
            status="queued",
            phase="准备",
            current=0,
            updated_at=time.time(),
            error=None,
            items=[dict(name=name, status="queued", completed=0, total=None) for name in names],
        ),
    )
    Installer(entry["id"]).run()
    job = read_job(entry["id"])
    print(json.dumps({"id": entry["id"], "status": job["status"], "phase": job["phase"]}, ensure_ascii=True))
    sys.exit(0 if job["status"] == "completed" else 1)

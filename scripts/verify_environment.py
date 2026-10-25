"""Real environment diagnosis/repair acceptance, reusing this machine's installed runtimes."""

import json
import time
from pathlib import Path

import httpx


def main():
    with httpx.Client(base_url="http://127.0.0.1:8765/api", trust_env=False, timeout=90) as client:
        response = client.get("/environment")
        response.raise_for_status()
        report = response.json()
        evidence = dict(checks=report["checks"], jobs=[])
        for target in ("local", "hybrid"):
            response = client.post("/environment/repair", json={"target": target})
            response.raise_for_status()
            started = time.monotonic()
            while time.monotonic() - started < 300:
                response = client.get("/environment/repair")
                response.raise_for_status()
                job = response.json()
                if job["status"] not in {"queued", "running"}:
                    break
                time.sleep(1)
            assert job["status"] == "completed", job
            assert all(item["status"] == "completed" for item in job["items"])
            evidence["jobs"].append(job)
            print(
                json.dumps(
                    dict(target=target, status=job["status"], seconds=round(time.monotonic() - started, 1)),
                    ensure_ascii=True,
                ),
                flush=True,
            )
        path = Path(__file__).resolve().parents[1] / "data/verification/environment.json"
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

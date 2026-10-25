"""Exercise first-run setup in a new project-owned sandbox, without inherited runtimes."""

import argparse
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    args = parser.parse_args()
    folder = (
        Path(args.resume).resolve()
        if args.resume
        else ROOT / ("data/verification/fresh-setup-" + uuid.uuid4().hex[:8])
    )
    if not folder.resolve().is_relative_to(
        (ROOT / "data/verification").resolve()
    ) or not folder.name.startswith("fresh-setup-"):
        raise ValueError("Acceptance workspace must stay within data/verification/fresh-setup-*")
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("start.cmd", "start.ps1", "pyproject.toml", "requirements.lock", ".env.example", "README.en.md"):
        shutil.copy2(ROOT / name, folder / name)
    for name in ("endlexx", "scripts"):
        shutil.copytree(
            ROOT / name, folder / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copytree(
        ROOT / "frontend",
        folder / "frontend",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "node_modules", "dist", "test-results", "playwright-report", "*.tsbuildinfo"
        ),
    )
    print("Fresh workspace: " + str(folder), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    system = Path(os.environ.get("SystemRoot", "C:/Windows"))
    env["PATH"] = os.pathsep.join(
        str(path) for path in (system / "System32", system, system / "System32/WindowsPowerShell/v1.0")
    )
    for key in list(env):
        if key.startswith("ENDLEXX_"):
            del env[key]
    subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "start.cmd",
            "--setup-only",
        ],
        cwd=folder,
        env=env,
        check=True,
        timeout=3600,
    )
    python = folder / ".venv/Scripts/python.exe"
    # Bind a separate port and use a separate database; never stop the user's workbench.
    env.update(
        ENDLEXX_DATABASE_URL="sqlite:///" + str(folder / "data/acceptance.db").replace("\\", "/"),
        ENDLEXX_DATA_DIR=str(folder / "data"),
        ENDLEXX_TASK_BACKEND="local",
    )
    log = folder / "data/logs/acceptance.log"
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(python), "-m", "uvicorn", "endlexx.api:app", "--host", "127.0.0.1", "--port", "8775"],
            cwd=folder,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    try:
        import httpx

        with httpx.Client(base_url="http://127.0.0.1:8775", timeout=90, trust_env=False) as client:
            for _ in range(60):
                try:
                    if client.get("/api/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                if process.poll() is not None:
                    raise RuntimeError("Fresh API failed to start; see " + str(log))
                time.sleep(0.5)
            response = client.get("/")
            assert response.is_success and "/static/" in response.text
            models = client.get("/api/model-connections").json()
            assert models["entries"] == [], models
            report = client.get("/api/environment").json()
            assert any(row["id"] == "services" and "SQLite" in row["detail"] for row in report["checks"])
            assert (folder / "data/environment/bootstrap.json").is_file()
            result = dict(
                status="passed",
                workspace=str(folder),
                python_installed=True,
                frontend_built=True,
                empty_model_registry=True,
                checks=report["checks"],
            )
            (ROOT / "data/verification/fresh-setup.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("Fresh setup and empty-model workbench acceptance passed.", flush=True)
    finally:
        process.terminate()
        process.wait(timeout=20)


if __name__ == "__main__":
    main()

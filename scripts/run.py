"""Start the local stack without visible helper windows. Run from the app environment."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def services_ready(env):
    """Do not silently switch a configured workspace when services are unavailable."""
    from sqlalchemy import create_engine, text

    url = env.get("ENDLEXX_DATABASE_URL", "sqlite:///./data/endlexx.db")
    engine = None
    try:
        engine = create_engine(
            url, connect_args={"connect_timeout": 3} if url.startswith("postgresql") else {}
        )
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        if env.get("ENDLEXX_TASK_BACKEND", "local") == "celery":
            import redis

            client = redis.Redis.from_url(
                env.get("ENDLEXX_REDIS_URL", "redis://127.0.0.1:6379/0"),
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            try:
                client.ping()
            finally:
                client.close()
        return True
    except Exception:
        return False
    finally:
        if engine:
            engine.dispose()


def ensure_services(env):
    if services_ready(env):
        return env
    from sqlalchemy.engine import make_url

    url = make_url(env.get("ENDLEXX_DATABASE_URL", "sqlite:///./data/endlexx.db"))
    owned = (
        (ROOT / "data/local-services.json").is_file()
        and url.host in {"localhost", "127.0.0.1"}
        and url.username == "endlexx"
        and url.database == "endlexx"
    )
    if owned:
        print("Attempting to start this project's existing local database/queue...")
        with (ROOT / "data/logs/services.log").open("a", encoding="utf-8") as stream:
            try:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/bootstrap_services.py")],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if result.returncode == 0:
                    refreshed = dict(
                        env,
                        **{
                            key: value
                            for key, value in dotenv_values(ROOT / ".env").items()
                            if value is not None
                        },
                    )
                    if services_ready(refreshed):
                        return refreshed
            except (OSError, subprocess.SubprocessError):
                pass
    raise RuntimeError(
        "Configured database/queue is unavailable. Existing data was not changed. Check the configured service (WSL may require administrator access/restart), or use start.cmd --lightweight for a separate local workspace. Details: data/logs/services.log"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground", action="store_true", help="Keep the launcher alive")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--lightweight", action="store_true", help="Use a separate local SQLite workspace without WSL/Redis"
    )
    args = parser.parse_args()
    env = dict(
        os.environ,
        **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None},
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        USE_TF="0",
        USE_FLAX="0",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )
    if args.lightweight:
        env.update(ENDLEXX_DATABASE_URL="sqlite:///./data/endlexx-local.db", ENDLEXX_TASK_BACKEND="local")
        print(
            "Using separate SQLite workspace data/endlexx-local.db (original database configuration preserved)"
        )
    folder = ROOT / "data/logs"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        api_running = httpx.get("http://127.0.0.1:8765/api/health", timeout=1, trust_env=False).is_success
    except httpx.HTTPError:
        api_running = False
    if not api_running:
        env = ensure_services(env)
    model_python = env.get("ENDLEXX_MODEL_PYTHON", str(ROOT / ".modelenv/Scripts/python.exe"))
    if not Path(model_python).is_absolute():
        model_python = str((ROOT / model_python).resolve())
    endless = Path(env.get("ENDLEXX_ENDLESS_ROOT", str(ROOT.parent / "Endless_xxx")))
    commands = []
    if (
        not args.skip_model
        and Path(model_python).is_file()
        and (endless / "models/Qwen3-VL-4B-Instruct/config.json").is_file()
    ):
        commands.append(
            (
                "model",
                8766,
                [
                    model_python,
                    str(ROOT / "scripts/model_runtime.py"),
                    "--model",
                    str(endless / "models/Qwen3-VL-4B-Instruct"),
                    "--embedding",
                    str(ROOT / "data/models/bge-m3"),
                    "--reranker",
                    str(ROOT / "data/models/bge-reranker-v2-m3"),
                ],
            )
        )
    if env.get("ENDLEXX_TASK_BACKEND") == "celery":
        commands.append(
            (
                "worker",
                None,
                [
                    sys.executable,
                    str(ROOT / "scripts/worker.py"),
                ],
            )
        )
    commands.append(
        (
            "api",
            8765,
            [sys.executable, "-m", "uvicorn", "endlexx.api:app", "--host", "127.0.0.1", "--port", "8765"],
        )
    )
    state_file = ROOT / "data/processes.json"
    processes = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    for name, port, command in commands:
        if port:
            try:
                if (
                    httpx.get(
                        f"http://127.0.0.1:{port}/" + ("health" if name == "model" else "api/health"),
                        timeout=2,
                        trust_env=False,
                    ).status_code
                    == 200
                ):
                    print(f"{name}: already listening on {port}")
                    continue
            except httpx.HTTPError:
                pass
        if name == "worker":
            from filelock import FileLock, Timeout

            from endlexx.queue import celery_app

            probe = FileLock(str(ROOT / "data/worker.lock"))
            try:
                probe.acquire(timeout=0)
                probe.release()
            except Timeout:
                print("worker: already running")
                continue
            try:
                if celery_app.control.inspect(timeout=1).ping():
                    print("worker: already available")
                    continue
            except Exception:
                pass
        with (folder / f"{name}.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        processes[name] = process.pid
        print(f"{name}: started (PID {process.pid})")
    (ROOT / "data/processes.json").write_text(json.dumps(processes), encoding="utf-8")
    print("Open http://127.0.0.1:8765")
    if args.foreground:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            for pid in processes.values():
                import signal

                os.kill(pid, signal.SIGTERM)


if __name__ == "__main__":
    from filelock import FileLock

    (ROOT / "data").mkdir(exist_ok=True)
    with FileLock(str(ROOT / "data/launcher.lock"), timeout=45):
        main()

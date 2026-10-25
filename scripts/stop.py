"""Stop only process IDs created by this project's launcher after checking their command line."""

import json
import os
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from endlexx.model_registry import atomic_json

    state = ROOT / "data/processes.json"
    processes = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
    remaining = {}
    managed = ROOT / "data/runtimes/ollama/service.json"
    if managed.exists():
        service = json.loads(managed.read_text(encoding="utf-8"))
        if service.get("owner") == str(ROOT) and Path(service["executable"]).resolve().is_relative_to(ROOT):
            processes["ollama"] = service["pid"]
    for name, pid in processes.items():
        if os.name == "nt":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "$ErrorActionPreference='Stop'; "
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine",
                ],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode:
                print(
                    f"Cannot inspect {name}; retained PID {pid}. Process-inspection permission is required."
                )
                remaining[name] = pid
                continue
            command = result.stdout.replace("\\", "/")
            if str(ROOT).replace("\\", "/").lower() not in command.lower():
                continue
            if not any(
                marker in command
                for marker in [
                    "endlexx.api:app",
                    "endlexx.queue:celery_app",
                    "scripts/model_runtime.py",
                    "scripts/worker.py",
                    "ollama.exe",
                ]
            ):
                continue
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode:
                    print(f"Could not stop {name}; retained PID {pid}.")
                    remaining[name] = pid
                    continue
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"Stopped {name}")
        except ProcessLookupError:
            pass
        except PermissionError:
            remaining[name] = pid
    if remaining:
        atomic_json(state, remaining)
        return 1
    state.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

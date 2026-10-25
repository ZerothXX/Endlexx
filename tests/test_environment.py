import os
import subprocess
from pathlib import Path

import pytest

from endlexx import deployments, environment
from endlexx.config import settings
from scripts import bootstrap_app, run


def test_bootstrap_checks_bom_lock_markers_and_missing_pinned_dependency(tmp_path, monkeypatch):
    (tmp_path / "requirements.lock").write_text(
        'fastapi==1.0\npywin32==312; sys_platform == "win32"\n', encoding="utf-8-sig"
    )
    monkeypatch.setattr(
        bootstrap_app.importlib.metadata, "version", lambda name: {"fastapi": "1.0", "pywin32": "312"}[name]
    )
    assert bootstrap_app.dependencies_valid(tmp_path)
    monkeypatch.setattr(bootstrap_app.importlib.metadata, "version", lambda _: "0.1")
    assert not bootstrap_app.dependencies_valid(tmp_path)


def test_first_run_config_is_local_and_never_overwrites_existing_config(tmp_path):
    (tmp_path / ".env.example").write_text(
        "ENDLEXX_DATABASE_URL=sqlite:///./data/endlexx.db\nENDLEXX_MODEL_RUNTIME_TOKEN=replace-with-random-local-token",
        encoding="utf-8",
    )
    bootstrap_app.initialize_config(tmp_path)
    first = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "replace-with-random-local-token" not in first
    assert "sqlite" in first
    (tmp_path / ".env").write_text("existing-private-config", encoding="utf-8")
    bootstrap_app.initialize_config(tmp_path)
    assert (tmp_path / ".env").read_text() == "existing-private-config"


def test_frontend_checks_actual_static_assets_and_source_changes(tmp_path):
    folder = tmp_path / "frontend"
    (folder / "src").mkdir(parents=True)
    source = folder / "src/App.tsx"
    source.write_text("old")
    initial = bootstrap_app.frontend_fingerprint(tmp_path)
    source.write_text("new")
    assert bootstrap_app.frontend_fingerprint(tmp_path) != initial
    (folder / "dist/static").mkdir(parents=True)
    (folder / "dist/index.html").write_text('<script src="/static/app.js"></script>')
    assert not bootstrap_app.frontend_present(tmp_path)
    (folder / "dist/static/app.js").write_text("valid")
    assert bootstrap_app.frontend_present(tmp_path)


def test_bootstrap_builds_in_frontend_directory_and_persists_completion(tmp_path, monkeypatch):
    (tmp_path / "frontend/src").mkdir(parents=True)
    (tmp_path / ".env.example").write_text("ENDLEXX_TASK_BACKEND=local")
    node = tmp_path / "runtime/node.exe"
    npm = tmp_path / "runtime/npm-cli.js"
    instance = bootstrap_app.Bootstrap(tmp_path)
    monkeypatch.setattr(bootstrap_app, "dependencies_valid", lambda _: True)
    monkeypatch.setattr(
        bootstrap_app.subprocess, "run", lambda *_, **__: type("Result", (), {"returncode": 0})()
    )
    monkeypatch.setattr(instance, "ensure_node", lambda: node)
    monkeypatch.setattr(instance, "find_npm", lambda _: npm)
    commands = []

    def command(arguments, phase, env=None):
        commands.append(arguments)
        if arguments[-2:] == ["run", "build"]:
            (tmp_path / "frontend/dist/static").mkdir(parents=True)
            (tmp_path / "frontend/dist/static/app.js").write_text("built")
            (tmp_path / "frontend/dist/index.html").write_text('<script src="/static/app.js"></script>')

    monkeypatch.setattr(instance, "command", command)
    instance.run()
    npm_commands = [args for args in commands if args[0] == str(node)]
    assert len(npm_commands) == 2
    assert all(args[2:4] == ["--prefix", str(tmp_path / "frontend")] for args in npm_commands)
    assert instance.job["status"] == "completed"


def test_missing_gpu_blocks_research_before_download_but_allows_local_cpu(tmp_path, monkeypatch):
    checks = environment.hardware_checks("hybrid", gpus=[], memory=16 * 1024**3, free=100 * 1024**3)
    assert next(row for row in checks if row["id"] == "gpu")["status"] == "blocked"
    cpu = environment.hardware_checks("local", gpus=[], memory=16 * 1024**3, free=100 * 1024**3)
    assert not any(row["status"] == "blocked" for row in cpu)
    low_disk = environment.hardware_checks("local", gpus=[], memory=16 * 1024**3, free=100)
    assert next(row for row in low_disk if row["id"] == "disk")["status"] == "blocked"
    old_driver = environment.hardware_checks(
        "hybrid",
        gpus=[dict(name="GPU", driver="400.0", memory_mb=8192)],
        memory=16 * 1024**3,
        free=100 * 1024**3,
    )
    assert next(row for row in old_driver if row["id"] == "gpu")["status"] == "blocked"
    monkeypatch.setattr(deployments, "torch_environment_valid", lambda _: False)

    def blocked(*_):
        raise ValueError("缺少 NVIDIA 驱动")

    monkeypatch.setattr(environment, "assert_deployment_hardware", blocked)
    installer = object.__new__(deployments.Installer)
    monkeypatch.setattr(
        installer, "command", lambda *_, **__: pytest.fail("Hardware gate must precede pip downloads")
    )
    with pytest.raises(ValueError, match="NVIDIA"):
        installer.ensure_torch()


def test_environment_api_redacts_secrets_and_rejects_arbitrary_commands(client, monkeypatch):
    monkeypatch.setattr(environment, "gpu_info", lambda: [])
    monkeypatch.setattr(environment, "refresh_model_paths", lambda: None)
    monkeypatch.setattr(deployments, "torch_environment_valid", lambda _: False)
    monkeypatch.setattr(settings(), "api_key", "private-environment-test-key")
    response = client.get("/api/environment")
    assert response.status_code == 200
    assert "private-environment-test-key" not in response.text
    assert next(row for row in response.json()["checks"] if row["id"] == "torch")["status"] == "repairable"
    assert client.post("/api/environment/repair", json={"target": "shell-command"}).status_code == 422


def test_repair_is_durable_serialized_cancelable_and_preserves_models(monkeypatch):
    monkeypatch.setattr(environment, "assert_deployment_hardware", lambda *_: None)
    launched = []
    monkeypatch.setattr(environment.subprocess, "Popen", lambda args, **kwargs: launched.append(args))
    environment.launch_repair("local")
    assert launched and "repair_environment.py" in launched[0][1]
    with pytest.raises(ValueError, match="进行中"):
        environment.launch_repair("local")
    environment.cancel_repair()
    monkeypatch.setattr(
        deployments, "ensure_ollama", lambda *_: pytest.fail("Cancelled repair must not start a download")
    )
    environment.run_repair()
    assert environment.repair_status()["status"] == "cancelled"
    assert not (settings().data_dir / "model-registry.json").exists()


def test_repair_success_verifies_runtime_without_pretending_weights_are_installed(monkeypatch):
    monkeypatch.setattr(environment, "assert_deployment_hardware", lambda *_: None)
    monkeypatch.setattr(environment.subprocess, "Popen", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(deployments, "ensure_ollama", lambda *_: calls.append("runtime"))
    environment.launch_repair("local")
    environment.run_repair()
    value = environment.repair_status()
    assert calls == ["runtime"]
    assert value["status"] == "completed"
    assert all(row["status"] == "completed" for row in value["items"])
    assert "权重" in value["items"][-1]["detail"]


def test_unavailable_external_services_are_not_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "services_ready", lambda _: False)
    monkeypatch.setattr(
        run.subprocess, "run", lambda *_, **__: pytest.fail("Must not reconfigure an external database")
    )
    configured = {
        "ENDLEXX_DATABASE_URL": "postgresql://private:secret@external.example/original",
        "ENDLEXX_TASK_BACKEND": "celery",
    }
    with pytest.raises(RuntimeError) as error:
        run.ensure_services(configured)
    assert "secret" not in str(error.value)
    assert "external.example" in configured["ENDLEXX_DATABASE_URL"]


def test_model_environment_probe_timeout_does_not_hang(monkeypatch, tmp_path):
    executable = tmp_path / ("python.exe" if os.name == "nt" else "python")
    executable.touch()

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] <= 60
        raise deployments.subprocess.TimeoutExpired("probe", 60)

    monkeypatch.setattr(deployments.subprocess, "run", timeout)
    assert not deployments.torch_environment_valid(executable)


@pytest.mark.skipif(os.name != "nt", reason="Windows first-run entry point")
def test_windows_launcher_parameter_metadata_has_no_conflicting_aliases():
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "$ErrorActionPreference='Stop'; $command = Get-Command -Name './start.ps1'; $command.Parameters.Keys",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "Lightweight" in result.stdout and "SetupOnly" in result.stdout

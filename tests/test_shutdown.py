import json
import os
from types import SimpleNamespace

import pytest

from scripts import stop

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process ownership and taskkill behavior")


def test_shutdown_retains_ownership_record_when_inspection_is_denied(monkeypatch, tmp_path):
    monkeypatch.setattr(stop, "ROOT", tmp_path)
    state = tmp_path / "data/processes.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"model": 12345}))
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(stop.subprocess, "run", run)
    assert stop.main() == 1
    assert json.loads(state.read_text()) == {"model": 12345}
    assert len(commands) == 1  # Never kill an unverified process.


def test_shutdown_never_terminates_a_reused_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(stop, "ROOT", tmp_path)
    state = tmp_path / "data/processes.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"api": 12345}))
    calls = []
    monkeypatch.setattr(
        stop.subprocess,
        "run",
        lambda *args, **_kwargs: (
            calls.append(args) or SimpleNamespace(returncode=0, stdout="C:/OtherApp/service.exe")
        ),
    )
    assert stop.main() == 0
    assert len(calls) == 1
    assert not state.exists()

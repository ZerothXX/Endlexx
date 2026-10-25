import os
import tempfile
from pathlib import Path

import pytest

_test_root = tempfile.TemporaryDirectory(prefix="endlexx-tests-")
root = Path(_test_root.name)
os.environ["ENDLEXX_DATABASE_URL"] = "sqlite:///" + str(root / "test.db").replace("\\", "/")
os.environ["ENDLEXX_DATA_DIR"] = str(root)
os.environ["ENDLEXX_TASK_BACKEND"] = "local"
os.environ["ENDLEXX_REQUIRE_ACCESS_TOKEN"] = "false"

from endlexx.db import Base, engine, init_db  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_database(monkeypatch, tmp_path):
    from endlexx.config import settings

    monkeypatch.setattr(settings(), "data_dir", tmp_path)
    monkeypatch.setattr("endlexx.model_registry.bootstrap", lambda: {"active_id": None, "entries": []})
    init_db()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from endlexx.api import app

    monkeypatch.setattr("endlexx.api.enqueue", lambda _task: None)
    with TestClient(app) as instance:
        yield instance

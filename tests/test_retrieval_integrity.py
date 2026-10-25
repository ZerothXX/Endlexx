import numpy as np
import pytest
from sqlalchemy import select

from endlexx.db import Chunk, Session
from endlexx.knowledge import retrieve


@pytest.mark.parametrize("vectors", [[], [[0.0] * 1024], [[float("nan")] * 1024] * 2])
def test_partial_or_invalid_embeddings_keep_every_document_chunk(client, monkeypatch, vectors):
    monkeypatch.setattr(
        "endlexx.knowledge.runtime_call", lambda *_: {"vectors": vectors, "model": "BAAI/bge-m3"}
    )
    role = client.post("/api/characters", json={"name": "索引完整性"}).json()
    response = client.post(
        f"/api/characters/{role['id']}/documents",
        json={"title": "长资料", "content": "右肩月牙徽记。" * 170, "source": "测试"},
    )
    assert response.status_code == 201
    with Session() as db:
        chunks = db.scalars(select(Chunk)).all()
        assert len(chunks) == 2
        assert all(row.embedding is None for row in chunks)
    assert response.json()["warning"]


@pytest.mark.parametrize("scores", [[], [float("inf")], [float("nan")]])
def test_invalid_rerank_preserves_candidates_and_reports_fallback(client, monkeypatch, scores):
    def runtime(path, payload):
        if path == "/embed":
            return {"vectors": [np.eye(1, 1024)[0].tolist()] * len(payload["texts"]), "model": "BAAI/bge-m3"}
        return {"scores": scores}

    monkeypatch.setattr("endlexx.knowledge.runtime_call", runtime)
    role = client.post("/api/characters", json={"name": "重排完整性"}).json()
    client.post(
        f"/api/characters/{role['id']}/documents", json={"title": "徽记", "content": "右肩佩戴月牙徽记。"}
    )
    result = retrieve(role["id"], "徽记在哪")
    assert len(result["citations"]) == 1
    assert result["warnings"] == ["重排模型暂不可用，保留融合排名"]

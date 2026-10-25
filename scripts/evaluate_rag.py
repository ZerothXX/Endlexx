"""Real-model retrieval ablation on a small, declared synthetic fixture."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "[RAG验收] "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url",
        default="sqlite:///" + str(ROOT / "data/verification/rag-evaluation.db").replace("\\", "/"),
    )
    args = parser.parse_args()
    os.environ["ENDLEXX_DATABASE_URL"] = args.database_url
    from endlexx.db import Character, Document, Session, init_db
    from endlexx.knowledge import index_document, retrieve

    (ROOT / "data/verification").mkdir(parents=True, exist_ok=True)
    init_db()
    fixture = json.loads((ROOT / "evals/rag_fixture.json").read_text(encoding="utf-8"))
    with Session() as db:
        role = Character(name="验收样例（非正式角色）", description=fixture["notice"])
        db.add(role)
        db.commit()
        role_id = role.id
    for doc in fixture["documents"]:
        with Session() as db:
            document = Document(
                **(
                    doc
                    | {
                        "character_id": role_id,
                        "title": PREFIX + doc["title"],
                        "source": "Endlexx RAG 验收 fixture",
                        "category": "user",
                    }
                )
            )
            db.add(document)
            db.commit()
            document_id = document.id
        if not index_document(document_id)["vector_indexed"]:
            raise RuntimeError("Real-model benchmark requires successful vector indexing")
    report = {
        "notice": fixture["notice"],
        "documents": len(fixture["documents"]),
        "queries": len(fixture["queries"]),
        "strategies": {},
    }
    # Warm up model loading separately; report steady-state request timings.
    retrieve(role_id, fixture["queries"][0]["query"], "初始")
    for strategy in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        rows, durations = [], []
        for case in fixture["queries"]:
            started = time.perf_counter()
            result = retrieve(role_id, case["query"], case["version"], 3, strategy)
            if strategy != "bm25" and any("向量召回" in warning for warning in result["warnings"]):
                raise RuntimeError("Vector retrieval degraded; refusing to report this as a dense benchmark")
            duration = time.perf_counter() - started
            durations.append(duration)
            expected = PREFIX + case["expected"]
            ranks = [
                i + 1
                for i, citation in enumerate(result["citations"])
                if citation["title"] == expected and citation["version"] in {case["version"], "默认"}
            ]
            rows.append(
                {
                    **case,
                    "rank": min(ranks) if ranks else None,
                    "retrieved": [
                        {"title": c["title"], "version": c["version"]} for c in result["citations"]
                    ],
                    "seconds": round(duration, 3),
                    "warnings": result["warnings"],
                }
            )
        summary = {
            "recall_at_3": sum(row["rank"] is not None for row in rows) / len(rows),
            "mrr_at_3": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / len(rows),
            "p50_seconds": statistics.median(durations),
            "p95_seconds": sorted(durations)[min(len(durations) - 1, int(len(durations) * 0.95))],
        }
        report["strategies"][strategy] = {"summary": summary, "cases": rows}
        print(strategy, json.dumps(summary), flush=True)
    output = ROOT / "data/verification/rag-ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

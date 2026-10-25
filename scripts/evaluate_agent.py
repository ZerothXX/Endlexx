"""Real inference quality checks against production planner/context, in an isolated DB.

Run with application Python: scripts/evaluate_agent.py --model native --label baseline.
For an already downloaded text model: --model ollama:qwen3:4b-instruct-2507-q4_K_M.
Results include raw outputs, declared lexical scoring, latency and implementation hash.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="native")
    parser.add_argument("--label", required=True)
    parser.add_argument("--only", choices=["plans", "summaries", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", help="Comma-separated fixture IDs")
    parser.add_argument(
        "--implementation-dir",
        type=Path,
        help="Frozen planner/context/provider sources for an apples-to-apples baseline",
    )
    args = parser.parse_args()
    if not all(c.isalnum() or c in "-_" for c in args.label):
        parser.error("label may only contain letters, digits, hyphens and underscores")
    folder = ROOT / "data/verification/agent-quality" / args.label
    folder.mkdir(parents=True, exist_ok=True)
    source_folder = folder / "source"
    source_folder.mkdir(exist_ok=True)
    for name in ["agent.py", "context.py", "providers.py"]:
        origin = (args.implementation_dir or ROOT / "endlexx") / name
        (source_folder / name).write_bytes(origin.read_bytes())
    os.environ["ENDLEXX_DATABASE_URL"] = "sqlite:///" + str(folder / "evaluation.db").replace("\\", "/")
    os.environ["ENDLEXX_TASK_BACKEND"] = "local"
    # Keep the existing runtime/model store/GPU lock; only DB and report are isolated.
    os.environ["ENDLEXX_DATA_DIR"] = str(ROOT / "data")
    if args.implementation_dir:
        import endlexx

        for name in ["providers", "context", "agent"]:
            spec = importlib.util.spec_from_file_location(
                "endlexx." + name, args.implementation_dir / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            setattr(endlexx, name, module)
            spec.loader.exec_module(module)
    import endlexx.agent as agent_module
    import endlexx.context as context_module
    from endlexx.agent import plan_node
    from endlexx.context import build_context
    from endlexx.db import Character, Chat, Message, ProfileVersion, Session, Task, init_db
    from endlexx.images import NeedsClarification
    from endlexx.providers import profile_snapshot

    model_outputs = []
    real_generate = agent_module.generate

    def observed_generate(*positional, **keywords):
        output = real_generate(*positional, **keywords)
        model_outputs.append(output)
        return output

    agent_module.generate = context_module.generate = observed_generate

    init_db()
    fixture = json.loads((ROOT / "evals/agent_quality.json").read_text(encoding="utf-8"))
    snapshot = profile_snapshot("local")
    if args.model.startswith("ollama:"):
        snapshot.update(controller_source="ollama", controller=args.model.removeprefix("ollama:"))
    report = {
        "notice": fixture["notice"],
        "model": args.model,
        "scoring": "Tools require exact sequence, conditions require exact alignment; summaries use explicit alternative-substring recall, with raw text for manual review. Not a semantic hallucination-proof metric.",
        "implementation_sha256": hashlib.sha256(
            b"".join(
                (
                    (args.implementation_dir / name) if args.implementation_dir else (ROOT / "endlexx" / name)
                ).read_bytes()
                for name in ["agent.py", "context.py", "providers.py"]
            )
        ).hexdigest(),
        "fixture_sha256": hashlib.sha256((ROOT / "evals/agent_quality.json").read_bytes()).hexdigest(),
        "cases": [],
    }
    with Session() as db:
        role = Character(name="模型评测虚构角色", package="fixture")
        db.add(role)
        db.flush()
        profile = ProfileVersion(
            character_id=role.id, label="初始", confirmed=True, constraints={"徽记": "右肩"}
        )
        db.add(profile)
        db.flush()
        role.active_version = profile.id
        db.commit()
        role_id = role.id

    def save():
        groups = {}
        for group in ["tool_selection", "complex_planning", "summary_fidelity"]:
            rows = [r for r in report["cases"] if r["group"] == group]
            if rows:
                groups[group] = {
                    "passed": sum(r["passed"] for r in rows),
                    "total": len(rows),
                    "p50_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
                }
        report["summary"] = groups
        (folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    cases = []
    if args.only in {"all", "plans"}:
        cases.extend(
            ("complex_planning" if c["id"].startswith("c") else "tool_selection", c) for c in fixture["plans"]
        )
    if args.only in {"all", "summaries"}:
        cases.extend(("summary_fidelity", c) for c in fixture["summaries"])
    if args.limit:
        cases = cases[: args.limit]
    if args.cases:
        cases = [(group, case) for group, case in cases if case["id"] in args.cases.split(",")]
    for group, case in cases:
        model_outputs.clear()
        started = time.perf_counter()
        row = {"id": case["id"], "group": group, "passed": False}
        with Session() as db:
            chat = Chat(character_id=role_id)
            db.add(chat)
            db.flush()
            history = case.get("history", [])
            if group == "summary_fidelity":
                # Put facts early, followed by a long harmless turn to force actual compaction.
                history = history + [
                    {"role": "assistant", "content": "下面讨论画布的显示效果，不增加任何创作要求。" * 35}
                ]
            base = datetime.now(timezone.utc)
            for i, old in enumerate(history):
                db.add(
                    Message(
                        chat_id=chat.id,
                        role=old["role"],
                        content=old["content"],
                        created_at=base + timedelta(seconds=i),
                    )
                )
            message = Message(
                chat_id=chat.id,
                role="user",
                content=case.get("query", "继续，不改变此前要求"),
                created_at=base + timedelta(seconds=len(history) + 1),
            )
            db.add(message)
            db.flush()
            task = Task(
                chat_id=chat.id,
                message_id=message.id,
                idempotency_key=message.id,
                status="running",
                snapshot=snapshot,
            )
            db.add(task)
            db.commit()
            ids = chat.id, message.id, task.id
        try:
            if group == "summary_fidelity":
                state = build_context(*ids[:2], snapshot, ids[2], budget=1200)
                text = state["context_summary"]
                matched = [any(term in text for term in alternatives) for alternatives in case["facts"]]
                row.update(
                    output=text,
                    fact_checks=matched,
                    fact_recall=sum(matched) / len(matched),
                    passed=all(matched),
                    compressed_messages=state["compressed_messages"],
                )
            else:
                try:
                    state = plan_node(
                        {
                            "task_id": ids[2],
                            "query": case["query"],
                            "snapshot": snapshot,
                            "character_id": role_id,
                            "asset_id": "fixture-image" if case.get("image") else None,
                            "context": history,
                        }
                    )
                    row.update(output=state, expected_tools=case.get("tools"))
                    expected_conditions = case.get("conditions", ["always"] * len(case.get("tools", [])))
                    actual_conditions = state.get("conditions", ["always"] * len(state["plan"]))
                    row["passed"] = (
                        not case.get("clarification")
                        and state["plan"] == case["tools"]
                        and actual_conditions == expected_conditions
                    )
                except NeedsClarification as exc:
                    row.update(clarification=str(exc), passed=bool(case.get("clarification")))
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc)[:250])
        row["seconds"] = round(time.perf_counter() - started, 3)
        row["raw_model_outputs"] = list(model_outputs)
        report["cases"].append(row)
        save()
        print(json.dumps({k: row[k] for k in ("id", "passed", "seconds")}, ensure_ascii=True), flush=True)
    print(json.dumps(report["summary"], ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

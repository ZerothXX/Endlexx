import json

import pytest
from sqlalchemy import select
from test_conversations import seed_history

from endlexx.agent import Plan, execute_node, plan_node, validate_requested_branch
from endlexx.context import build_context, compact_summary, history_chunks
from endlexx.db import ChatContext, Session, Task, Trace
from endlexx.images import NeedsClarification


@pytest.mark.parametrize(
    "statuses,stop,expected",
    [
        (["match"], False, False),
        (["unknown"], False, False),
        (["not_visible"], False, False),
        (["mismatch"], False, True),
        (["mismatch", "unknown"], True, False),
        (["mismatch", "not_visible"], True, False),
        (["mismatch", "match"], True, True),
    ],
)
def test_conditional_modification_uses_observations_and_replays_receipt(
    monkeypatch, statuses, stop, expected
):
    _, _, task_id, _ = seed_history()
    calls = []
    monkeypatch.setattr(
        "endlexx.agent.transfer_image",
        lambda *_args, **_kwargs: calls.append(True) or {"tool": "transfer", "asset": {"id": "new-image"}},
    )
    state = {
        "task_id": task_id,
        "cursor": 1,
        "plan": ["check", "transfer"],
        "conditions": ["always", "if_mismatch"],
        "stop_on_uncertainty": stop,
        "asset_id": "original",
        "snapshot": {},
        "results": [
            {
                "tool": "check",
                "asset_id": "original",
                "report": {"items": [{"status": value} for value in statuses]},
            }
        ],
    }
    delta = execute_node(state)
    assert bool(calls) == expected
    assert delta["cursor"] == 2
    assert bool(delta.get("skipped_tools")) != expected
    assert execute_node(state) == delta
    assert len(calls) == int(expected)
    with Session() as db:
        trace = db.scalar(select(Trace).where(Trace.task_id == task_id))
        assert trace.data["skipped"] != expected


def test_conditional_edit_rejects_a_stale_image_check():
    _, _, task_id, _ = seed_history()
    with pytest.raises(NeedsClarification, match="当前图片"):
        execute_node(
            {
                "task_id": task_id,
                "cursor": 1,
                "plan": ["check", "repair"],
                "conditions": ["always", "if_mismatch"],
                "asset_id": "different",
                "snapshot": {},
                "results": [{"tool": "check", "asset_id": "old"}],
            }
        )


def test_plan_rejects_unpaired_or_ungrounded_conditions():
    with pytest.raises(ValueError):
        Plan(tools=["transfer"], conditions=["if_mismatch"], summary="bad")
    with pytest.raises(ValueError):
        Plan(tools=["check", "repair"], conditions=["always"], summary="bad")
    assert Plan(tools=[], conditions=["always"], summary="普通交流").tools == ["chat"]
    assert Plan(tools=[], summary="普通交流").conditions == ["always"]
    with pytest.raises(ValueError):
        Plan(tools=[], conditions=["if_mismatch"], summary="bad")


def test_invalid_model_output_is_retry_data_and_never_promoted_to_system_rules(monkeypatch):
    _, _, task_id, _ = seed_history()
    calls = []

    def generate(_snapshot, messages, **_kwargs):
        calls.append([dict(message) for message in messages])
        return "UNTRUSTED_RETRY_INSTRUCTION" if len(calls) == 1 else '{"tools":["chat"],"summary":"交流"}'

    monkeypatch.setattr("endlexx.agent.generate", generate)
    result = plan_node({"task_id": task_id, "snapshot": {}, "query": "你好"})
    assert result["plan"] == ["chat"]
    assert "UNTRUSTED_RETRY_INSTRUCTION" not in calls[1][0]["content"]
    correction = json.loads(calls[1][1]["content"])
    assert correction["request"] == "你好"
    assert correction["previous_output"] == "UNTRUSTED_RETRY_INSTRUCTION"
    assert correction["validation_error"]


def test_explicit_branch_cannot_silently_finish_after_only_checking():
    request = "先检查外观；只有确定不符合时才做角色迁移。若无法判断就停止。"
    with pytest.raises(ValueError, match="不能只生成check"):
        validate_requested_branch(Plan(tools=["check"], summary="check"), request)
    with pytest.raises(ValueError, match="stop_on_uncertainty"):
        validate_requested_branch(
            Plan(tools=["check", "transfer"], conditions=["always", "if_mismatch"], summary="branch"), request
        )
    validate_requested_branch(
        Plan(
            tools=["check", "transfer"],
            conditions=["always", "if_mismatch"],
            stop_on_uncertainty=True,
            summary="branch",
        ),
        request,
    )
    validate_requested_branch(
        Plan(tools=["check"], summary="check"), "检查图片，说明“如果不符就迁移”的含义，不要修改。"
    )
    validate_requested_branch(Plan(tools=["check"], summary="check"), "检查图片，只有确定不符时才不要迁移。")


def test_summary_overshoot_retries_without_cutting_a_negation_or_advancing_boundary(monkeypatch):
    chat_id, message_id, task_id, rows = seed_history()
    calls = []

    def summarize(_snapshot, messages, **_kwargs):
        calls.append(json.loads(messages[-1]["content"]))
        return '{"keep":[999]}'

    monkeypatch.setattr("endlexx.context.generate", summarize)
    with pytest.raises(ValueError, match="摘要校验未通过"):
        build_context(chat_id, message_id, {}, task_id)
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    with Session() as db:
        assert db.get(ChatContext, chat_id) is None
        assert db.get(Task, task_id).result == {}
    for chunk in history_chunks(rows):
        for line in chunk.splitlines():
            value = json.loads(line)
            assert value["role"] in {"user", "assistant"}
            assert value["part"] <= value["parts"]


def test_extractive_summary_keeps_negation_and_exact_filename_and_rejects_oversize(monkeypatch):
    monkeypatch.setattr("endlexx.context.generate", lambda *_args, **_kwargs: '{"keep":[0,1]}')
    chunk = json.dumps(
        {"role": "user", "content": "不要自动修改。文件名draft_blue_2026.png。"}, ensure_ascii=False
    )
    summary = compact_summary({}, "", chunk, 256)
    assert summary == "用户：不要自动修改。\n用户：文件名draft_blue_2026.png。"
    with pytest.raises(ValueError, match="超出预算"):
        compact_summary({}, "", chunk, 20)


def test_fitting_user_facts_cannot_be_omitted_by_memory_selector(monkeypatch):
    monkeypatch.setattr("endlexx.context.generate", lambda *_args, **_kwargs: '{"keep":[0]}')
    chunk = json.dumps(
        {"role": "user", "content": "生日没有确认。我们只确认了琥珀色眼睛。"}, ensure_ascii=False
    )
    summary = compact_summary({}, "", chunk, 256)
    assert "生日没有确认" in summary
    assert "琥珀色眼睛" in summary

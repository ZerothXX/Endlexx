from types import SimpleNamespace

import pytest

from scripts.model_runtime import ContextBudgetExceeded, fit_context


def test_token_budget_drops_old_turns_and_preserves_rules_and_current_request():
    class Processor:
        def apply_chat_template(self, messages, **_):
            return "|".join(message["content"] for message in messages)

        def __call__(self, text, **_):
            return {"input_ids": SimpleNamespace(shape=(1, len(text[0]))), "text": text[0]}

    messages = [
        {"role": "system", "content": "RULES"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
    ]
    inputs, dropped = fit_context(Processor(), messages, [], limit=15)
    assert inputs["text"] == "RULES|latest"
    assert dropped == 2
    assert len(messages) == 4
    with pytest.raises(ContextBudgetExceeded, match="上下文预算"):
        fit_context(Processor(), messages, [], limit=5)
    # Managed conversations must summarize at the application layer, never silently drop turns.
    with pytest.raises(ContextBudgetExceeded, match="压缩"):
        fit_context(Processor(), messages, [], limit=15, preserve_history=True)

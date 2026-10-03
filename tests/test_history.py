"""Tests for follow-up question rewriting. No real model calls."""
from __future__ import annotations

from types import SimpleNamespace

from src.rag.history import rewrite_with_history


class _FakeCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, reply: str):
        self.completions = _FakeCompletions(reply)
        self.chat = SimpleNamespace(completions=self.completions)


def test_no_history_returns_question_without_calling_model():
    client = _FakeClient("should not be used")

    result = rewrite_with_history(client, "m", "Who is eligible?", None)

    assert result == "Who is eligible?"
    assert client.completions.calls == 0


def test_empty_history_also_skips_the_model():
    client = _FakeClient("should not be used")

    result = rewrite_with_history(client, "m", "Who is eligible?", [])

    assert result == "Who is eligible?"
    assert client.completions.calls == 0


def test_follow_up_is_rewritten_using_history():
    client = _FakeClient("Is a widow eligible for PM-KISAN?")
    history = [
        {"role": "user", "content": "Who is eligible for PM-KISAN?"},
        {"role": "assistant", "content": "Landholding farmer families..."},
    ]

    result = rewrite_with_history(client, "m", "what about widows?", history)

    assert result == "Is a widow eligible for PM-KISAN?"
    assert client.completions.calls == 1


def test_empty_model_output_falls_back_to_original_question():
    client = _FakeClient("   ")
    history = [{"role": "user", "content": "PM-KISAN?"}]

    result = rewrite_with_history(client, "m", "and for PMAY-G?", history)

    assert result == "and for PMAY-G?"

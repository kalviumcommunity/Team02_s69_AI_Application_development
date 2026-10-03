"""Tests for the unanswered-query and feedback logs."""
from __future__ import annotations

import pytest

from src.query_log import log_feedback, log_unanswered, question_hash


def test_unanswered_log_writes_header_then_rows(tmp_path):
    path = tmp_path / "unanswered.csv"

    log_unanswered("What is the capital of France?", 0.12, path=path)
    log_unanswered("Weather tomorrow?", None, path=path)

    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "timestamp,query,top_score"
    assert len(lines) == 3
    assert "What is the capital of France?" in lines[1]
    assert lines[2].endswith(",")  # no score available


def test_feedback_stores_hash_not_question_text(tmp_path):
    path = tmp_path / "feedback.csv"
    question = "Is my family eligible for PM-KISAN?"

    log_feedback(question, "up", path=path)

    content = path.read_text(encoding="utf-8")

    assert question not in content
    assert question_hash(question) in content
    assert content.splitlines()[0] == "timestamp,question_hash,rating"


def test_feedback_rejects_unknown_rating(tmp_path):
    with pytest.raises(ValueError):
        log_feedback("q", "meh", path=tmp_path / "f.csv")


def test_question_hash_ignores_case_and_surrounding_space():
    assert question_hash("  PM-KISAN? ") == question_hash("pm-kisan?")

"""Tests for the evaluation script's logic, with no real API calls."""
from __future__ import annotations

from pathlib import Path

from evaluation.run_eval import (
    QUESTIONS_FILE,
    evaluate_refusal,
    evaluate_retrieval,
    load_questions,
)


class FakeVectorStore:
    """Returns fixed results regardless of query, for a deterministic test."""

    def __init__(self, scheme_ids):
        self._scheme_ids = scheme_ids

    def search(self, query, top_k=5):
        return [
            {"metadata": {"scheme_id": scheme_id}}
            for scheme_id in self._scheme_ids
        ]


def test_questions_file_exists_and_parses():
    assert QUESTIONS_FILE.exists()

    questions = load_questions()
    assert questions

    required_fields = {
        "question_id",
        "question_text",
        "expected_scheme_id",
        "expected_citation",
        "category",
    }
    for question in questions:
        assert required_fields.issubset(question.keys())


def test_questions_cover_expected_categories():
    questions = load_questions()
    categories = {q["category"] for q in questions}

    assert "clear-eligible" in categories
    assert "clear-ineligible" in categories
    assert "ambiguous" in categories
    assert "out-of-corpus" in categories


def test_questions_cover_at_least_one_per_scheme():
    questions = load_questions()

    with open(
        Path(__file__).resolve().parents[1] / "data" / "schemes.json",
        encoding="utf-8",
    ) as file:
        import json

        scheme_ids = {s["scheme_id"] for s in json.load(file)}

    covered = {
        q["expected_scheme_id"]
        for q in questions
        if q["expected_scheme_id"]
    }

    assert scheme_ids.issubset(covered)


def test_out_of_corpus_questions_have_no_expected_scheme():
    questions = load_questions()

    for question in questions:
        if question["category"] == "out-of-corpus":
            assert question["expected_scheme_id"] == ""


def test_evaluate_retrieval_hit():
    question = {
        "question_id": "q1",
        "question_text": "test",
        "category": "clear-eligible",
        "expected_scheme_id": "pmkisan",
    }
    store = FakeVectorStore(["pmay_g", "pmkisan", "pmegp"])

    result = evaluate_retrieval(question, store)

    assert result["hit"] is True
    assert result["retrieved_scheme_ids"] == ["pmay_g", "pmkisan", "pmegp"]


def test_evaluate_retrieval_miss():
    question = {
        "question_id": "q1",
        "question_text": "test",
        "category": "clear-eligible",
        "expected_scheme_id": "pmkisan",
    }
    store = FakeVectorStore(["pmay_g", "pmegp"])

    result = evaluate_retrieval(question, store)

    assert result["hit"] is False


def test_evaluate_refusal_detects_refusal(monkeypatch):
    question = {
        "question_id": "q10",
        "question_text": "What is the capital of France?",
        "category": "out-of-corpus",
    }

    monkeypatch.setattr(
        "evaluation.run_eval.answer_question",
        lambda query: {"status": "insufficient_information"},
    )

    result = evaluate_refusal(question)
    assert result["refused"] is True


def test_evaluate_refusal_detects_a_wrongly_answered_question(monkeypatch):
    question = {
        "question_id": "q10",
        "question_text": "What is the capital of France?",
        "category": "out-of-corpus",
    }

    monkeypatch.setattr(
        "evaluation.run_eval.answer_question",
        lambda query: {"status": "answered"},
    )

    result = evaluate_refusal(question)
    assert result["refused"] is False

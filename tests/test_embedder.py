"""Tests for the embedder's response handling. No real API calls.

Some OpenAI-compatible providers (Gemini, for one) return embedding items
with no usable ``index`` field, so the embedder must fall back to response
order instead of crashing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.indexing.embedder import OpenAIEmbedder


class _FakeEmbeddings:
    def __init__(self, data):
        self._data = data

    def create(self, model, input):  # noqa: A002 - matches SDK signature
        return SimpleNamespace(data=self._data)


def _embedder(monkeypatch, data) -> OpenAIEmbedder:
    monkeypatch.setattr("src.indexing.embedder.EMBEDDING_API_KEY", "fake-key")
    embedder = OpenAIEmbedder(min_interval_seconds=0)
    embedder.client = SimpleNamespace(embeddings=_FakeEmbeddings(data))
    return embedder


def test_sorts_by_index_when_every_item_has_one(monkeypatch):
    data = [
        SimpleNamespace(embedding=[2.0], index=1),
        SimpleNamespace(embedding=[1.0], index=0),
    ]

    result = _embedder(monkeypatch, data).embed(["a", "b"])

    assert result == [[1.0], [2.0]]


def test_falls_back_to_response_order_when_index_is_none(monkeypatch):
    data = [
        SimpleNamespace(embedding=[1.0], index=None),
        SimpleNamespace(embedding=[2.0], index=None),
    ]

    result = _embedder(monkeypatch, data).embed(["a", "b"])

    assert result == [[1.0], [2.0]]


def test_falls_back_when_index_attribute_is_missing(monkeypatch):
    data = [SimpleNamespace(embedding=[1.0]), SimpleNamespace(embedding=[2.0])]

    result = _embedder(monkeypatch, data).embed(["a", "b"])

    assert result == [[1.0], [2.0]]


def test_empty_input_returns_empty_list(monkeypatch):
    assert _embedder(monkeypatch, []).embed([]) == []


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.setattr("src.indexing.embedder.EMBEDDING_API_KEY", "")

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        OpenAIEmbedder()

"""Tests for the end-to-end pipeline entrypoint.

These tests stub out the ingestion, chunking and indexing steps so they run
instantly and never touch real PDFs, the filesystem layout, or the
embeddings API.
"""
from __future__ import annotations

import src.pipeline as pipeline


def test_skip_index_runs_ingestion_and_chunking_only(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline, "run_ingestion", lambda: calls.append("ingestion")
    )
    monkeypatch.setattr(
        pipeline, "run_chunking", lambda: calls.append("chunking")
    )
    monkeypatch.setattr(
        pipeline, "run_build_index", lambda: calls.append("index")
    )

    pipeline.main(["--skip-index"])

    assert calls == ["ingestion", "chunking"]


def test_default_run_includes_indexing(monkeypatch):
    calls = []

    monkeypatch.setattr(
        pipeline, "run_ingestion", lambda: calls.append("ingestion")
    )
    monkeypatch.setattr(
        pipeline, "run_chunking", lambda: calls.append("chunking")
    )
    monkeypatch.setattr(
        pipeline, "run_build_index", lambda: calls.append("index")
    )

    pipeline.main([])

    assert calls == ["ingestion", "chunking", "index"]


def test_steps_run_in_order_even_if_registered_out_of_order(monkeypatch):
    # Guards against a future refactor silently reordering the pipeline.
    calls = []

    monkeypatch.setattr(
        pipeline, "run_build_index", lambda: calls.append("index")
    )
    monkeypatch.setattr(
        pipeline, "run_ingestion", lambda: calls.append("ingestion")
    )
    monkeypatch.setattr(
        pipeline, "run_chunking", lambda: calls.append("chunking")
    )

    pipeline.main([])

    assert calls == ["ingestion", "chunking", "index"]


def test_parse_args_defaults_to_full_run():
    args = pipeline.parse_args([])
    assert args.skip_index is False


def test_parse_args_skip_index_flag():
    args = pipeline.parse_args(["--skip-index"])
    assert args.skip_index is True

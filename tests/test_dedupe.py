"""Tests for chunk de-duplication and single-scheme loading."""
from __future__ import annotations

import json

from src.indexing.build_index import dedupe_chunks, load_chunks


def _chunk(scheme_id: str, text: str, chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "scheme_id": scheme_id,
        "section_title": "",
        "page_number": 1,
        "text": text,
    }


def test_dedupe_removes_same_text_within_one_scheme():
    chunks = [
        _chunk("pmkisan", "Same boilerplate.", "a"),
        _chunk("pmkisan", "Same   boilerplate.", "b"),  # whitespace/case
    ]

    unique, removed = dedupe_chunks(chunks)

    assert [c["chunk_id"] for c in unique] == ["a"]
    assert removed == 1


def test_dedupe_keeps_identical_text_across_different_schemes():
    chunks = [
        _chunk("pmkisan", "Shared wording.", "a"),
        _chunk("pmay_g", "Shared wording.", "b"),
    ]

    unique, removed = dedupe_chunks(chunks)

    assert len(unique) == 2
    assert removed == 0


def test_load_chunks_filters_to_one_scheme(tmp_path):
    schemes = tmp_path / "schemes.json"
    schemes.write_text(json.dumps([
        {"scheme_id": "a", "scheme_name": "A", "department": "D"},
        {"scheme_id": "b", "scheme_name": "B", "department": "D"},
    ]), encoding="utf-8")

    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        json.dumps(_chunk("a", "text a", "a1")) + "\n"
        + json.dumps(_chunk("b", "text b", "b1")) + "\n",
        encoding="utf-8",
    )

    only_a = load_chunks(chunks_file, schemes, scheme_id="a")

    assert [c["chunk_id"] for c in only_a] == ["a1"]
    assert only_a[0]["scheme_name"] == "A"

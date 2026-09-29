"""Tests for the Streamlit app's helper logic.

Importing app.main directly runs the whole script in Streamlit's "bare
mode" (no real browser session) -- this alone confirms the file executes
top to bottom without raising, including both tab-rendering functions.
Streamlit's bare-mode warnings on stderr are expected and harmless.
"""
from __future__ import annotations

from app.main import _is_cited


def _chunk(text: str, scheme_name: str, page: int) -> dict:
    return {
        "chunk": text,
        "score": 0.9,
        "metadata": {
            "scheme_id": "test",
            "scheme_name": scheme_name,
            "department": "Test",
            "section_title": "ELIGIBILITY",
            "page_number": page,
        },
    }


def test_is_cited_true_for_exact_match():
    chunk = _chunk("PM-KISAN eligibility text.", "PM-KISAN", 3)
    citations = [
        {
            "scheme_name": "PM-KISAN",
            "page": 3,
            "section": "ELIGIBILITY",
            "snippet": "PM-KISAN eligibility text.",
        }
    ]

    assert _is_cited(chunk, citations) is True


def test_is_cited_false_when_not_referenced():
    chunk = _chunk("Unrelated retrieved text.", "PM-KISAN", 3)
    citations = [
        {
            "scheme_name": "PM-KISAN",
            "page": 3,
            "section": "ELIGIBILITY",
            "snippet": "A different chunk's text.",
        }
    ]

    assert _is_cited(chunk, citations) is False


def test_is_cited_false_with_no_citations():
    chunk = _chunk("Some text.", "PM-KISAN", 3)
    assert _is_cited(chunk, []) is False


def test_is_cited_requires_matching_page_not_just_text():
    # Same text on a different page should not count as cited -- guards
    # against a false positive if two schemes ever share wording.
    chunk = _chunk("Shared boilerplate text.", "PM-KISAN", 5)
    citations = [
        {
            "scheme_name": "PM-KISAN",
            "page": 3,
            "section": "ELIGIBILITY",
            "snippet": "Shared boilerplate text.",
        }
    ]

    assert _is_cited(chunk, citations) is False

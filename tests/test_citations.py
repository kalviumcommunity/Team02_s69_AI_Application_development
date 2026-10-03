"""Tests for citation presentation helpers."""
from __future__ import annotations

from src.rag.citations import MAX_SECTION_LENGTH, display_section, unique_citations


def test_long_heading_sentence_is_not_shown_as_section():
    long_heading = "5.3.4 The State Governments have to put in place the administrative mechanism"

    assert display_section(long_heading) == ""


def test_short_section_is_kept():
    assert display_section("ELIGIBILITY") == "ELIGIBILITY"


def test_missing_section_is_empty():
    assert display_section(None) == ""
    assert display_section("") == ""


def test_section_at_the_limit_is_kept():
    section = "x" * MAX_SECTION_LENGTH

    assert display_section(section) == section


def test_unique_citations_keeps_one_per_scheme_and_page():
    citations = [
        {"scheme_name": "PM-KISAN", "page": 2, "section": "A", "snippet": "a"},
        {"scheme_name": "PM-KISAN", "page": 2, "section": "B", "snippet": "b"},
        {"scheme_name": "PM-KISAN", "page": 3, "section": "C", "snippet": "c"},
        {"scheme_name": "PMAY-G", "page": 2, "section": "D", "snippet": "d"},
    ]

    unique = unique_citations(citations)

    assert [(c["scheme_name"], c["page"]) for c in unique] == [
        ("PM-KISAN", 2),
        ("PM-KISAN", 3),
        ("PMAY-G", 2),
    ]
    assert unique[0]["section"] == "A"

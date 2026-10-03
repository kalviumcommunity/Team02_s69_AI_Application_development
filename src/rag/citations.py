"""Helpers for presenting citations to the user."""

from __future__ import annotations

# Heading detection sometimes picks up a whole sentence as a "section".
# Anything longer than this is not shown as a section label.
MAX_SECTION_LENGTH = 60


def display_section(section: str | None) -> str:
    """A section label worth showing, or '' if it is missing or too long."""
    if not section:
        return ""

    section = section.strip()

    return section if len(section) <= MAX_SECTION_LENGTH else ""


def unique_citations(citations: list[dict]) -> list[dict]:
    """Keep one citation per scheme and page, in the order given.

    Several retrieved chunks can come from the same page; listing that page
    twice adds nothing for the reader.
    """
    seen: set[tuple[str, int]] = set()
    unique: list[dict] = []

    for citation in citations:
        key = (citation["scheme_name"], citation["page"])

        if key in seen:
            continue

        seen.add(key)
        unique.append(citation)

    return unique

"""Guardrails for insufficient-information RAG responses."""

from __future__ import annotations


def is_low_confidence(
    chunks: list[dict],
    threshold: float,
) -> bool:
    """Return True when retrieved context is below the confidence threshold."""
    if not chunks:
        return True

    top_score = chunks[0].get("score")

    if top_score is None:
        return True

    return top_score < threshold

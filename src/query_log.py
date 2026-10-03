"""Append-only CSV logs for the Program Officer's review.

- unanswered_queries.csv: questions the assistant refused, so the content
  owners can see where the source documents fall short.
- feedback.csv: thumbs up/down on answers. Stores a hash of the question,
  not its text.

Neither log stores names, IDs or anything typed into the eligibility form.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOGS_DIR


UNANSWERED_LOG = LOGS_DIR / "unanswered_queries.csv"
FEEDBACK_LOG = LOGS_DIR / "feedback.csv"

UNANSWERED_HEADER = ["timestamp", "query", "top_score"]
FEEDBACK_HEADER = ["timestamp", "question_hash", "rating"]


def _append(path: Path, header: list[str], row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(header)

        writer.writerow(row)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def question_hash(question: str) -> str:
    """A short, stable fingerprint of a question for feedback rows."""
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:12]


def log_unanswered(
    query: str,
    top_score: float | None,
    path: Path = UNANSWERED_LOG,
) -> None:
    """Record a question the assistant could not answer from its documents."""
    _append(
        path,
        UNANSWERED_HEADER,
        [_now(), query, "" if top_score is None else f"{top_score:.4f}"],
    )


def log_feedback(
    question: str,
    rating: str,
    path: Path = FEEDBACK_LOG,
) -> None:
    """Record a thumbs-up ("up") or thumbs-down ("down") on an answer."""
    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")

    _append(
        path,
        FEEDBACK_HEADER,
        [_now(), question_hash(question), rating],
    )

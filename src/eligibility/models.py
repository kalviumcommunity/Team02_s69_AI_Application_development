"""Typed models for the profile and eligibility results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """The three possible checker outcomes."""

    ELIGIBLE = "Eligible"
    UNCLEAR = "Unclear"
    NOT_ELIGIBLE = "Not Eligible"


@dataclass
class UserProfile:
    """Facts collected from a citizen; unknown values remain ``None``."""

    income: float | None = None
    land_acres: float | None = None
    category: str | None = None
    state: str | None = None
    age: int | None = None


@dataclass
class SchemeResult:
    """Eligibility verdict and the evidence or uncertainty behind it."""

    scheme_name: str
    verdict: Verdict
    failing_rule: dict | None = None
    missing_fields: list[str] = field(default_factory=list)
    citation: str | None = None
    unresolved_rules: list[str] = field(default_factory=list)
    cited_rules: list[dict] = field(default_factory=list)

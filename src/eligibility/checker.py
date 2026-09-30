"""Generic JSON-driven eligibility rule evaluation."""

from __future__ import annotations

import json
import operator
from typing import Any

from src.config import DATA_DIR
from src.eligibility.models import SchemeResult, UserProfile, Verdict


RULES_DIR = DATA_DIR / "rules"
SCHEMES_FILE = DATA_DIR / "schemes.json"
PROFILE_FIELDS = {
    "income": "income",
    "age": "age",
    "land": "land_acres",
    "category": "category",
    "location": "state",
}
COMPARATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _threshold(value: Any, rule: dict) -> Any:
    """Parse rule thresholds, converting supported measures to profile units.

    The form records income as annual rupees and land in acres. Rules may
    express income thresholds per month or year and land in acres/hectares.
    """
    unit = str(rule.get("unit", "")).strip().casefold()
    boolean_value = _parse_bool(value)
    if unit == "boolean":
        return boolean_value

    if rule["rule_type"] in {"income", "age", "land"}:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if rule["rule_type"] == "income":
            if unit in {"inr/month", "rs/month", "rupees/month"}:
                return number * 12
            if unit in {"inr/year", "inr/annum", "rs/year", "rs/annum", "inr"}:
                return number
            return None
        if rule["rule_type"] == "land":
            if unit in {"acres", "acre"}:
                return number
            if unit in {"hectares", "hectare", "ha"}:
                return number * 2.4710538147
            return None
        if unit in {"years", "year", "age"}:
            return number
        return None

    if rule["rule_type"] in {"category", "location"}:
        if isinstance(value, list):
            return [str(item).strip().casefold() for item in value]
        return str(value).strip().casefold()
    return None


def _evaluate(profile_value: Any, rule: dict) -> bool | None:
    """Return pass/fail, or ``None`` when this profile cannot decide it."""
    threshold = _threshold(rule.get("threshold_value"), rule)
    if threshold is None:
        return None

    if rule.get("unit", "").casefold() == "boolean":
        profile_value = _parse_bool(profile_value)
        if profile_value is None:
            return None
    elif rule["rule_type"] in {"income", "age", "land"}:
        try:
            profile_value = float(profile_value)
        except (TypeError, ValueError):
            return None
    elif rule["rule_type"] in {"category", "location"}:
        if isinstance(profile_value, str):
            profile_value = profile_value.strip().casefold()
        else:
            return None

    op = rule.get("operator")
    if op in COMPARATORS:
        try:
            return bool(COMPARATORS[op](profile_value, threshold))
        except TypeError:
            return None
    if op == "in":
        options = threshold if isinstance(threshold, list) else [threshold]
        return profile_value in options
    if op == "not_in":
        options = threshold if isinstance(threshold, list) else [threshold]
        return profile_value not in options
    return None


def _citation(rule: dict | None, scheme_name: str) -> str | None:
    if not rule or not isinstance(rule.get("source_page"), int):
        return None
    return f"{scheme_name}, page {rule['source_page']}"


def _check_scheme(profile: UserProfile, scheme: dict, data: dict) -> SchemeResult:
    scheme_name = scheme["scheme_name"]
    failures: list[dict] = []
    passed: list[dict] = []
    missing_fields: set[str] = set()
    unresolved: list[str] = []
    unclear_citation: str | None = None
    # Notes record important eligibility requirements that the profile
    # cannot assess. They prevent an otherwise passing partial rule set
    # from being presented as a definitive Eligible verdict.
    unresolved.extend(data.get("notes", []))

    for rule in data.get("rules", []):
        if rule.get("scheme_id") != data.get("scheme_id"):
            unresolved.append(f"{rule.get('rule_id', 'Rule')} has a mismatched scheme id")
            continue

        field_name = PROFILE_FIELDS.get(rule.get("rule_type"))
        if field_name is None:
            unresolved.append(f"{rule.get('rule_id', 'Rule')} uses an unsupported rule type")
            continue

        profile_value = getattr(profile, field_name)
        if profile_value is None:
            missing_fields.add(field_name)
            if unclear_citation is None:
                unclear_citation = _citation(rule, scheme_name)
            continue

        if not rule.get("verified", False):
            unresolved.append(f"{rule.get('rule_id', 'Rule')} is not source-verified")
            if unclear_citation is None:
                unclear_citation = _citation(rule, scheme_name)
            continue

        result = _evaluate(profile_value, rule)
        if result is None:
            unresolved.append(
                f"{rule.get('rule_id', 'Rule')} needs information the profile cannot represent"
            )
            if unclear_citation is None:
                unclear_citation = _citation(rule, scheme_name)
        elif not result:
            failures.append(rule)
        else:
            passed.append(rule)

    # A proven exclusion takes precedence over missing or unsupported facts.
    if failures:
        failed = failures[0]
        return SchemeResult(
            scheme_name=scheme_name,
            verdict=Verdict.NOT_ELIGIBLE,
            failing_rule=failed,
            citation=_citation(failed, scheme_name),
            missing_fields=sorted(missing_fields),
            unresolved_rules=unresolved,
            cited_rules=[failed],
        )

    if missing_fields or unresolved or not data.get("rules"):
        if not data.get("rules") and not unresolved:
            unresolved.append("No profile-evaluable verified rules are available")
        return SchemeResult(
            scheme_name=scheme_name,
            verdict=Verdict.UNCLEAR,
            missing_fields=sorted(missing_fields),
            citation=unclear_citation or next(
                (_citation(rule, scheme_name) for rule in passed
                 if _citation(rule, scheme_name)),
                None,
            ),
            unresolved_rules=unresolved,
            cited_rules=[rule for rule in passed if _citation(rule, scheme_name)],
        )

    citations = [
        _citation(rule, scheme_name)
        for rule in passed
        if _citation(rule, scheme_name)
    ]
    return SchemeResult(
        scheme_name=scheme_name,
        verdict=Verdict.ELIGIBLE,
        citation="; ".join(citations) if citations else None,
        cited_rules=passed,
    )


def check_all(profile: UserProfile) -> list[SchemeResult]:
    """Evaluate every rule file and rank results by verdict and uncertainty."""
    with SCHEMES_FILE.open(encoding="utf-8") as file:
        schemes = {item["scheme_id"]: item for item in json.load(file)}

    results: list[SchemeResult] = []
    for path in sorted(RULES_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        scheme_id = data["scheme_id"]
        scheme = schemes.get(scheme_id, {"scheme_name": scheme_id})
        results.append(_check_scheme(profile, scheme, data))

    rank = {
        Verdict.ELIGIBLE: 0,
        Verdict.UNCLEAR: 1,
        Verdict.NOT_ELIGIBLE: 2,
    }
    results.sort(key=lambda item: (
        rank[item.verdict],
        len(item.missing_fields) if item.verdict == Verdict.UNCLEAR else 0,
        item.scheme_name.casefold(),
    ))
    return results

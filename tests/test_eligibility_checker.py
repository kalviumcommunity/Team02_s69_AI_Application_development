"""Tests for source-backed eligibility evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.eligibility import checker
from src.eligibility.checker import _evaluate
from src.eligibility.models import UserProfile, Verdict


RULES_DIR = Path(__file__).resolve().parents[1] / "data" / "rules"


def _result_for(profile: UserProfile, scheme_id: str):
    return next(
        result for result in checker.check_all(profile)
        if result.scheme_name == {"pmay_g": "PMAY-G", "pmkisan": "PM-KISAN"}[scheme_id]
    )


def _write_rules(directory: Path, scheme_id: str, rules: list[dict]) -> None:
    directory.joinpath(f"{scheme_id}.json").write_text(
        json.dumps({"scheme_id": scheme_id, "rules": rules, "notes": []}),
        encoding="utf-8",
    )


def _rule(
    rule_id: str,
    scheme_id: str,
    rule_type: str,
    operator: str,
    threshold: str,
    unit: str,
    page: int,
    verified: bool = True,
) -> dict:
    return {
        "rule_id": rule_id,
        "scheme_id": scheme_id,
        "rule_type": rule_type,
        "condition": f"{rule_type} {operator} {threshold}",
        "operator": operator,
        "threshold_value": threshold,
        "unit": unit,
        "source_page": page,
        "verified": verified,
    }


def test_income_above_existing_verified_threshold_fails_with_citation():
    # PMAY-G's checked rule excludes a family member earning over 10,000/month.
    result = _result_for(UserProfile(income=120001), "pmay_g")

    assert result.verdict is Verdict.NOT_ELIGIBLE
    assert result.failing_rule["rule_id"] == "pmayg-income-01"
    assert result.citation == "PMAY-G, page 92"


def test_income_below_existing_threshold_does_not_fail_that_rule():
    result = _result_for(UserProfile(income=119999), "pmay_g")

    # Other PMAY-G criteria are not represented by these profile fields, so
    # passing the income rule alone correctly leaves the overall verdict Unclear.
    assert result.verdict is Verdict.UNCLEAR
    assert result.failing_rule is None
    assert "income" not in result.missing_fields
    assert result.citation == "PMAY-G, page 92"


def test_missing_required_field_is_unclear_and_reported():
    result = _result_for(UserProfile(), "pmay_g")

    assert result.verdict is Verdict.UNCLEAR
    assert {"income", "land_acres", "category"}.issubset(result.missing_fields)
    assert result.citation == "PMAY-G, page 92"


def test_current_check_only_returns_schemes_with_rule_files_present():
    results = checker.check_all(UserProfile())
    available_ids = {path.stem for path in RULES_DIR.glob("*.json")}

    assert len(results) == len(available_ids) == 2
    assert {"PM-KISAN", "PMAY-G"} == {result.scheme_name for result in results}


def test_missing_values_are_not_coerced_to_zero_or_false():
    result = _result_for(UserProfile(), "pmkisan")

    assert result.verdict is Verdict.UNCLEAR
    assert {"income", "land_acres", "category"}.issubset(result.missing_fields)
    assert result.failing_rule is None


def test_social_category_does_not_satisfy_boolean_exclusion_rules():
    result = _result_for(UserProfile(category="SC"), "pmkisan")

    assert result.verdict is Verdict.UNCLEAR
    assert result.unresolved_rules
    assert result.failing_rule is None


def test_results_rank_eligible_unclear_then_not_eligible(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schemes_path = tmp_path / "schemes.json"
    schemes_path.write_text(
        json.dumps([
            {"scheme_id": "a", "scheme_name": "Eligible scheme"},
            {"scheme_id": "b", "scheme_name": "Unclear scheme"},
            {"scheme_id": "d", "scheme_name": "More unclear scheme"},
            {"scheme_id": "c", "scheme_name": "Not eligible scheme"},
        ]),
        encoding="utf-8",
    )
    _write_rules(rules_dir, "a", [_rule("a1", "a", "age", ">", "18", "years", 1)])
    _write_rules(
        rules_dir, "b", [_rule("b1", "b", "location", "==", "Karnataka", "state", 2)]
    )
    _write_rules(rules_dir, "d", [
        _rule("d1", "d", "location", "==", "Karnataka", "state", 4),
        _rule("d2", "d", "category", "==", "SC", "category", 5),
    ])
    _write_rules(
        rules_dir, "c", [_rule("c1", "c", "income", "<=", "50000", "INR/year", 3)]
    )

    with patch.object(checker, "RULES_DIR", rules_dir), patch.object(
        checker, "SCHEMES_FILE", schemes_path
    ):
        results = checker.check_all(UserProfile(income=60000, age=20))

    assert [result.verdict for result in results] == [
        Verdict.ELIGIBLE,
        Verdict.UNCLEAR,
        Verdict.UNCLEAR,
        Verdict.NOT_ELIGIBLE,
    ]
    assert [result.missing_fields for result in results[1:3]] == [
        ["state"], ["category", "state"]
    ]


def test_additional_rule_files_are_discovered_dynamically(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schemes_path = tmp_path / "schemes.json"
    schemes = [
        {"scheme_id": "apy", "scheme_name": "Test APY"},
        {"scheme_id": "pmjay", "scheme_name": "Test PM-JAY"},
        {"scheme_id": "future", "scheme_name": "Future scheme"},
    ]
    schemes_path.write_text(json.dumps(schemes), encoding="utf-8")
    _write_rules(rules_dir, "apy", [_rule("a1", "apy", "age", ">=", "18", "years", 1)])
    _write_rules(
        rules_dir, "pmjay", [_rule("j1", "pmjay", "income", "<=", "10000", "INR/year", 2)]
    )
    _write_rules(
        rules_dir,
        "future",
        [_rule("f1", "future", "age", ">", "18", "years", 3, verified=False)],
    )

    with patch.object(checker, "RULES_DIR", rules_dir), patch.object(
        checker, "SCHEMES_FILE", schemes_path
    ):
        results = checker.check_all(UserProfile(income=5000, age=25))

    assert len(results) == 3
    by_name = {result.scheme_name: result for result in results}
    assert by_name["Test APY"].verdict is Verdict.ELIGIBLE
    assert by_name["Test PM-JAY"].verdict is Verdict.ELIGIBLE
    assert by_name["Future scheme"].verdict is Verdict.UNCLEAR


def test_unverified_rule_is_unclear_with_its_citation(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schemes_path = tmp_path / "schemes.json"
    schemes_path.write_text(
        json.dumps([{"scheme_id": "future", "scheme_name": "Future scheme"}]),
        encoding="utf-8",
    )
    _write_rules(
        rules_dir,
        "future",
        [_rule("f1", "future", "age", ">", "18", "years", 7, verified=False)],
    )

    with patch.object(checker, "RULES_DIR", rules_dir), patch.object(
        checker, "SCHEMES_FILE", schemes_path
    ):
        result = checker.check_all(UserProfile(age=25))[0]

    assert result.verdict is Verdict.UNCLEAR
    assert result.citation == "Future scheme, page 7"
    assert result.unresolved_rules


def test_supported_comparison_operators_and_boolean_parsing():
    rule = {
        "rule_type": "income",
        "unit": "INR/year",
        "operator": "<=",
        "threshold_value": "100000",
    }
    assert _evaluate(100000, rule) is True
    assert _evaluate(100001, rule) is False

    boolean_rule = {
        "rule_type": "category",
        "unit": "boolean",
        "operator": "==",
        "threshold_value": "false",
    }
    assert _evaluate(False, boolean_rule) is True
    assert _evaluate("SC", boolean_rule) is None

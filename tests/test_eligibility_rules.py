"""Schema validation for the hand-written eligibility rule files.

These are reviewed data files (data/rules/*.json), not generated output,
but they still need a structural guarantee: every rule has an allowed
rule_type, every verified rule cites the source page it was checked
against, and the operator/threshold pair is usable by a checker.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RULES_DIR = Path(__file__).resolve().parents[1] / "data" / "rules"

ALLOWED_RULE_TYPES = {"income", "age", "land", "category", "location"}
ALLOWED_OPERATORS = {"<", "<=", ">", ">=", "==", "!=", "in", "not_in"}

REQUIRED_RULE_FIELDS = {
    "rule_id",
    "scheme_id",
    "rule_type",
    "condition",
    "operator",
    "threshold_value",
    "unit",
    "source_page",
    "verified",
}

RULE_FILES = sorted(RULES_DIR.glob("*.json"))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_at_least_one_rule_file_exists():
    assert RULE_FILES, f"No rule files found in {RULES_DIR}"


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_file_has_expected_top_level_shape(path: Path):
    data = _load(path)

    assert "scheme_id" in data
    assert "rules" in data and isinstance(data["rules"], list)
    assert data["rules"], f"{path.name} has no rules"
    assert "notes" in data and isinstance(data["notes"], list)


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_every_rule_has_required_fields(path: Path):
    data = _load(path)

    for rule in data["rules"]:
        missing = REQUIRED_RULE_FIELDS - rule.keys()
        assert not missing, f"{path.name}:{rule.get('rule_id')} missing {missing}"


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_rule_type_is_allowed(path: Path):
    data = _load(path)

    for rule in data["rules"]:
        assert rule["rule_type"] in ALLOWED_RULE_TYPES, (
            f"{path.name}:{rule['rule_id']} has invalid rule_type "
            f"{rule['rule_type']!r}"
        )


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_operator_is_allowed(path: Path):
    data = _load(path)

    for rule in data["rules"]:
        assert rule["operator"] in ALLOWED_OPERATORS, (
            f"{path.name}:{rule['rule_id']} has invalid operator "
            f"{rule['operator']!r}"
        )


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_verified_rules_cite_a_source_page(path: Path):
    data = _load(path)

    for rule in data["rules"]:
        if rule["verified"]:
            assert isinstance(rule["source_page"], int), (
                f"{path.name}:{rule['rule_id']} is verified but "
                f"source_page is not an int: {rule['source_page']!r}"
            )
            assert rule["source_page"] > 0


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_rule_ids_are_unique_within_file(path: Path):
    data = _load(path)

    rule_ids = [rule["rule_id"] for rule in data["rules"]]
    assert len(rule_ids) == len(set(rule_ids))


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.stem)
def test_rule_scheme_id_matches_file_scheme_id(path: Path):
    data = _load(path)

    for rule in data["rules"]:
        assert rule["scheme_id"] == data["scheme_id"]


def test_rule_files_cover_the_two_schemes_this_ticket_asked_for():
    scheme_ids = {_load(path)["scheme_id"] for path in RULE_FILES}
    assert {"pmkisan", "pmay_g"}.issubset(scheme_ids)

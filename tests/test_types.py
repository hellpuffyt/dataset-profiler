"""Type inference: per-value classification and the widening lattice."""

from __future__ import annotations

import itertools
from datetime import date, datetime

import pytest

from dataset_profiler import types as t


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", t.EMPTY),
        ("   ", t.EMPTY),
        ("NA", t.EMPTY),
        ("n/a", t.EMPTY),
        ("NULL", t.EMPTY),
        ("NaN", t.EMPTY),
        ("-", t.EMPTY),
        ("?", t.EMPTY),
        ("true", t.BOOLEAN),
        ("FALSE", t.BOOLEAN),
        ("yes", t.BOOLEAN),
        ("N", t.BOOLEAN),
        ("0", t.INTEGER),
        ("42", t.INTEGER),
        ("-7", t.INTEGER),
        ("+7", t.INTEGER),
        ("  19  ", t.INTEGER),
        ("3.5", t.FLOAT),
        ("-0.25", t.FLOAT),
        ("1e5", t.FLOAT),
        ("2024-01-02", t.DATE),
        ("2024/01/02", t.DATE),
        ("2024-01-02T03:04:05", t.DATETIME),
        ("2024-01-02 03:04:05", t.DATETIME),
        ("2024-01-02T03:04:05Z", t.DATETIME),
        ("hello", t.STRING),
        ("inf", t.STRING),
        ("-Infinity", t.STRING),
        ("2024-13-01", t.STRING),
        ("2024-1-2", t.STRING),
    ],
)
def test_classify_recognises_each_kind(value: str, expected: str) -> None:
    assert t.classify(value) == expected


def test_zero_and_one_are_numbers_not_booleans() -> None:
    """A column of 0/1 is nearly always a count or a flag people sum."""
    assert t.classify("0") == t.INTEGER
    assert t.classify("1") == t.INTEGER
    assert t.parse_boolean("0") is None


def test_null_tokens_can_be_replaced() -> None:
    custom = frozenset({"", "missing"})
    assert t.classify("missing", custom) == t.EMPTY
    assert t.classify("NA", custom) == t.STRING


def test_is_null_ignores_case_and_padding() -> None:
    assert t.is_null("  N/A  ")
    assert not t.is_null("na-ish")


def test_null_token_reports_the_stripped_spelling() -> None:
    assert t.null_token("  N/A ") == "N/A"
    assert t.null_token("   ") == ""


@pytest.mark.parametrize("value", ["true", "T", "Yes", " y "])
def test_true_spellings(value: str) -> None:
    assert t.parse_boolean(value) is True


@pytest.mark.parametrize("value", ["false", "F", "No", " n "])
def test_false_spellings(value: str) -> None:
    assert t.parse_boolean(value) is False


def test_parse_boolean_rejects_other_text() -> None:
    assert t.parse_boolean("maybe") is None


def test_to_number_parses_and_rejects() -> None:
    assert t.to_number(" 3.5 ") == 3.5
    assert t.to_number("") is None
    assert t.to_number("abc") is None


def test_to_number_rejects_infinity_and_nan() -> None:
    """They parse in Python but a profile cannot summarise them."""
    assert t.to_number("inf") is None
    assert t.to_number("-inf") is None
    assert t.to_number("nan") is None


def test_parse_date_accepts_both_separators() -> None:
    assert t.parse_date("2024-01-02") == date(2024, 1, 2)
    assert t.parse_date("2024/01/02") == date(2024, 1, 2)


def test_parse_date_rejects_impossible_and_unpadded_dates() -> None:
    assert t.parse_date("2024-02-31") is None
    assert t.parse_date("2024-1-2") is None
    assert t.parse_date("not a date") is None


def test_parse_datetime_handles_zulu() -> None:
    parsed = t.parse_datetime("2024-01-02T03:04:05Z")
    assert parsed is not None
    assert parsed.utcoffset() is not None
    assert parsed.replace(tzinfo=None) == datetime(2024, 1, 2, 3, 4, 5)


def test_parse_datetime_rejects_short_and_bad_text() -> None:
    assert t.parse_datetime("2024-01-02") is None
    assert t.parse_datetime("yesterday at noon") is None


def test_unify_is_idempotent() -> None:
    for kind in t.KINDS:
        assert t.unify(kind, kind) == kind


def test_empty_is_the_identity_of_the_lattice() -> None:
    for kind in t.KINDS:
        assert t.unify(t.EMPTY, kind) == kind
        assert t.unify(kind, t.EMPTY) == kind


def test_string_absorbs_everything() -> None:
    for kind in t.KINDS:
        assert t.unify(t.STRING, kind) == t.STRING


def test_unify_is_commutative() -> None:
    for left, right in itertools.product(t.KINDS, repeat=2):
        assert t.unify(left, right) == t.unify(right, left)


def test_unify_is_associative() -> None:
    """Associativity is what makes the inferred type row-order independent."""
    for a, b, c in itertools.product(t.KINDS, repeat=3):
        assert t.unify(t.unify(a, b), c) == t.unify(a, t.unify(b, c))


def test_integers_widen_to_float() -> None:
    assert t.unify(t.INTEGER, t.FLOAT) == t.FLOAT


def test_dates_widen_to_datetime() -> None:
    assert t.unify(t.DATE, t.DATETIME) == t.DATETIME


def test_unrelated_kinds_fall_back_to_string() -> None:
    assert t.unify(t.BOOLEAN, t.INTEGER) == t.STRING
    assert t.unify(t.DATE, t.FLOAT) == t.STRING


def test_infer_folds_a_histogram_of_kinds() -> None:
    assert t.infer({t.INTEGER: 9, t.FLOAT: 1}) == t.FLOAT
    assert t.infer({t.INTEGER: 9, t.EMPTY: 3}) == t.INTEGER


def test_infer_ignores_kinds_with_a_zero_count() -> None:
    assert t.infer({t.INTEGER: 4, t.STRING: 0}) == t.INTEGER


def test_infer_of_nothing_is_empty() -> None:
    """Never having seen a value is a different fact from having seen text."""
    assert t.infer({}) == t.EMPTY
    assert t.infer({t.EMPTY: 12}) == t.EMPTY


def test_inference_does_not_depend_on_row_order() -> None:
    values = ["1", "2.5", "", "3"]
    forward = t.infer(_histogram(values))
    backward = t.infer(_histogram(list(reversed(values))))
    assert forward == backward == t.FLOAT


def _histogram(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        kind = t.classify(value)
        counts[kind] = counts.get(kind, 0) + 1
    return counts

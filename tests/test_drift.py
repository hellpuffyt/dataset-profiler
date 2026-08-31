"""Comparing a profile against a baseline."""

from __future__ import annotations

import io

import pytest

from dataset_profiler.drift import (
    ADDED,
    DRIFTED,
    MAJOR,
    MODERATE,
    NONE,
    OK,
    REMOVED,
    TYPE_CHANGED,
    UNKNOWN,
    compare,
    severity_of,
)
from dataset_profiler.errors import UsageError
from dataset_profiler.io import read_table
from dataset_profiler.profile import (
    ColumnProfile,
    DatasetProfile,
    NumericSummary,
    ProfileOptions,
    profile_table,
)


def build(
    text: str, source: str = "mem", options: ProfileOptions | None = None
) -> DatasetProfile:
    return profile_table(read_table(io.StringIO(text)), options=options, source=source)


NUMBERS = "v\n" + "".join(f"{value}\n" for value in range(1, 21))
SHIFTED = "v\n" + "".join(f"{value + 100}\n" for value in range(1, 21))
CATEGORIES = "v\n" + "a\n" * 10 + "b\n" * 10
FLIPPED = "v\n" + "a\n" * 2 + "b\n" * 18


@pytest.mark.parametrize(
    ("score", "expected"),
    [(None, UNKNOWN), (0.0, NONE), (0.099, NONE), (0.1, MODERATE), (0.3, MAJOR)],
)
def test_severity_bands(score: float | None, expected: str) -> None:
    assert severity_of(score) == expected


def test_a_dataset_compared_against_itself_has_not_drifted() -> None:
    profile = build(NUMBERS)
    report = compare(profile, profile)
    assert not report.has_drift
    assert report.changes == ()
    assert report.column("v").status == OK
    assert report.column("v").psi == pytest.approx(0.0)


def test_the_report_carries_both_sides_of_the_comparison() -> None:
    report = compare(build(NUMBERS, "old.csv"), build(SHIFTED, "new.csv"))
    assert report.baseline_source == "old.csv"
    assert report.current_source == "new.csv"
    assert report.baseline_rows == 20
    assert report.current_rows == 20


def test_a_numeric_column_moved_off_the_baseline_range_is_major_drift() -> None:
    report = compare(build(NUMBERS), build(SHIFTED))
    column = report.column("v")
    assert column.status == DRIFTED
    assert column.severity == MAJOR
    assert column.psi is not None and column.psi > 1.0


def test_a_small_numeric_shift_is_below_the_default_threshold() -> None:
    nudged = "v\n" + "".join(f"{value + 1}\n" for value in range(1, 21))
    report = compare(build(NUMBERS), build(nudged))
    assert report.column("v").status == OK


def test_the_psi_threshold_can_be_tightened() -> None:
    nudged = "v\n" + "".join(f"{value + 1}\n" for value in range(1, 21))
    report = compare(build(NUMBERS), build(nudged), psi_threshold=0.001)
    assert report.column("v").status == DRIFTED


def test_a_categorical_shift_is_detected() -> None:
    report = compare(build(CATEGORIES), build(FLIPPED))
    column = report.column("v")
    assert column.status == DRIFTED
    assert column.psi is not None and column.psi > 0.2
    assert any("PSI" in note for note in column.notes)


def test_an_unchanged_categorical_column_scores_zero() -> None:
    report = compare(build(CATEGORIES), build(CATEGORIES))
    assert report.column("v").psi == pytest.approx(0.0)


def test_a_column_that_changed_type_is_not_given_a_psi() -> None:
    """A distance between text and numbers would be a fabricated number."""
    report = compare(build("v\n1\n2\n3\n"), build("v\nx\ny\nz\n"))
    column = report.column("v")
    assert column.status == TYPE_CHANGED
    assert column.psi is None
    assert column.severity == UNKNOWN
    assert column.notes == ("type changed: integer -> string",)


def test_a_rise_in_nulls_is_drift_on_its_own() -> None:
    baseline = build("v\na\nb\nc\nd\n")
    current = build("v\na\nb\n\n\n")
    column = compare(baseline, current).column("v")
    assert column.status == DRIFTED
    assert column.null_rate_delta == pytest.approx(0.5)
    assert any("null rate" in note for note in column.notes)


def test_a_null_rate_change_inside_the_threshold_is_ignored() -> None:
    baseline = build("v\n" + "a\n" * 100)
    current = build("v\n" + "a\n" * 98 + "\n\n")
    report = compare(baseline, current, null_threshold=0.05)
    assert report.column("v").status == OK


def test_the_null_threshold_can_be_tightened() -> None:
    baseline = build("v\n" + "a\n" * 100)
    current = build("v\n" + "a\n" * 98 + "\n\n")
    report = compare(baseline, current, null_threshold=0.001)
    assert report.column("v").status == DRIFTED


def test_losing_uniqueness_is_reported() -> None:
    """A key that stopped being a key breaks joins downstream."""
    baseline = build("v\n1\n2\n3\n4\n")
    current = build("v\n1\n1\n2\n2\n")
    column = compare(baseline, current).column("v")
    assert column.status == DRIFTED
    assert "values are no longer unique" in column.notes


def test_losing_uniqueness_is_drift_even_when_nothing_else_moved() -> None:
    """Near-unique text gets no PSI, so this is the only signal there is."""
    baseline = build("v\nid-1\nid-2\nid-3\nid-4\n")
    current = build("v\nid-1\nid-1\nid-2\nid-2\n")
    column = compare(baseline, current).column("v")
    assert column.psi is None
    assert column.status == DRIFTED
    assert "values are no longer unique" in column.notes


def test_a_baseline_with_an_empty_histogram_yields_no_psi() -> None:
    """Hand-edited or truncated baselines must not divide by zero."""
    summary = NumericSummary(
        count=0,
        minimum=0.0,
        maximum=1.0,
        mean=0.0,
        stddev=0.0,
        quantiles={},
        edges=[0.0, 1.0],
        counts=[0],
        sampled=False,
    )
    column = ColumnProfile(
        name="v", count=1, null_count=0, inferred_type="integer", numeric=summary
    )
    profile = DatasetProfile(row_count=1, columns=(column,))
    assert compare(profile, profile).column("v").psi is None


def test_a_removed_column_is_reported() -> None:
    report = compare(build("a,b\n1,2\n"), build("a\n1\n"))
    column = report.column("b")
    assert column.status == REMOVED
    assert column.baseline_type == "integer"
    assert column.current_type is None
    assert column.baseline_distinct == 1
    assert report.has_drift


def test_an_added_column_is_reported() -> None:
    report = compare(build("a\n1\n"), build("a,b\n1,2\n"))
    column = report.column("b")
    assert column.status == ADDED
    assert column.baseline_type is None
    assert column.current_type == "integer"
    assert column.current_distinct == 1


def test_column_order_follows_the_baseline_with_new_columns_appended() -> None:
    report = compare(build("a,b\n1,2\n"), build("b,c,a\n2,3,1\n"))
    assert [column.name for column in report.columns] == ["a", "b", "c"]


def test_a_missing_column_has_no_null_rate_delta() -> None:
    report = compare(build("a,b\n1,2\n"), build("a\n1\n"))
    assert report.column("b").null_rate_delta is None


def test_asking_for_a_column_the_report_does_not_have() -> None:
    report = compare(build("a\n1\n"), build("a\n1\n"))
    with pytest.raises(KeyError):
        report.column("nope")


def test_near_unique_text_columns_are_not_compared_by_frequency() -> None:
    """Counting values that each occur once measures nothing but the ids."""
    baseline = build("v\nid-1\nid-2\nid-3\nid-4\n")
    current = build("v\nid-5\nid-6\nid-7\nid-8\n")
    column = compare(baseline, current).column("v")
    assert column.psi is None
    assert column.status == OK
    assert "near-unique" in column.notes[0]


def test_an_empty_column_cannot_be_compared() -> None:
    report = compare(build("v\n\n\n"), build("v\n\n\n"))
    column = report.column("v")
    assert column.psi is None
    assert "not enough values" in column.notes[0]


def test_a_truncated_category_list_is_flagged_as_approximate() -> None:
    options = ProfileOptions(max_categories=2)
    baseline = build("v\na\nb\nc\na\nb\nc\n", options=options)
    current = build("v\na\nb\nc\na\nb\nc\n", options=options)
    column = compare(baseline, current).column("v")
    assert any("truncated" in note for note in column.notes)


def test_a_sampled_histogram_is_flagged_as_approximate() -> None:
    options = ProfileOptions(sample_size=5)
    baseline = build(NUMBERS, options=options)
    current = build(SHIFTED, options=options)
    column = compare(baseline, current).column("v")
    assert any("sample" in note for note in column.notes)


def test_thresholds_are_validated() -> None:
    profile = build("v\n1\n")
    with pytest.raises(UsageError, match="psi-threshold"):
        compare(profile, profile, psi_threshold=-1)
    with pytest.raises(UsageError, match="null-threshold"):
        compare(profile, profile, null_threshold=1.5)


def test_the_report_serialises_to_json_ready_data() -> None:
    report = compare(build(NUMBERS, "a.csv"), build(SHIFTED, "b.csv"))
    data = report.to_dict()
    assert data["format"] == "dataset-drift/1"
    assert data["has_drift"] is True
    assert data["columns"][0]["name"] == "v"
    assert data["columns"][0]["severity"] == MAJOR


def test_a_column_entry_serialises_its_notes_as_a_list() -> None:
    report = compare(build("v\n1\n2\n3\n"), build("v\nx\ny\nz\n"))
    entry = report.column("v").to_dict()
    assert entry["notes"] == ["type changed: integer -> string"]
    assert entry["psi"] is None


def test_comparing_two_empty_datasets_is_quiet() -> None:
    report = compare(build("v\n"), build("v\n"))
    assert not report.has_drift
    assert report.column("v").psi is None


def test_type_change_takes_priority_over_a_null_rate_note() -> None:
    baseline = build("v\n1\n2\n3\n4\n")
    current = build("v\nx\n\n\n\n")
    column = compare(baseline, current).column("v")
    assert column.status == TYPE_CHANGED
    assert column.notes[0].startswith("type changed")
    assert any("null rate" in note for note in column.notes)
    assert column.psi is None


def test_changes_lists_only_what_moved() -> None:
    baseline = build("a,b\n1,x\n2,x\n3,x\n4,x\n")
    current = build("a,b\n101,x\n102,x\n103,x\n104,x\n")
    report = compare(baseline, current)
    assert [column.name for column in report.changes] == ["a"]
    assert len(report.columns) == 2


def test_is_change_matches_the_status() -> None:
    report = compare(build("a,b\n1,x\n2,x\n"), build("a,b\n1,x\n2,x\n"))
    assert all(not column.is_change for column in report.columns)

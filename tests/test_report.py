"""Rendering: the aligned table, the machine format, and the JSON."""

from __future__ import annotations

import io
import json

import pytest

from dataset_profiler.drift import compare
from dataset_profiler.io import read_table
from dataset_profiler.profile import DatasetProfile, ProfileOptions, profile_table
from dataset_profiler.report import (
    JSON,
    TEXT,
    TSV,
    column_detail,
    format_number,
    format_rate,
    render_drift,
    render_drift_text,
    render_profile,
    render_profile_text,
    render_profile_tsv,
    render_table,
)

SALES = "id,region,amount\n1,north,10\n2,north,20\n3,south,\n"


def build(text: str, source: str = "sales.csv") -> DatasetProfile:
    return profile_table(read_table(io.StringIO(text)), source=source)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3.0, "3"), (-2.0, "-2"), (2.5, "2.5"), (1 / 3, "0.333333")],
)
def test_numbers_are_rendered_compactly(value: float, expected: str) -> None:
    assert format_number(value) == expected


def test_very_large_numbers_keep_their_exponent() -> None:
    assert "e" in format_number(1.2e20)


def test_rates_are_percentages_with_one_decimal() -> None:
    assert format_rate(0.1667) == "16.7%"
    assert format_rate(0.0) == "0.0%"


def test_render_table_aligns_columns_and_rules_the_header() -> None:
    text = render_table([["a", "bbb"], ["cccc", "d"]])
    lines = text.splitlines()
    assert lines[0] == "a     bbb"
    assert lines[1] == "----  ---"
    assert lines[2] == "cccc  d"


def test_render_table_can_omit_the_rule() -> None:
    text = render_table([["a", "b"], ["c", "d"]], rule_after_header=False)
    assert text.splitlines() == ["a  b", "c  d"]


def test_render_table_pads_short_rows() -> None:
    text = render_table([["a", "b"], ["c"]])
    assert text.splitlines()[2] == "c"


def test_render_table_of_nothing_is_empty() -> None:
    assert render_table([]) == ""


def test_the_profile_report_names_the_source_and_the_shape() -> None:
    text = render_profile_text(build(SALES))
    assert "dataset: sales.csv" in text
    assert "rows: 3, columns: 3" in text


def test_a_profile_of_stdin_says_so() -> None:
    text = render_profile_text(build(SALES, source=""))
    assert "dataset: (stdin)" in text


def test_the_profile_report_has_a_line_per_column() -> None:
    text = render_profile_text(build(SALES))
    for name in ("id", "region", "amount"):
        assert any(line.startswith(name) for line in text.splitlines())


def test_ragged_rows_are_warned_about() -> None:
    text = render_profile_text(build("a,b\n1\n"))
    assert "warning: 1 row(s) did not match the header width" in text


def test_a_dataset_with_no_columns_says_so() -> None:
    text = render_profile_text(DatasetProfile(row_count=0, columns=()))
    assert text.endswith("no columns")


def test_a_truncated_cardinality_is_shown_as_a_lower_bound() -> None:
    profile = profile_table(
        read_table(io.StringIO("v\na\nb\nc\n")),
        options=ProfileOptions(max_distinct=1),
    )
    assert ">=1" in render_profile_text(profile)


def test_detail_calls_out_a_key_column() -> None:
    assert column_detail(build(SALES).column("id")).startswith("unique")


def test_detail_calls_out_a_constant_column() -> None:
    detail = column_detail(build("v\nx\nx\n").column("v"))
    assert detail.startswith("constant 'x'")


def test_detail_calls_out_a_column_that_was_never_populated() -> None:
    assert column_detail(build("v\n\n\n").column("v")) == "never populated"


def test_detail_gives_range_and_mean_for_numbers() -> None:
    detail = column_detail(build(SALES).column("amount"))
    assert "range 10..20" in detail
    assert "mean 15" in detail


def test_detail_gives_the_top_value_and_lengths_for_text() -> None:
    detail = column_detail(build(SALES).column("region"))
    assert "top 'north' x2" in detail
    assert "len 5..5" in detail


def test_detail_reports_how_nulls_were_spelt() -> None:
    detail = column_detail(build("v\n1\nN/A\nNULL\nN/A\n").column("v"))
    assert "null spelt 'N/A' x2, 'NULL' x1" in detail


def test_detail_ignores_the_plain_empty_spelling() -> None:
    """Every file has blanks; only unusual spellings are worth the space."""
    detail = column_detail(build("v\n1\n\n").column("v"))
    assert "null spelt" not in detail


def test_a_constant_column_of_numbers_gets_both_facts() -> None:
    detail = column_detail(build("v\n5\n5\n").column("v"))
    assert detail.startswith("constant '5'")
    assert "range 5..5" in detail


def test_the_tsv_profile_is_one_line_per_column_plus_a_header() -> None:
    lines = render_profile_tsv(build(SALES)).splitlines()
    assert lines[0] == "column\ttype\tnulls\tdistinct\tunique_ratio"
    assert len(lines) == 4
    assert lines[1] == "id\tinteger\t0\t3\t1.0000"


def test_the_json_profile_is_the_baseline_format() -> None:
    profile = build(SALES)
    text = render_profile(profile, JSON)
    assert DatasetProfile.from_json(text) == profile


def test_the_drift_report_states_its_thresholds() -> None:
    report = compare(build(SALES, "a.csv"), build(SALES, "b.csv"))
    text = render_drift_text(report)
    assert "thresholds: psi >= 0.2, null rate delta > 5.0%" in text
    assert "baseline: a.csv (3 rows)" in text
    assert "current:  b.csv (3 rows)" in text


def test_an_unchanged_comparison_says_no_drift() -> None:
    report = compare(build(SALES), build(SALES))
    assert render_drift_text(report).endswith("no drift detected")


def test_a_changed_comparison_counts_the_columns() -> None:
    report = compare(build(SALES), build("id,region,amount\n1,west,900\n"))
    assert render_drift_text(report).endswith("column(s) changed")


def test_a_drift_report_with_no_columns() -> None:
    empty = DatasetProfile(row_count=0, columns=())
    text = render_drift_text(compare(empty, empty))
    assert "no columns" in text


def test_the_tsv_drift_report_shows_a_dash_for_an_absent_psi() -> None:
    report = compare(build("v\n1\n2\n3\n"), build("v\nx\ny\nz\n"))
    lines = render_drift(report, TSV).splitlines()
    assert lines[0] == "column\tstatus\tseverity\tpsi"
    assert lines[1] == "v\ttype_changed\tunknown\t-"


def test_the_json_drift_report_parses() -> None:
    report = compare(build(SALES), build(SALES))
    data = json.loads(render_drift(report, JSON))
    assert data["has_drift"] is False


def test_dispatch_rejects_an_unknown_format() -> None:
    profile = build(SALES)
    with pytest.raises(ValueError, match="unknown format"):
        render_profile(profile, "xml")
    with pytest.raises(ValueError, match="unknown format"):
        render_drift(compare(profile, profile), "xml")


def test_dispatch_reaches_every_named_format() -> None:
    profile = build(SALES)
    report = compare(profile, profile)
    for fmt in (TEXT, TSV, JSON):
        assert render_profile(profile, fmt)
        assert render_drift(report, fmt)

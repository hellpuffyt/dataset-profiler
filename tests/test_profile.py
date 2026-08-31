"""Building profiles, and getting them back out of JSON unchanged."""

from __future__ import annotations

import io
import json
import tracemalloc

import pytest

from dataset_profiler.errors import BaselineError, ColumnError, UsageError
from dataset_profiler.io import read_table
from dataset_profiler.profile import (
    FORMAT,
    ColumnProfile,
    DatasetProfile,
    NumericSummary,
    ProfileOptions,
    profile_rows,
    profile_table,
)

SALES = (
    "order_id,region,amount,active\n"
    "1,north,10,true\n"
    "2,north,20,true\n"
    "3,south,30,false\n"
    "4,south,,N/A\n"
)


def build(
    text: str,
    options: ProfileOptions | None = None,
    select: list[str] | None = None,
) -> DatasetProfile:
    table = read_table(io.StringIO(text))
    return profile_table(table, options=options, source="mem", select=select)


def test_row_count_and_column_names() -> None:
    profile = build(SALES)
    assert profile.row_count == 4
    assert profile.names == ["order_id", "region", "amount", "active"]


def test_types_are_inferred_per_column() -> None:
    profile = build(SALES)
    assert profile.column("order_id").inferred_type == "integer"
    assert profile.column("region").inferred_type == "string"
    assert profile.column("amount").inferred_type == "integer"
    assert profile.column("active").inferred_type == "boolean"


def test_a_single_decimal_widens_the_whole_column() -> None:
    profile = build("v\n1\n2\n3.5\n")
    assert profile.column("v").inferred_type == "float"


def test_nulls_are_counted_and_their_spellings_recorded() -> None:
    profile = build("v\n1\n\nN/A\nna\n")
    column = profile.column("v")
    assert column.null_count == 3
    assert column.null_tokens == {"": 1, "N/A": 1, "na": 1}


def test_null_rate_and_filled_count() -> None:
    column = build(SALES).column("amount")
    assert column.null_count == 1
    assert column.filled_count == 3
    assert column.null_rate == pytest.approx(0.25)


def test_null_rate_of_an_empty_dataset_is_zero_not_a_crash() -> None:
    column = build("v\n").column("v")
    assert column.count == 0
    assert column.null_rate == 0.0
    assert column.unique_ratio == 0.0


def test_cardinality_counts_distinct_non_null_values() -> None:
    profile = build(SALES)
    assert profile.column("region").distinct == 2
    assert profile.column("amount").distinct == 3


def test_a_key_column_is_recognised_as_unique() -> None:
    column = build(SALES).column("order_id")
    assert column.is_unique
    assert column.unique_ratio == pytest.approx(1.0)


def test_a_column_with_a_null_is_not_unique() -> None:
    """A key with a hole in it is not a key."""
    assert not build(SALES).column("amount").is_unique


def test_a_constant_column_is_flagged() -> None:
    column = build("v\nx\nx\nx\n").column("v")
    assert column.is_constant
    assert not column.is_unique


def test_an_all_null_column_is_empty_typed_and_flagged() -> None:
    column = build("v\n\n\n\n").column("v")
    assert column.is_empty
    assert column.inferred_type == "empty"
    assert column.distinct == 0
    assert column.numeric is None


def test_values_are_stripped_before_counting() -> None:
    column = build("v\n a \na\n").column("v")
    assert column.distinct == 1


def test_lengths_are_recorded_for_non_null_values() -> None:
    column = build("v\nab\nabcd\n\n").column("v")
    assert column.min_length == 2
    assert column.max_length == 4


def test_lengths_are_absent_when_nothing_was_ever_populated() -> None:
    column = build("v\n\n").column("v")
    assert column.min_length is None
    assert column.max_length is None


def test_top_values_are_ordered_by_frequency_then_alphabetically() -> None:
    column = build("v\nb\na\nb\nc\na\nb\n").column("v")
    assert column.top_values() == [("b", 3), ("a", 2), ("c", 1)]
    assert column.top_values(1) == [("b", 3)]


def test_the_category_table_can_be_capped() -> None:
    options = ProfileOptions(max_categories=2)
    column = build("v\na\nb\nc\nd\n", options=options).column("v")
    assert column.categories_truncated
    assert len(column.categories) == 2
    assert column.distinct == 4, "cardinality is still counted exactly"


def test_repeat_values_still_count_after_the_category_cap() -> None:
    options = ProfileOptions(max_categories=1)
    column = build("v\na\nb\na\n", options=options).column("v")
    assert column.categories == {"a": 2}
    assert column.categories_truncated


def test_cardinality_stops_counting_at_max_distinct() -> None:
    options = ProfileOptions(max_distinct=1)
    column = build("v\na\nb\nc\n", options=options).column("v")
    assert column.distinct == 1
    assert column.distinct_truncated
    assert not column.is_constant, "a truncated count of 1 proves nothing"


def test_numeric_summary_matches_hand_arithmetic() -> None:
    summary = build("v\n1\n2\n3\n4\n").column("v").numeric
    assert summary is not None
    assert summary.count == 4
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0
    assert summary.mean == pytest.approx(2.5)
    assert summary.span == pytest.approx(3.0)
    assert summary.quantiles["p50"] == pytest.approx(2.5)
    assert sum(summary.counts) == 4


def test_the_histogram_spans_the_observed_range() -> None:
    summary = build("v\n0\n10\n").column("v").numeric
    assert summary is not None
    assert summary.edges[0] == 0.0
    assert summary.edges[-1] == 10.0
    assert len(summary.edges) == 11


def test_the_bin_count_is_configurable() -> None:
    summary = build("v\n1\n2\n3\n", options=ProfileOptions(bins=4)).column("v").numeric
    assert summary is not None
    assert len(summary.counts) == 4


def test_a_full_column_is_not_marked_as_sampled() -> None:
    summary = build("v\n1\n2\n3\n").column("v").numeric
    assert summary is not None
    assert not summary.sampled


def test_a_column_beyond_the_sample_size_is_marked_as_sampled() -> None:
    text = "v\n" + "".join(f"{value}\n" for value in range(50))
    options = ProfileOptions(sample_size=10)
    summary = build(text, options=options).column("v").numeric
    assert summary is not None
    assert summary.sampled
    assert summary.count == 50, "moments still see every value"
    assert summary.minimum == 0.0
    assert summary.maximum == 49.0


def test_profiling_is_reproducible_for_a_given_seed() -> None:
    text = "v\n" + "".join(f"{value}\n" for value in range(200))
    options = ProfileOptions(sample_size=20, seed=3)
    assert build(text, options=options) == build(text, options=options)


def test_non_numeric_columns_have_no_numeric_summary() -> None:
    assert build(SALES).column("region").numeric is None


def test_short_rows_are_profiled_as_missing_values() -> None:
    profile = build("a,b\n1\n")
    assert profile.column("b").null_count == 1
    assert profile.ragged_rows == 1


def test_ragged_rows_are_carried_into_the_profile() -> None:
    assert build("a,b\n1,2,3\n").ragged_rows == 1


def test_selecting_a_subset_of_columns() -> None:
    profile = build(SALES, select=["region", "order_id"])
    assert profile.names == ["region", "order_id"]


def test_selecting_a_column_that_does_not_exist() -> None:
    with pytest.raises(ColumnError, match="no such column 'nope'"):
        build(SALES, select=["nope"])


def test_asking_for_a_column_that_does_not_exist() -> None:
    with pytest.raises(ColumnError, match="dataset has: order_id"):
        build(SALES).column("nope")


def test_column_error_of_a_dataset_with_no_columns() -> None:
    profile = DatasetProfile(row_count=0, columns=())
    with pytest.raises(ColumnError, match=r"\(none\)"):
        profile.column("x")


def test_profile_rows_accepts_bare_sequences() -> None:
    profile = profile_rows(["a"], [["1"], ["2"]])
    assert profile.row_count == 2
    assert profile.column("a").inferred_type == "integer"


def test_rows_shorter_than_the_header_do_not_raise() -> None:
    profile = profile_rows(["a", "b"], [["1"]])
    assert profile.column("b").null_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 0},
        {"max_distinct": -1},
        {"sample_size": 0},
        {"bins": 0},
    ],
)
def test_bounded_resource_options_must_be_positive(kwargs: dict[str, int]) -> None:
    with pytest.raises(UsageError, match="must be positive"):
        ProfileOptions(**kwargs)


def test_custom_null_tokens_change_what_counts_as_missing() -> None:
    options = ProfileOptions(null_tokens=frozenset({"", "missing"}))
    column = build("v\nmissing\nNA\n", options=options).column("v")
    assert column.null_count == 1
    assert column.distinct == 1


def test_a_profile_survives_a_json_round_trip() -> None:
    """A baseline that does not round-trip is a printout, not a baseline."""
    original = build(SALES)
    assert DatasetProfile.from_json(original.to_json()) == original


def test_the_serialised_form_is_tagged_with_the_format() -> None:
    assert json.loads(build(SALES).to_json())["format"] == FORMAT


def test_round_tripping_preserves_the_numeric_summary() -> None:
    original = build("v\n1\n2\n3.5\n")
    restored = DatasetProfile.from_json(original.to_json())
    assert restored.column("v").numeric == original.column("v").numeric


def test_round_tripping_preserves_ragged_and_source() -> None:
    original = build("a,b\n1\n")
    restored = DatasetProfile.from_json(original.to_json())
    assert restored.ragged_rows == 1
    assert restored.source == "mem"


def test_a_profile_with_no_columns_round_trips() -> None:
    original = DatasetProfile(row_count=0, columns=())
    assert DatasetProfile.from_json(original.to_json()) == original


def test_some_other_json_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(BaselineError, match="not a dataset profile"):
        DatasetProfile.from_json('{"hello": "world"}')


def test_invalid_json_names_the_file() -> None:
    with pytest.raises(BaselineError, match=r"base\.json: invalid JSON"):
        DatasetProfile.from_json("{not json", "base.json")


def test_a_non_object_profile_is_rejected() -> None:
    with pytest.raises(BaselineError, match="expected an object, found list"):
        DatasetProfile.from_dict([])


def test_the_file_name_is_added_to_nested_errors() -> None:
    raw = json.loads(build(SALES).to_json())
    raw["row_count"] = "four"
    with pytest.raises(BaselineError, match=r"base\.json: profile\.row_count"):
        DatasetProfile.from_dict(raw, "base.json")


def test_a_bad_column_list_is_rejected() -> None:
    with pytest.raises(BaselineError, match="expected a list"):
        DatasetProfile.from_dict({"format": FORMAT, "row_count": 0, "columns": {}})


def test_a_boolean_is_not_accepted_where_an_integer_is_required() -> None:
    """JSON booleans are Python integers; a profile must not accept them."""
    raw = json.loads(build(SALES).to_json())
    raw["row_count"] = True
    with pytest.raises(BaselineError, match="expected an integer, found bool"):
        DatasetProfile.from_dict(raw)


def test_a_missing_column_name_is_rejected() -> None:
    with pytest.raises(BaselineError, match="expected a string, found null"):
        ColumnProfile.from_dict({"count": 1})


def test_a_bad_type_count_map_is_rejected() -> None:
    raw = json.loads(build(SALES).to_json())
    raw["columns"][0]["type_counts"] = {"integer": "lots"}
    with pytest.raises(BaselineError, match="type_counts"):
        DatasetProfile.from_dict(raw)


def test_a_histogram_with_the_wrong_number_of_edges_is_rejected() -> None:
    raw = json.loads(build(SALES).to_json())
    raw["columns"][0]["numeric"]["histogram"]["edges"] = [0.0, 1.0]
    with pytest.raises(BaselineError, match="edges and counts disagree"):
        DatasetProfile.from_dict(raw)


def test_a_non_numeric_mean_is_rejected() -> None:
    raw = json.loads(build(SALES).to_json())
    raw["columns"][0]["numeric"]["mean"] = "middling"
    with pytest.raises(BaselineError, match="expected a number, found str"):
        DatasetProfile.from_dict(raw)


def test_optional_lengths_may_be_absent_or_present() -> None:
    raw = json.loads(build(SALES).to_json())
    raw["columns"][0]["min_length"] = None
    restored = DatasetProfile.from_dict(raw)
    assert restored.columns[0].min_length is None
    assert restored.columns[0].max_length == 1


def test_a_non_integer_length_is_rejected() -> None:
    raw = json.loads(build(SALES).to_json())
    raw["columns"][0]["max_length"] = 1.5
    with pytest.raises(BaselineError, match="max_length"):
        DatasetProfile.from_dict(raw)


def test_numeric_summary_round_trips_on_its_own() -> None:
    summary = NumericSummary(
        count=2,
        minimum=0.0,
        maximum=1.0,
        mean=0.5,
        stddev=0.5,
        quantiles={"p50": 0.5},
        edges=[0.0, 1.0],
        counts=[2],
        sampled=True,
    )
    assert NumericSummary.from_dict(summary.to_dict(), "n") == summary


def test_a_numeric_summary_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(BaselineError, match="n: expected an object"):
        NumericSummary.from_dict("nope", "n")


def test_memory_does_not_grow_with_the_number_of_rows() -> None:
    """The bounded-resource options are the whole memory story.

    Measured on the machine that wrote this: 1,000,000 rows of five
    columns peak at 20 MB of Python heap with the defaults and 1.8 MB
    with --max-distinct 1000. This asserts the shape of that claim on a
    size small enough to run in CI.
    """
    text = "v\n" + "".join(f"value-{n}\n" for n in range(100_000))
    options = ProfileOptions(max_distinct=100, max_categories=10, sample_size=100)
    tracemalloc.start()
    try:
        profile = build(text, options=options)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert profile.row_count == 100_000
    assert profile.column("v").distinct_truncated
    assert peak < 8_000_000, f"peak heap was {peak} bytes"

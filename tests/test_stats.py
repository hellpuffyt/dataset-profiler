"""The arithmetic: moments, reservoir sampling, quantiles, binning, PSI."""

from __future__ import annotations

import math

import pytest

from dataset_profiler.stats import (
    Reservoir,
    RunningMoments,
    bin_edges,
    bucket,
    histogram,
    proportions,
    psi,
    quantile,
    rebin,
)


def moments_of(values: list[float]) -> RunningMoments:
    running = RunningMoments()
    for value in values:
        running.add(value)
    return running


def test_moments_of_a_textbook_sample() -> None:
    running = moments_of([2, 4, 4, 4, 5, 5, 7, 9])
    assert running.count == 8
    assert running.mean == pytest.approx(5.0)
    assert running.stddev == pytest.approx(2.0)
    assert running.minimum == 2
    assert running.maximum == 9


def test_moments_start_empty() -> None:
    running = RunningMoments()
    assert running.count == 0
    assert running.variance == 0.0
    assert running.stddev == 0.0
    assert running.minimum == math.inf
    assert running.maximum == -math.inf


def test_variance_of_one_value_is_zero_not_undefined() -> None:
    assert moments_of([7.0]).variance == 0.0


def test_variance_stays_positive_on_timestamp_scale_values() -> None:
    """The naive sum-of-squares formula returns a negative variance here."""
    running = moments_of([1_700_000_000.0, 1_700_000_001.0, 1_700_000_002.0])
    assert running.variance == pytest.approx(2 / 3)
    assert running.stddev > 0


def test_constant_input_has_zero_spread() -> None:
    running = moments_of([3.0] * 50)
    assert running.stddev == 0.0
    assert running.minimum == running.maximum == 3.0


def test_reservoir_keeps_everything_below_capacity() -> None:
    reservoir = Reservoir(10)
    for value in range(4):
        reservoir.add(float(value))
    assert reservoir.sorted_values() == [0.0, 1.0, 2.0, 3.0]
    assert reservoir.complete


def test_reservoir_stops_growing_at_capacity() -> None:
    reservoir = Reservoir(5)
    for value in range(1000):
        reservoir.add(float(value))
    assert len(reservoir.sorted_values()) == 5
    assert reservoir.seen == 1000
    assert not reservoir.complete


def test_reservoir_is_deterministic_for_a_given_seed() -> None:
    """Two runs over the same file must produce the same profile."""
    first = Reservoir(20, seed=7)
    second = Reservoir(20, seed=7)
    for value in range(500):
        first.add(float(value))
        second.add(float(value))
    assert first.sorted_values() == second.sorted_values()


def test_reservoir_seeds_select_different_samples() -> None:
    first = Reservoir(20, seed=1)
    second = Reservoir(20, seed=2)
    for value in range(500):
        first.add(float(value))
        second.add(float(value))
    assert first.sorted_values() != second.sorted_values()


def test_reservoir_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        Reservoir(0)


def test_quantiles_interpolate_linearly() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert quantile(values, 0.0) == 1.0
    assert quantile(values, 0.25) == pytest.approx(1.75)
    assert quantile(values, 0.5) == pytest.approx(2.5)
    assert quantile(values, 1.0) == 4.0


def test_quantile_hits_an_exact_index_without_interpolating() -> None:
    assert quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_quantile_of_a_single_value() -> None:
    assert quantile([9.0], 0.25) == 9.0


def test_quantile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no values"):
        quantile([], 0.5)


@pytest.mark.parametrize("q", [-0.01, 1.01])
def test_quantile_range_is_checked(q: float) -> None:
    with pytest.raises(ValueError, match="out of range"):
        quantile([1.0, 2.0], q)


def test_bin_edges_are_evenly_spaced() -> None:
    assert bin_edges(0.0, 10.0, 5) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_bin_edges_end_exactly_on_the_maximum() -> None:
    edges = bin_edges(0.0, 1.0, 3)
    assert edges[-1] == 1.0
    assert len(edges) == 4


def test_a_degenerate_range_collapses_to_one_bin() -> None:
    """Otherwise every bin has zero width and the next step divides by it."""
    assert bin_edges(5.0, 5.0, 10) == [5.0, 5.0]


def test_bin_edges_reject_nonsense() -> None:
    with pytest.raises(ValueError, match="bins must be positive"):
        bin_edges(0.0, 1.0, 0)
    with pytest.raises(ValueError, match="finite"):
        bin_edges(0.0, math.inf, 4)


def test_bucket_clamps_outside_values_into_the_outer_bins() -> None:
    edges = [0.0, 1.0, 2.0]
    assert bucket(-99.0, edges) == 0
    assert bucket(0.5, edges) == 0
    assert bucket(1.5, edges) == 1
    assert bucket(99.0, edges) == 1


def test_bucket_puts_the_top_edge_in_the_last_bin() -> None:
    assert bucket(2.0, [0.0, 1.0, 2.0]) == 1


def test_bucket_needs_two_edges() -> None:
    with pytest.raises(ValueError, match="two edges"):
        bucket(1.0, [0.0])


def test_histogram_counts_every_value_once() -> None:
    counts = histogram([1.0, 2.0, 3.0, 9.0], [0.0, 2.0, 4.0])
    assert counts == [1, 3]
    assert sum(counts) == 4


def test_rebinning_onto_the_same_edges_changes_nothing() -> None:
    counts = [3, 0, 5]
    edges = [0.0, 1.0, 2.0, 3.0]
    assert rebin(counts, edges, edges) == pytest.approx([3.0, 0.0, 5.0])


def test_rebinning_preserves_total_mass() -> None:
    counts = [4, 6, 2]
    edges = [0.0, 1.0, 2.0, 3.0]
    moved = rebin(counts, edges, [0.0, 1.5, 3.0])
    assert sum(moved) == pytest.approx(12.0)


def test_rebinning_splits_a_source_bin_uniformly() -> None:
    moved = rebin([10], [0.0, 1.0], [0.0, 0.5, 1.0])
    assert moved == pytest.approx([5.0, 5.0])


def test_mass_outside_the_target_range_is_clamped_not_dropped() -> None:
    """A shift off the end of the baseline range must still be visible."""
    moved = rebin([10], [0.0, 1.0], [5.0, 6.0, 7.0])
    assert moved == pytest.approx([10.0, 0.0])
    above = rebin([10], [8.0, 9.0], [5.0, 6.0, 7.0])
    assert above == pytest.approx([0.0, 10.0])


def test_rebinning_a_degenerate_source_bin() -> None:
    moved = rebin([4], [3.0, 3.0], [0.0, 5.0, 10.0])
    assert moved == pytest.approx([4.0, 0.0])


def test_rebinning_skips_empty_source_bins() -> None:
    moved = rebin([0, 7], [0.0, 1.0, 2.0], [0.0, 2.0])
    assert moved == pytest.approx([7.0])


def test_rebinning_checks_its_arguments() -> None:
    with pytest.raises(ValueError, match="disagree"):
        rebin([1, 2], [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="target edges"):
        rebin([1], [0.0, 1.0], [0.0])


def test_proportions_sum_to_one() -> None:
    assert sum(proportions([1, 1, 2])) == pytest.approx(1.0)


def test_proportions_floor_empty_bins() -> None:
    assert proportions([0, 4])[0] == pytest.approx(1e-6)


def test_proportions_of_an_empty_histogram_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty histogram"):
        proportions([0, 0])


def test_psi_of_identical_histograms_is_zero() -> None:
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)


def test_psi_ignores_scale() -> None:
    """PSI compares shapes, so doubling every count changes nothing."""
    assert psi([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0)


def test_psi_is_symmetric() -> None:
    forward = psi([10, 20, 30], [30, 20, 10])
    backward = psi([30, 20, 10], [10, 20, 30])
    assert forward == pytest.approx(backward)


def test_psi_grows_with_the_size_of_the_shift() -> None:
    small = psi([50, 50], [45, 55])
    large = psi([50, 50], [10, 90])
    assert 0 < small < large


def test_psi_of_a_vanished_category_is_large_but_finite() -> None:
    score = psi([30, 70], [0, 100])
    assert math.isfinite(score)
    assert score > 1.0


def test_psi_checks_shape() -> None:
    with pytest.raises(ValueError, match="same number of bins"):
        psi([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="empty histograms"):
        psi([], [])

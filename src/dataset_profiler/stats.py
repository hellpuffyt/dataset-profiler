"""Numeric machinery: running moments, quantiles, histograms, PSI.

Nothing here knows what a dataset is. It is the arithmetic that the
profiler and the drift comparison both need, kept separate so it can be
tested against values worked out by hand.

Two choices are worth explaining.

**Welford, not sum-of-squares.** The mean and standard deviation are
accumulated with the Welford online algorithm. The textbook
``sum(x*x) - n*mean**2`` loses catastrophically on values that are large
relative to their spread -- a column of Unix timestamps is enough to make
it return a negative variance -- and a profiler that reports a negative
variance is worse than one that reports nothing.

**A seeded reservoir, not the whole column.** Quantiles need the values,
which is the one thing a streaming profiler cannot keep. So values go
into a bounded reservoir with a caller-supplied seed: the sample is
random with respect to the data but *identical* for identical input, so
two runs of the profiler over the same file always agree.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


class RunningMoments:
    """Exact count, min, max, mean and standard deviation in one pass."""

    __slots__ = ("_m2", "count", "maximum", "mean", "minimum")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self._m2 = 0.0

    def add(self, value: float) -> None:
        """Fold one observation in."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    @property
    def variance(self) -> float:
        """Population variance; ``0.0`` for fewer than two observations."""
        if self.count < 2:
            return 0.0
        return max(self._m2 / self.count, 0.0)

    @property
    def stddev(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)


class Reservoir:
    """A bounded, deterministic uniform sample of a stream (algorithm R)."""

    __slots__ = ("_rng", "_values", "capacity", "seen")

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = capacity
        self.seen = 0
        self._rng = random.Random(seed)
        self._values: list[float] = []

    def add(self, value: float) -> None:
        """Offer one value to the sample."""
        self.seen += 1
        if len(self._values) < self.capacity:
            self._values.append(value)
            return
        index = self._rng.randrange(self.seen)
        if index < self.capacity:
            self._values[index] = value

    @property
    def complete(self) -> bool:
        """Whether the sample is every value seen, not a subset of them."""
        return self.seen <= self.capacity

    def sorted_values(self) -> list[float]:
        """The sample, ascending."""
        return sorted(self._values)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-quantile of already-sorted values, linearly interpolated.

    This is the same definition as the default of ``numpy.quantile``, so
    results are comparable with the tools people will check them against.
    """
    if not sorted_values:
        raise ValueError("quantile of no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile out of range: {q}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    below = float(sorted_values[lower])
    above = float(sorted_values[upper])
    return below * (1.0 - weight) + above * weight


def bin_edges(low: float, high: float, bins: int) -> list[float]:
    """``bins + 1`` equally spaced edges from ``low`` to ``high``.

    A degenerate range -- every value identical -- collapses to a single
    bin rather than producing zero-width bins that later divide by zero.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("bin edges must be finite")
    if high <= low:
        return [low, high]
    width = (high - low) / bins
    edges = [low + width * index for index in range(bins)]
    edges.append(high)
    return edges


def bucket(value: float, edges: Sequence[float]) -> int:
    """Index of the bin ``value`` falls in, clamped to the outer bins."""
    count = len(edges) - 1
    if count < 1:
        raise ValueError("need at least two edges")
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return count - 1
    low, high = edges[0], edges[-1]
    index = int((value - low) / (high - low) * count)
    return min(max(index, 0), count - 1)


def histogram(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    """Count how many values land in each bin defined by ``edges``."""
    counts = [0] * (len(edges) - 1)
    for value in values:
        counts[bucket(value, edges)] += 1
    return counts


def rebin(
    counts: Sequence[float],
    edges: Sequence[float],
    target_edges: Sequence[float],
) -> list[float]:
    """Re-express a histogram on a different set of bin edges.

    Mass inside a source bin is assumed uniformly distributed across it,
    which is the standard approximation and the only one available once
    the values themselves are gone. Mass falling outside the target range
    is clamped into the nearest outer target bin rather than discarded,
    so the total is preserved exactly.
    """
    if len(counts) != len(edges) - 1:
        raise ValueError("counts and edges disagree")
    out = [0.0] * (len(target_edges) - 1)
    if not out:
        raise ValueError("need at least two target edges")
    first, last = target_edges[0], target_edges[-1]
    for index, count in enumerate(counts):
        if not count:
            continue
        low, high = edges[index], edges[index + 1]
        if high <= low:
            out[bucket(low, target_edges)] += count
            continue
        width = high - low
        out[0] += count * max(0.0, min(high, first) - low) / width
        out[-1] += count * max(0.0, high - max(low, last)) / width
        for target in range(len(out)):
            overlap = min(high, target_edges[target + 1]) - max(
                low, target_edges[target]
            )
            if overlap > 0:
                out[target] += count * overlap / width
    return out


def proportions(counts: Sequence[float], floor: float = 1e-6) -> list[float]:
    """Normalise counts to proportions, with a floor so PSI stays finite."""
    total = math.fsum(counts)
    if total <= 0:
        raise ValueError("cannot take proportions of an empty histogram")
    return [max(count / total, floor) for count in counts]


def psi(
    expected: Sequence[float], actual: Sequence[float], floor: float = 1e-6
) -> float:
    """Population Stability Index between two histograms of equal length.

    ``sum((a - e) * ln(a / e))``. It is symmetric, zero for identical
    distributions, and the conventional reading is: under 0.1 stable,
    0.1 to 0.25 moderate movement, above 0.25 a material shift.

    An empty bin would send the logarithm to infinity, so proportions are
    floored. That is a real trade-off, not a rounding detail: a category
    that goes from 30% to absent scores high but finite.
    """
    if len(expected) != len(actual):
        raise ValueError("histograms must have the same number of bins")
    if not expected:
        raise ValueError("cannot compare empty histograms")
    expected_p = proportions(expected, floor)
    actual_p = proportions(actual, floor)
    return math.fsum(
        (a - e) * math.log(a / e)
        for e, a in zip(expected_p, actual_p, strict=True)
    )

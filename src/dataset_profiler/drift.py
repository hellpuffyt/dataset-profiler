"""Comparing a profile against a stored baseline.

Drift detection is mostly a question of what you are allowed to compare.
Two profiles of the same dataset a month apart do not share bin edges,
do not share a category list, and may not even share columns, so a naive
"subtract the histograms" is wrong more often than it is right.

What happens here instead:

- **Numeric columns** are compared by re-expressing the current
  histogram on the *baseline* bin edges (:func:`~.stats.rebin`) and then
  taking the PSI. The baseline defines the grid, which is the whole idea
  of a baseline; values that have moved outside its range are clamped
  into the outer bins, so a shift off the end of the old range shows up
  as a large PSI rather than as missing mass.
- **Everything else** is compared as a categorical frequency table over
  the union of both value sets.
- **Columns that changed type** are reported as changed and *not* given
  a PSI, because a number compared against text has no meaningful
  distance and inventing one would be the worst possible outcome here.

Nulls are tracked separately from the value distribution throughout. A
column that goes from 0% to 40% null has a serious problem that a PSI
over its non-null values will not show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import UsageError
from .profile import ColumnProfile, DatasetProfile
from .stats import psi, rebin

OK = "ok"
DRIFTED = "drifted"
TYPE_CHANGED = "type_changed"
ADDED = "added"
REMOVED = "removed"

#: Conventional PSI reading, used for the human-facing severity label.
NONE = "none"
MODERATE = "moderate"
MAJOR = "major"
UNKNOWN = "unknown"

MODERATE_PSI = 0.1
MAJOR_PSI = 0.25

DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_NULL_THRESHOLD = 0.05

#: Above this unique-values-per-row ratio, a frequency table is noise:
#: comparing the counts of values that each occur once measures nothing
#: but which particular ids happened to be in each extract.
NEAR_UNIQUE_RATIO = 0.9


def severity_of(value: float | None) -> str:
    """The conventional PSI band for ``value``."""
    if value is None:
        return UNKNOWN
    if value < MODERATE_PSI:
        return NONE
    if value < MAJOR_PSI:
        return MODERATE
    return MAJOR


@dataclass(frozen=True)
class ColumnDrift:
    """What happened to one column between two profiles."""

    name: str
    status: str
    baseline_type: str | None = None
    current_type: str | None = None
    psi: float | None = None
    baseline_null_rate: float | None = None
    current_null_rate: float | None = None
    baseline_distinct: int | None = None
    current_distinct: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def severity(self) -> str:
        """The PSI band, or ``unknown`` when no PSI could be taken."""
        return severity_of(self.psi)

    @property
    def null_rate_delta(self) -> float | None:
        """Change in null rate, current minus baseline."""
        if self.baseline_null_rate is None or self.current_null_rate is None:
            return None
        return self.current_null_rate - self.baseline_null_rate

    @property
    def is_change(self) -> bool:
        """Whether this column is something a reviewer has to look at."""
        return self.status != OK

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping."""
        return {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "baseline_type": self.baseline_type,
            "current_type": self.current_type,
            "psi": self.psi,
            "baseline_null_rate": self.baseline_null_rate,
            "current_null_rate": self.current_null_rate,
            "baseline_distinct": self.baseline_distinct,
            "current_distinct": self.current_distinct,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DriftReport:
    """The result of comparing a current profile against a baseline."""

    columns: tuple[ColumnDrift, ...]
    baseline_rows: int = 0
    current_rows: int = 0
    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    null_threshold: float = DEFAULT_NULL_THRESHOLD
    baseline_source: str = ""
    current_source: str = ""

    @property
    def changes(self) -> tuple[ColumnDrift, ...]:
        """Only the columns that are not ``ok``."""
        return tuple(column for column in self.columns if column.is_change)

    @property
    def has_drift(self) -> bool:
        """Whether anything at all moved."""
        return bool(self.changes)

    def column(self, name: str) -> ColumnDrift:
        """One column of the report by name."""
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping."""
        return {
            "format": "dataset-drift/1",
            "baseline_source": self.baseline_source,
            "current_source": self.current_source,
            "baseline_rows": self.baseline_rows,
            "current_rows": self.current_rows,
            "psi_threshold": self.psi_threshold,
            "null_threshold": self.null_threshold,
            "has_drift": self.has_drift,
            "columns": [column.to_dict() for column in self.columns],
        }


def compare(
    baseline: DatasetProfile,
    current: DatasetProfile,
    *,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
) -> DriftReport:
    """Compare ``current`` against ``baseline``.

    Column order follows the baseline, with columns that only exist in
    the current profile appended -- so a diff reads as "here is what the
    baseline said, and here is what is new".
    """
    if psi_threshold < 0:
        raise UsageError("--psi-threshold must not be negative")
    if not 0 <= null_threshold <= 1:
        raise UsageError("--null-threshold must be between 0 and 1")

    current_by_name = {column.name: column for column in current.columns}
    seen: set[str] = set()
    results: list[ColumnDrift] = []

    for old in baseline.columns:
        seen.add(old.name)
        new = current_by_name.get(old.name)
        if new is None:
            results.append(
                ColumnDrift(
                    name=old.name,
                    status=REMOVED,
                    baseline_type=old.inferred_type,
                    baseline_null_rate=old.null_rate,
                    baseline_distinct=old.distinct,
                    notes=("column is missing from the current dataset",),
                )
            )
            continue
        results.append(_compare_column(old, new, psi_threshold, null_threshold))

    for new in current.columns:
        if new.name in seen:
            continue
        results.append(
            ColumnDrift(
                name=new.name,
                status=ADDED,
                current_type=new.inferred_type,
                current_null_rate=new.null_rate,
                current_distinct=new.distinct,
                notes=("column is not in the baseline",),
            )
        )

    return DriftReport(
        columns=tuple(results),
        baseline_rows=baseline.row_count,
        current_rows=current.row_count,
        psi_threshold=psi_threshold,
        null_threshold=null_threshold,
        baseline_source=baseline.source,
        current_source=current.source,
    )


def _compare_column(
    old: ColumnProfile,
    new: ColumnProfile,
    psi_threshold: float,
    null_threshold: float,
) -> ColumnDrift:
    notes: list[str] = []
    status = OK

    if old.inferred_type != new.inferred_type:
        status = TYPE_CHANGED
        notes.append(f"type changed: {old.inferred_type} -> {new.inferred_type}")

    delta = new.null_rate - old.null_rate
    if abs(delta) > null_threshold:
        if status == OK:
            status = DRIFTED
        notes.append(
            f"null rate {_percent(old.null_rate)} -> {_percent(new.null_rate)}"
        )

    score = None if status == TYPE_CHANGED else _score(old, new, notes)
    if score is not None and score >= psi_threshold and status == OK:
        status = DRIFTED
    if score is not None and score >= psi_threshold:
        notes.append(f"distribution PSI {score:.3f} ({severity_of(score)})")

    if old.is_unique and not new.is_unique:
        if status == OK:
            status = DRIFTED
        notes.append("values are no longer unique")

    return ColumnDrift(
        name=old.name,
        status=status,
        baseline_type=old.inferred_type,
        current_type=new.inferred_type,
        psi=score,
        baseline_null_rate=old.null_rate,
        current_null_rate=new.null_rate,
        baseline_distinct=old.distinct,
        current_distinct=new.distinct,
        notes=tuple(notes),
    )


def _score(old: ColumnProfile, new: ColumnProfile, notes: list[str]) -> float | None:
    """PSI between two same-typed columns, or ``None`` if it cannot be taken."""
    if old.numeric is not None and new.numeric is not None:
        moved = rebin(new.numeric.counts, new.numeric.edges, old.numeric.edges)
        if sum(old.numeric.counts) == 0 or sum(moved) == 0:
            return None
        if old.numeric.sampled or new.numeric.sampled:
            notes.append("histogram is from a sample; PSI is approximate")
        return psi(old.numeric.counts, moved)

    if not old.categories or not new.categories:
        notes.append("not enough values to compare distributions")
        return None
    if max(old.unique_ratio, new.unique_ratio) > NEAR_UNIQUE_RATIO:
        notes.append("values are near-unique; frequency comparison skipped")
        return None
    if old.categories_truncated or new.categories_truncated:
        notes.append("category list was truncated; PSI is approximate")
    keys = sorted(set(old.categories) | set(new.categories))
    expected = [float(old.categories.get(key, 0)) for key in keys]
    actual = [float(new.categories.get(key, 0)) for key in keys]
    return psi(expected, actual)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"

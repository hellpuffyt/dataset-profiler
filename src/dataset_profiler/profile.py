"""What a profile is, how one is built, and how one is stored.

A :class:`DatasetProfile` is a value: it can be compared, serialised to
JSON, read back, and compared again with the same result. That matters
more than it sounds. The whole point of a baseline is that it outlives
the data it came from, so a profile that does not survive a round trip
through JSON is not a baseline, it is a printout.

Three deliberate limits are recorded in the profile itself rather than
being silently applied:

- ``distinct_truncated`` -- exact cardinality is counted up to
  ``max_distinct`` values and then stops. Counting further would mean
  keeping every distinct value of an id column in memory.
- ``categories_truncated`` -- the value/frequency table used for
  categorical drift keeps the first ``max_categories`` distinct values.
- ``sampled`` -- quantiles and the histogram come from a bounded,
  deterministic reservoir rather than from every value.

A consumer can therefore tell the difference between "this is exact" and
"this is the best a bounded profiler could do", which is the difference
between a number you can act on and one you cannot.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import types as coltypes
from .errors import BaselineError, ColumnError, UsageError
from .io import Table
from .stats import Reservoir, RunningMoments, bin_edges, histogram, quantile

#: Written into every serialised profile so a future format can be told apart.
FORMAT = "dataset-profile/1"

#: Quantiles stored for every numeric column.
QUANTILES = (("p01", 0.01), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p99", 0.99))

#: Decimal places floats are rounded to before storage.
PRECISION = 10


@dataclass(frozen=True)
class ProfileOptions:
    """The bounded-resource knobs, validated once at construction."""

    max_categories: int = 50
    max_distinct: int = 100_000
    sample_size: int = 20_000
    seed: int = 0
    bins: int = 10
    null_tokens: frozenset[str] = coltypes.DEFAULT_NULL_TOKENS

    def __post_init__(self) -> None:
        for name in ("max_categories", "max_distinct", "sample_size", "bins"):
            value = getattr(self, name)
            if value <= 0:
                raise UsageError(f"--{name.replace('_', '-')} must be positive")


@dataclass(frozen=True)
class NumericSummary:
    """The shape of the numeric values in one column."""

    count: int
    minimum: float
    maximum: float
    mean: float
    stddev: float
    quantiles: dict[str, float]
    edges: list[float]
    counts: list[int]
    sampled: bool

    @property
    def span(self) -> float:
        """Distance from the smallest value to the largest."""
        return self.maximum - self.minimum

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping."""
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "stddev": self.stddev,
            "quantiles": dict(self.quantiles),
            "histogram": {"edges": list(self.edges), "counts": list(self.counts)},
            "sampled": self.sampled,
        }

    @staticmethod
    def from_dict(raw: object, where: str) -> NumericSummary:
        """Rebuild from :meth:`to_dict` output, checking every field."""
        data = _mapping(raw, where)
        hist = _mapping(data.get("histogram"), f"{where}.histogram")
        edge_at = f"{where}.histogram.edges"
        count_at = f"{where}.histogram.counts"
        edges = [_number(e, edge_at) for e in _seq(hist.get("edges"), edge_at)]
        counts = [_integer(c, count_at) for c in _seq(hist.get("counts"), count_at)]
        if len(edges) != len(counts) + 1:
            raise BaselineError(f"{where}.histogram: edges and counts disagree")
        quantile_at = f"{where}.quantiles"
        quantiles = {
            str(key): _number(value, quantile_at)
            for key, value in _mapping(data.get("quantiles"), quantile_at).items()
        }
        return NumericSummary(
            count=_integer(data.get("count"), f"{where}.count"),
            minimum=_number(data.get("min"), f"{where}.min"),
            maximum=_number(data.get("max"), f"{where}.max"),
            mean=_number(data.get("mean"), f"{where}.mean"),
            stddev=_number(data.get("stddev"), f"{where}.stddev"),
            quantiles=quantiles,
            edges=edges,
            counts=counts,
            sampled=bool(data.get("sampled", False)),
        )


@dataclass(frozen=True)
class ColumnProfile:
    """Everything known about one column."""

    name: str
    count: int
    null_count: int
    inferred_type: str
    type_counts: dict[str, int] = field(default_factory=dict)
    null_tokens: dict[str, int] = field(default_factory=dict)
    distinct: int = 0
    distinct_truncated: bool = False
    categories: dict[str, int] = field(default_factory=dict)
    categories_truncated: bool = False
    min_length: int | None = None
    max_length: int | None = None
    numeric: NumericSummary | None = None

    @property
    def filled_count(self) -> int:
        """Rows where this column had an actual value."""
        return self.count - self.null_count

    @property
    def null_rate(self) -> float:
        """Fraction of rows that were null; ``0.0`` for an empty dataset."""
        return self.null_count / self.count if self.count else 0.0

    @property
    def unique_ratio(self) -> float:
        """Distinct values per non-null row; ``1.0`` means a candidate key."""
        return self.distinct / self.filled_count if self.filled_count else 0.0

    @property
    def is_constant(self) -> bool:
        """One value, repeated -- usually a column nobody meant to ship."""
        return self.distinct == 1 and not self.distinct_truncated

    @property
    def is_unique(self) -> bool:
        """Every non-null value differs, and none was missing."""
        return (
            not self.distinct_truncated
            and self.filled_count > 0
            and self.distinct == self.filled_count
            and self.null_count == 0
        )

    @property
    def is_empty(self) -> bool:
        """Never once had a value."""
        return self.filled_count == 0

    def top_values(self, limit: int = 5) -> list[tuple[str, int]]:
        """The most frequent values, ties broken alphabetically."""
        ordered = sorted(self.categories.items(), key=lambda item: (-item[1], item[0]))
        return ordered[:limit]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping."""
        return {
            "name": self.name,
            "count": self.count,
            "null_count": self.null_count,
            "inferred_type": self.inferred_type,
            "type_counts": dict(self.type_counts),
            "null_tokens": dict(self.null_tokens),
            "distinct": self.distinct,
            "distinct_truncated": self.distinct_truncated,
            "categories": dict(self.categories),
            "categories_truncated": self.categories_truncated,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "numeric": self.numeric.to_dict() if self.numeric else None,
        }

    @staticmethod
    def from_dict(raw: object, where: str = "column") -> ColumnProfile:
        """Rebuild from :meth:`to_dict` output, checking every field."""
        data = _mapping(raw, where)
        name = _text(data.get("name"), f"{where}.name")
        spot = f"column {name!r}"
        numeric_raw = data.get("numeric")
        return ColumnProfile(
            name=name,
            count=_integer(data.get("count"), f"{spot}.count"),
            null_count=_integer(data.get("null_count"), f"{spot}.null_count"),
            inferred_type=_text(data.get("inferred_type"), f"{spot}.inferred_type"),
            type_counts=_counts(data.get("type_counts"), f"{spot}.type_counts"),
            null_tokens=_counts(data.get("null_tokens"), f"{spot}.null_tokens"),
            distinct=_integer(data.get("distinct"), f"{spot}.distinct"),
            distinct_truncated=bool(data.get("distinct_truncated", False)),
            categories=_counts(data.get("categories"), f"{spot}.categories"),
            categories_truncated=bool(data.get("categories_truncated", False)),
            min_length=_optional_integer(data.get("min_length"), f"{spot}.min_length"),
            max_length=_optional_integer(data.get("max_length"), f"{spot}.max_length"),
            numeric=(
                None
                if numeric_raw is None
                else NumericSummary.from_dict(numeric_raw, f"{spot}.numeric")
            ),
        )


@dataclass(frozen=True)
class DatasetProfile:
    """A whole dataset, profiled."""

    row_count: int
    columns: tuple[ColumnProfile, ...]
    source: str = ""
    ragged_rows: int = 0

    @property
    def names(self) -> list[str]:
        """Column names, in the order the dataset has them."""
        return [column.name for column in self.columns]

    def column(self, name: str) -> ColumnProfile:
        """One column by name, or :class:`~.errors.ColumnError`."""
        for column in self.columns:
            if column.name == name:
                return column
        raise ColumnError(name, self.names)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping, tagged with the format version."""
        return {
            "format": FORMAT,
            "source": self.source,
            "row_count": self.row_count,
            "ragged_rows": self.ragged_rows,
            "columns": [column.to_dict() for column in self.columns],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """The profile as JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @staticmethod
    def from_dict(raw: object, path: str | None = None) -> DatasetProfile:
        """Rebuild from :meth:`to_dict` output.

        The format tag is checked first and by itself, because the most
        common mistake is pointing ``--baseline`` at some other JSON file
        and the message for that should say so.
        """
        try:
            data = _mapping(raw, "profile")
            found = data.get("format")
            if found != FORMAT:
                raise BaselineError(
                    f"not a dataset profile (expected format {FORMAT!r}, "
                    f"found {found!r})"
                )
            columns = tuple(
                ColumnProfile.from_dict(item, "column")
                for item in _seq(data.get("columns"), "profile.columns")
            )
            return DatasetProfile(
                row_count=_integer(data.get("row_count"), "profile.row_count"),
                columns=columns,
                source=str(data.get("source", "")),
                ragged_rows=_integer(data.get("ragged_rows", 0), "profile.ragged_rows"),
            )
        except BaselineError as error:
            if path is None or error.path is not None:
                raise
            raise BaselineError(str(error), path) from error

    @staticmethod
    def from_json(text: str, path: str | None = None) -> DatasetProfile:
        """Parse JSON text into a profile."""
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise BaselineError(f"invalid JSON: {error.msg}", path) from error
        return DatasetProfile.from_dict(raw, path)


class _ColumnAccumulator:
    """Builds one :class:`ColumnProfile` from values fed one at a time."""

    def __init__(self, name: str, options: ProfileOptions) -> None:
        self.name = name
        self._options = options
        self._count = 0
        self._null_count = 0
        self._type_counts: dict[str, int] = {}
        self._null_tokens: dict[str, int] = {}
        self._distinct: set[str] = set()
        self._distinct_truncated = False
        self._categories: dict[str, int] = {}
        self._categories_truncated = False
        self._min_length: int | None = None
        self._max_length: int | None = None
        self._moments = RunningMoments()
        self._reservoir = Reservoir(options.sample_size, options.seed)

    def add(self, raw: str) -> None:
        """Fold one field in."""
        self._count += 1
        kind = coltypes.classify(raw, self._options.null_tokens)
        self._type_counts[kind] = self._type_counts.get(kind, 0) + 1
        if kind == coltypes.EMPTY:
            self._null_count += 1
            token = coltypes.null_token(raw)
            self._null_tokens[token] = self._null_tokens.get(token, 0) + 1
            return

        value = raw.strip()
        length = len(value)
        if self._min_length is None or length < self._min_length:
            self._min_length = length
        if self._max_length is None or length > self._max_length:
            self._max_length = length

        if len(self._distinct) < self._options.max_distinct:
            self._distinct.add(value)
        elif value not in self._distinct:
            self._distinct_truncated = True

        if value in self._categories:
            self._categories[value] += 1
        elif len(self._categories) < self._options.max_categories:
            self._categories[value] = 1
        else:
            self._categories_truncated = True

        if kind in coltypes.NUMERIC_KINDS:
            number = coltypes.to_number(value)
            if number is not None:
                self._moments.add(number)
                self._reservoir.add(number)

    def _numeric(self) -> NumericSummary | None:
        if not self._moments.count:
            return None
        sample = self._reservoir.sorted_values()
        edges = bin_edges(
            self._moments.minimum, self._moments.maximum, self._options.bins
        )
        return NumericSummary(
            count=self._moments.count,
            minimum=_round(self._moments.minimum),
            maximum=_round(self._moments.maximum),
            mean=_round(self._moments.mean),
            stddev=_round(self._moments.stddev),
            quantiles={
                label: _round(quantile(sample, q)) for label, q in QUANTILES
            },
            edges=[_round(edge) for edge in edges],
            counts=histogram(sample, edges),
            sampled=not self._reservoir.complete,
        )

    def finish(self) -> ColumnProfile:
        """The finished profile for this column."""
        return ColumnProfile(
            name=self.name,
            count=self._count,
            null_count=self._null_count,
            inferred_type=coltypes.infer(self._type_counts),
            type_counts=dict(self._type_counts),
            null_tokens=dict(self._null_tokens),
            distinct=len(self._distinct),
            distinct_truncated=self._distinct_truncated,
            categories=dict(self._categories),
            categories_truncated=self._categories_truncated,
            min_length=self._min_length,
            max_length=self._max_length,
            numeric=self._numeric(),
        )


def profile_rows(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    options: ProfileOptions | None = None,
    source: str = "",
    select: Sequence[str] | None = None,
) -> DatasetProfile:
    """Profile ``rows`` against ``header``.

    Rows are consumed lazily and never kept, so the memory this uses is
    set by the options, not by the length of the input.
    """
    settings = options or ProfileOptions()
    names = list(header)
    wanted = _resolve_selection(names, select)
    accumulators = [_ColumnAccumulator(names[index], settings) for index in wanted]
    row_count = 0
    for row in rows:
        row_count += 1
        for accumulator, index in zip(accumulators, wanted, strict=True):
            accumulator.add(row[index] if index < len(row) else "")
    return DatasetProfile(
        row_count=row_count,
        columns=tuple(accumulator.finish() for accumulator in accumulators),
        source=source,
    )


def profile_table(
    table: Table,
    *,
    options: ProfileOptions | None = None,
    source: str = "",
    select: Sequence[str] | None = None,
) -> DatasetProfile:
    """Profile a :class:`~.io.Table`, carrying its ragged-row count over."""
    profile = profile_rows(
        table.header, table.rows, options=options, source=source, select=select
    )
    return DatasetProfile(
        row_count=profile.row_count,
        columns=profile.columns,
        source=profile.source,
        ragged_rows=table.ragged,
    )


def _resolve_selection(names: Sequence[str], select: Sequence[str] | None) -> list[int]:
    """Column indexes to profile, in the order asked for."""
    if select is None:
        return list(range(len(names)))
    indexes: list[int] = []
    for name in select:
        try:
            indexes.append(list(names).index(name))
        except ValueError:
            raise ColumnError(name, list(names)) from None
    return indexes


def _round(value: float) -> float:
    """Round for storage, so a profile equals itself after a round trip."""
    return round(value, PRECISION)


def _mapping(raw: object, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise BaselineError(f"{where}: expected an object, found {_kind(raw)}")
    return raw


def _seq(raw: object, where: str) -> list[Any]:
    if not isinstance(raw, list):
        raise BaselineError(f"{where}: expected a list, found {_kind(raw)}")
    return raw


def _integer(raw: object, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise BaselineError(f"{where}: expected an integer, found {_kind(raw)}")
    return raw


def _optional_integer(raw: object, where: str) -> int | None:
    return None if raw is None else _integer(raw, where)


def _number(raw: object, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BaselineError(f"{where}: expected a number, found {_kind(raw)}")
    return float(raw)


def _text(raw: object, where: str) -> str:
    if not isinstance(raw, str):
        raise BaselineError(f"{where}: expected a string, found {_kind(raw)}")
    return raw


def _counts(raw: object, where: str) -> dict[str, int]:
    data = _mapping(raw, where)
    return {str(key): _integer(value, where) for key, value in data.items()}


def _kind(raw: object) -> str:
    return "null" if raw is None else type(raw).__name__

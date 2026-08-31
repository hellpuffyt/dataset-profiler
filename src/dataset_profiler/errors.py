"""Errors raised by dataset_profiler.

Every failure a user can cause is one of these. A profiler exists to be
believed, so it must never half-answer: a baseline that cannot be parsed,
a column that is not in the data, or a threshold that is not a number is
a refusal to start, not a warning buried in a report.
"""

from __future__ import annotations


class ProfilerError(Exception):
    """Base class for every error this package raises deliberately."""


class UsageError(ProfilerError):
    """The command line asked for something that cannot be done."""


class DataError(ProfilerError):
    """The input table could not be read as a table."""


class ColumnError(ProfilerError):
    """A column was referenced that the dataset does not have."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        known = ", ".join(available) if available else "(none)"
        super().__init__(f"no such column {name!r}; dataset has: {known}")


class BaselineError(ProfilerError):
    """A stored baseline profile is missing, malformed, or the wrong shape.

    ``path`` names the file when the baseline came from one, because the
    usual cause is pointing at the wrong JSON entirely.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)

"""dataset_profiler — profile a tabular dataset and detect drift.

Two things live here. :func:`dataset_profiler.profile.profile_rows` turns
rows into a :class:`~dataset_profiler.profile.DatasetProfile` — inferred
types, cardinality, null patterns and numeric shape. ``compare`` in
:mod:`dataset_profiler.drift` takes two of those profiles and reports what
moved. The command line in :mod:`dataset_profiler.cli` is a thin wrapper
over both.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

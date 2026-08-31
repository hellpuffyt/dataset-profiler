"""Turning profiles and drift reports into something to read.

Three renderings, for three different readers.

``text`` is for a person at a terminal: aligned columns, and a "detail"
field that says the most useful single thing about each column rather
than everything known about it.

``tsv`` is for the next program in the pipe -- one line per column, no
alignment, no colour, stable field order. It is what the CI job in this
repository asserts against, which is also why it is the format least
likely to change under anyone.

``json`` is the full structure, and is the same shape that
``--baseline`` reads back.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .drift import DriftReport
from .profile import ColumnProfile, DatasetProfile

TEXT = "text"
TSV = "tsv"
JSON = "json"
FORMATS = (TEXT, TSV, JSON)


def format_number(value: float) -> str:
    """Compact, stable rendering of a number for human output."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def format_rate(value: float) -> str:
    """A fraction as a percentage with one decimal place."""
    return f"{value * 100:.1f}%"


def render_table(rows: Sequence[Sequence[str]], rule_after_header: bool = True) -> str:
    """Left-aligned fixed-width table; the first row is the header."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    sizes = [max(len(row[index]) for row in padded) for index in range(width)]
    lines = []
    for position, row in enumerate(padded):
        cells = [cell.ljust(sizes[index]) for index, cell in enumerate(row)]
        lines.append("  ".join(cells).rstrip())
        if position == 0 and rule_after_header:
            lines.append("  ".join("-" * size for size in sizes))
    return "\n".join(lines)


def column_detail(column: ColumnProfile) -> str:
    """The one or two facts most worth knowing about a column."""
    parts: list[str] = []
    if column.is_empty:
        return "never populated"
    if column.is_unique:
        parts.append("unique")
    elif column.is_constant:
        top = column.top_values(1)
        parts.append(f"constant {top[0][0]!r}" if top else "constant")

    if column.numeric is not None:
        summary = column.numeric
        parts.append(
            f"range {format_number(summary.minimum)}"
            f"..{format_number(summary.maximum)}"
        )
        parts.append(f"mean {format_number(summary.mean)}")
    else:
        top = column.top_values(1)
        if top and not column.is_unique and not column.is_constant:
            parts.append(f"top {top[0][0]!r} x{top[0][1]}")
        if column.min_length is not None and column.max_length is not None:
            parts.append(f"len {column.min_length}..{column.max_length}")

    spellings = sorted(
        ((token, count) for token, count in column.null_tokens.items() if token),
        key=lambda item: (-item[1], item[0]),
    )
    if spellings:
        shown = ", ".join(f"{token!r} x{count}" for token, count in spellings[:2])
        parts.append(f"null spelt {shown}")
    return "; ".join(parts)


def render_profile_text(profile: DatasetProfile) -> str:
    """A profile as an aligned report."""
    head = [
        f"dataset: {profile.source or '(stdin)'}",
        f"rows: {profile.row_count}, columns: {len(profile.columns)}",
    ]
    if profile.ragged_rows:
        head.append(
            f"warning: {profile.ragged_rows} row(s) did not match the header width"
        )
    if not profile.columns:
        return "\n".join([*head, "", "no columns"])

    rows: list[list[str]] = [["column", "type", "nulls", "distinct", "detail"]]
    for column in profile.columns:
        distinct = f"{column.distinct}"
        if column.distinct_truncated:
            distinct = f">={column.distinct}"
        rows.append(
            [
                column.name,
                column.inferred_type,
                f"{column.null_count} ({format_rate(column.null_rate)})",
                distinct,
                column_detail(column),
            ]
        )
    return "\n".join([*head, "", render_table(rows)])


def render_profile_tsv(profile: DatasetProfile) -> str:
    """A profile as one tab-separated line per column."""
    lines = ["column\ttype\tnulls\tdistinct\tunique_ratio"]
    for column in profile.columns:
        lines.append(
            "\t".join(
                [
                    column.name,
                    column.inferred_type,
                    str(column.null_count),
                    str(column.distinct),
                    f"{column.unique_ratio:.4f}",
                ]
            )
        )
    return "\n".join(lines)


def render_profile_json(profile: DatasetProfile) -> str:
    """A profile as JSON -- the same text ``--baseline`` reads."""
    return profile.to_json()


def render_drift_text(report: DriftReport) -> str:
    """A drift report as an aligned report."""
    head = [
        f"baseline: {report.baseline_source or '(unnamed)'} "
        f"({report.baseline_rows} rows)",
        f"current:  {report.current_source or '(stdin)'} "
        f"({report.current_rows} rows)",
        f"thresholds: psi >= {report.psi_threshold:g}, "
        f"null rate delta > {format_rate(report.null_threshold)}",
    ]
    rows: list[list[str]] = [["column", "status", "severity", "psi", "notes"]]
    for column in report.columns:
        rows.append(
            [
                column.name,
                column.status,
                column.severity,
                "-" if column.psi is None else f"{column.psi:.3f}",
                "; ".join(column.notes),
            ]
        )
    tail = (
        f"{len(report.changes)} of {len(report.columns)} column(s) changed"
        if report.has_drift
        else "no drift detected"
    )
    body = render_table(rows) if report.columns else "no columns"
    return "\n".join([*head, "", body, "", tail])


def render_drift_tsv(report: DriftReport) -> str:
    """A drift report as one tab-separated line per column."""
    lines = ["column\tstatus\tseverity\tpsi"]
    for column in report.columns:
        lines.append(
            "\t".join(
                [
                    column.name,
                    column.status,
                    column.severity,
                    "-" if column.psi is None else f"{column.psi:.4f}",
                ]
            )
        )
    return "\n".join(lines)


def render_drift_json(report: DriftReport) -> str:
    """A drift report as JSON."""
    return json.dumps(report.to_dict(), indent=2)


def render_profile(profile: DatasetProfile, fmt: str) -> str:
    """Dispatch to the renderer named by ``fmt``."""
    if fmt == TEXT:
        return render_profile_text(profile)
    if fmt == TSV:
        return render_profile_tsv(profile)
    if fmt == JSON:
        return render_profile_json(profile)
    raise ValueError(f"unknown format: {fmt}")


def render_drift(report: DriftReport, fmt: str) -> str:
    """Dispatch to the renderer named by ``fmt``."""
    if fmt == TEXT:
        return render_drift_text(report)
    if fmt == TSV:
        return render_drift_tsv(report)
    if fmt == JSON:
        return render_drift_json(report)
    raise ValueError(f"unknown format: {fmt}")

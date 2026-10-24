"""The ``dataset-profiler`` command line.

Two subcommands, because there are two questions. ``profile`` answers
"what is in this file"; ``compare`` answers "what changed since the
baseline". Both share the reading and profiling flags, which are defined
once in :func:`add_profile_options` so the two can never drift apart --
a baseline written with different options than the profile compared
against it would be a silent source of false alarms.

Exit codes are part of the interface:

===  =========================================================
0    success, and for ``compare`` no drift worth failing over
1    the input, the baseline or the command line was unusable
2    ``compare --fail-on-drift`` found drift
===  =========================================================
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

from . import __version__
from .drift import DEFAULT_NULL_THRESHOLD, DEFAULT_PSI_THRESHOLD, compare
from .errors import BaselineError, ProfilerError, UsageError
from .io import open_input, open_output, read_table
from .profile import DatasetProfile, ProfileOptions, profile_table
from .report import FORMATS, TEXT, render_drift, render_profile
from .types import DEFAULT_NULL_TOKENS

PROGRAM = "dataset-profiler"

#: Exit status for ``compare --fail-on-drift`` when drift was found.
DRIFT_EXIT = 2


def add_profile_options(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand that reads and profiles a file."""
    parser.add_argument("input", nargs="?", help="input CSV, or - for stdin")
    parser.add_argument("-o", "--output", help="write here instead of stdout")
    parser.add_argument(
        "-f",
        "--format",
        choices=FORMATS,
        default=TEXT,
        help="output format (default: text)",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        help="input delimiter; sniffed from the file when omitted",
    )
    parser.add_argument("--encoding", default="utf-8", help="text encoding")
    parser.add_argument(
        "--columns", help="comma-separated subset of columns to profile"
    )
    parser.add_argument(
        "--null-tokens",
        help=(
            "comma-separated text that counts as missing, replacing the "
            "default set (empty, NA, N/A, NaN, null, none, nil, -, ?)"
        ),
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=ProfileOptions.max_categories,
        help="distinct values kept per column for frequency comparison",
    )
    parser.add_argument(
        "--max-distinct",
        type=int,
        default=ProfileOptions.max_distinct,
        help="cardinality counted exactly up to this many values",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=ProfileOptions.sample_size,
        help="numeric values sampled per column for quantiles and histogram",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the numeric sample"
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="histogram bins per numeric column"
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Profile a tabular dataset -- inferred types, cardinality and "
            "null patterns -- and measure distribution drift against a "
            "stored baseline."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    profile_parser = subcommands.add_parser(
        "profile",
        help="describe a dataset",
        description=(
            "Profile a dataset. Write the result with --format json to keep "
            "it as a baseline for a later compare."
        ),
    )
    add_profile_options(profile_parser)

    compare_parser = subcommands.add_parser(
        "compare",
        help="measure drift against a stored baseline",
        description=(
            "Profile a dataset and compare it against a baseline profile. "
            "Exits 2 with --fail-on-drift when anything moved."
        ),
    )
    add_profile_options(compare_parser)
    compare_parser.add_argument(
        "-b",
        "--baseline",
        required=True,
        help="baseline profile written by: dataset-profiler profile --format json",
    )
    compare_parser.add_argument(
        "--psi-threshold",
        type=float,
        default=DEFAULT_PSI_THRESHOLD,
        help=f"PSI at or above which a column has drifted "
        f"(default: {DEFAULT_PSI_THRESHOLD})",
    )
    compare_parser.add_argument(
        "--null-threshold",
        type=float,
        default=DEFAULT_NULL_THRESHOLD,
        help=f"null-rate change treated as drift "
        f"(default: {DEFAULT_NULL_THRESHOLD})",
    )
    compare_parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help=f"exit {DRIFT_EXIT} when any column changed",
    )
    return parser


def parse_columns(value: str | None) -> list[str] | None:
    """Split a ``--columns`` value, rejecting one that selects nothing."""
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise UsageError("--columns selected no columns")
    return names


def parse_null_tokens(value: str | None) -> frozenset[str]:
    """Build the null-token set, keeping the empty string always."""
    if value is None:
        return DEFAULT_NULL_TOKENS
    tokens = {token.strip().lower() for token in value.split(",")}
    tokens.add("")
    return frozenset(tokens)


def options_from(args: argparse.Namespace) -> ProfileOptions:
    """Build validated :class:`~.profile.ProfileOptions` from parsed flags."""
    return ProfileOptions(
        max_categories=args.max_categories,
        max_distinct=args.max_distinct,
        sample_size=args.sample_size,
        seed=args.seed,
        bins=args.bins,
        null_tokens=parse_null_tokens(args.null_tokens),
    )


def load_baseline(path: str) -> DatasetProfile:
    """Read a stored baseline profile."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise BaselineError(f"cannot read baseline: {error.strerror}", path) from error
    return DatasetProfile.from_json(text, path)


def build_profile(args: argparse.Namespace) -> DatasetProfile:
    """Read the input named on the command line and profile it."""
    options = options_from(args)
    select = parse_columns(args.columns)
    with open_input(args.input, args.encoding) as handle:
        table = read_table(handle, args.delimiter)
        return profile_table(
            table,
            options=options,
            source="" if args.input in (None, "-") else args.input,
            select=select,
        )


def run_profile(args: argparse.Namespace) -> int:
    """The ``profile`` subcommand."""
    profile = build_profile(args)
    _emit(args, render_profile(profile, args.format))
    return 0


def run_compare(args: argparse.Namespace) -> int:
    """The ``compare`` subcommand."""
    baseline = load_baseline(args.baseline)
    current = build_profile(args)
    report = compare(
        baseline,
        current,
        psi_threshold=args.psi_threshold,
        null_threshold=args.null_threshold,
    )
    _emit(args, render_drift(report, args.format))
    return DRIFT_EXIT if args.fail_on_drift and report.has_drift else 0


def _emit(args: argparse.Namespace, text: str) -> None:
    with open_output(args.output, args.encoding) as handle:
        handle.write(text)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "profile":
            return run_profile(args)
        return run_compare(args)
    except ProfilerError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as error:
        print(
            f"{PROGRAM}: input is not valid {args.encoding} ({error.reason} at byte "
            f"{error.start}); pass --encoding",
            file=sys.stderr,
        )
        return 1
    except LookupError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return 1
    except csv.Error as error:
        print(f"{PROGRAM}: malformed CSV input: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # pragma: no cover - depends on the reader
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

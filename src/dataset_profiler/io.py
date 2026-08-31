"""Reading tables in, without surprises.

A profiler is pointed at files nobody has looked at yet -- that is the
point of it -- so reading has to survive what real exports contain:
semicolons instead of commas, a UTF-8 BOM from a spreadsheet, rows with
the wrong number of fields, duplicate header names.

Every one of those is handled rather than raised on, because a profile
that says "3 rows were ragged" is useful and a traceback is not. The one
thing that *is* an error is a file with no header at all, since there is
then nothing to profile.
"""

from __future__ import annotations

import csv
import io
import itertools
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TextIO

from .errors import DataError

#: Delimiters worth sniffing for. Anything else needs ``--delimiter``.
CANDIDATE_DELIMITERS = ",;\t|"

#: How many characters to sniff the dialect from.
SNIFF_BYTES = 8192


class Table:
    """A header plus a lazy stream of equal-length rows.

    ``ragged`` counts rows whose field count did not match the header. It
    is only final once ``rows`` has been consumed, which is why it is an
    attribute rather than a constructor argument.
    """

    def __init__(self, header: list[str], rows: Iterator[list[str]]) -> None:
        self.header = header
        self.rows = rows
        self.ragged = 0


def sniff_delimiter(sample: str) -> str:
    """Guess the delimiter of ``sample``, falling back to a comma.

    The standard library sniffer is happy to return a letter that happens
    to appear regularly, so its answer is only accepted when it is one of
    the delimiters a real CSV actually uses.
    """
    if not sample.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=CANDIDATE_DELIMITERS)
    except csv.Error:
        return ","
    found = dialect.delimiter
    return found if found in CANDIDATE_DELIMITERS else ","


def dedupe_header(names: Iterable[str]) -> list[str]:
    """Make header names unique and non-empty, preserving order.

    A blank name becomes ``column_1`` (1-based, matching how people count
    columns in a spreadsheet); a repeat becomes ``name_2``. Silently
    letting duplicates collide would make one column of the profile a
    quiet lie about two columns of the data.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for index, raw in enumerate(names):
        name = raw.strip() or f"column_{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
            while name in seen:
                seen[name] = seen.get(name, 1) + 1
                name = f"{name}_{seen[name]}"
        seen[name] = 1
        out.append(name)
    return out


def read_table(handle: TextIO, delimiter: str | None = None) -> Table:
    """Read ``handle`` as a table, sniffing the dialect when not told.

    The sniffed sample is pushed back in front of the stream, so the
    header is never eaten by the sniff -- the classic way to lose the
    first row of every file.
    """
    sample = handle.read(SNIFF_BYTES)
    if sample.startswith("﻿"):
        sample = sample[1:]
    # Finish the line the sample stopped in the middle of, then hand the
    # reader the sample followed by the rest of the stream. Reading the
    # whole file to push the sample back would make the memory this uses
    # a function of the file size, which is the one thing it must not be.
    sample += handle.readline()
    stream: Iterable[str] = itertools.chain(io.StringIO(sample), handle)
    if delimiter is None:
        delimiter = sniff_delimiter(sample)
    elif len(delimiter) != 1:
        raise DataError(f"delimiter must be one character, got {delimiter!r}")

    reader = csv.reader(stream, delimiter=delimiter)
    try:
        raw_header = next(reader)
    except StopIteration:
        raise DataError("the input has no header row") from None
    header = dedupe_header(raw_header)
    table = Table(header, iter(()))
    table.rows = _normalise(reader, len(header), table)
    return table


def _normalise(
    reader: Iterable[list[str]], width: int, table: Table
) -> Iterator[list[str]]:
    """Pad short rows, truncate long ones, and count both as ragged.

    A blank line reaches us as ``[]``, which is ambiguous: in a
    single-column file it is an empty value, and in any other file it is
    formatting. It is treated as each of those in turn, because dropping
    the blank lines of a one-column file would silently delete exactly
    the rows a profiler exists to count.
    """
    for row in reader:
        if not row:
            if width != 1:
                continue
            row = [""]
        if len(row) == width:
            yield row
            continue
        table.ragged += 1
        if len(row) < width:
            yield row + [""] * (width - len(row))
        else:
            yield row[:width]


@contextmanager
def open_input(path: str | None, encoding: str = "utf-8") -> Iterator[TextIO]:
    """Open ``path`` for reading; ``None`` or ``-`` means stdin."""
    if path is None or path == "-":
        yield sys.stdin
        return
    try:
        # The handle is owned by this context manager and closed below.
        handle = open(path, encoding=encoding, newline="")  # noqa: SIM115
    except OSError as error:
        raise DataError(f"cannot read {path}: {error.strerror}") from error
    try:
        yield handle
    finally:
        handle.close()


@contextmanager
def open_output(path: str | None, encoding: str = "utf-8") -> Iterator[TextIO]:
    """Open ``path`` for writing; ``None`` or ``-`` means stdout."""
    if path is None or path == "-":
        yield sys.stdout
        return
    try:
        # The handle is owned by this context manager and closed below.
        handle = open(path, "w", encoding=encoding, newline="")  # noqa: SIM115
    except OSError as error:
        raise DataError(f"cannot write {path}: {error.strerror}") from error
    try:
        yield handle
    finally:
        handle.close()

"""Reading tables: dialects, BOMs, ragged rows and awkward headers."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from dataset_profiler.errors import DataError
from dataset_profiler.io import (
    SNIFF_BYTES,
    dedupe_header,
    open_input,
    open_output,
    read_table,
    sniff_delimiter,
)

CSV = "name,age\nAda,36\nBob,9\n"


def read(text: str, delimiter: str | None = None) -> tuple[list[str], list[list[str]]]:
    table = read_table(io.StringIO(text), delimiter)
    rows = list(table.rows)
    return table.header, rows


def test_a_plain_file_reads_header_and_rows() -> None:
    header, rows = read(CSV)
    assert header == ["name", "age"]
    assert rows == [["Ada", "36"], ["Bob", "9"]]


def test_the_sniffed_sample_is_not_swallowed() -> None:
    """Sniffing consumes the stream; the header has to be pushed back."""
    header, rows = read(CSV)
    assert header == ["name", "age"]
    assert len(rows) == 2


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_common_delimiters_are_detected(delimiter: str) -> None:
    text = f"name{delimiter}age\nAda{delimiter}36\nBob{delimiter}9\n"
    header, rows = read(text)
    assert header == ["name", "age"]
    assert rows[0] == ["Ada", "36"]


def test_an_explicit_delimiter_overrides_sniffing() -> None:
    header, rows = read("name;age\nAda;36\n", delimiter=";")
    assert header == ["name", "age"]
    assert rows[0] == ["Ada", "36"]


def test_a_multi_character_delimiter_is_rejected() -> None:
    with pytest.raises(DataError, match="one character"):
        read(CSV, delimiter="::")


def test_a_spreadsheet_byte_order_mark_is_stripped() -> None:
    """Otherwise the first column is named '\\ufeffname' and never matches."""
    header, _ = read("﻿" + CSV)
    assert header == ["name", "age"]


def test_quoted_fields_survive_commas_and_newlines() -> None:
    text = 'name,note\n"Ada","a, b\nc"\n'
    _, rows = read(text)
    assert rows == [["Ada", "a, b\nc"]]


def test_short_rows_are_padded_and_counted() -> None:
    table = read_table(io.StringIO("a,b,c\n1,2\n"))
    rows = list(table.rows)
    assert rows == [["1", "2", ""]]
    assert table.ragged == 1


def test_long_rows_are_truncated_and_counted() -> None:
    table = read_table(io.StringIO("a,b\n1,2,3,4\n"))
    rows = list(table.rows)
    assert rows == [["1", "2"]]
    assert table.ragged == 1


def test_well_formed_rows_are_not_counted_as_ragged() -> None:
    table = read_table(io.StringIO(CSV))
    list(table.rows)
    assert table.ragged == 0


def test_ragged_count_is_only_final_after_consumption() -> None:
    table = read_table(io.StringIO("a,b\n1\n2\n"))
    assert table.ragged == 0
    list(table.rows)
    assert table.ragged == 2


def test_blank_lines_are_skipped_without_counting_as_ragged() -> None:
    table = read_table(io.StringIO("a,b\n1,2\n\n3,4\n"))
    assert list(table.rows) == [["1", "2"], ["3", "4"]]
    assert table.ragged == 0


def test_a_blank_line_in_a_one_column_file_is_a_value_not_formatting() -> None:
    """Dropping it would delete exactly the rows a profiler exists to count."""
    table = read_table(io.StringIO("v\n1\n\n2\n"))
    assert list(table.rows) == [["1"], [""], ["2"]]
    assert table.ragged == 0


class CountingStream:
    """A text stream that records how much of itself has been handed out."""

    def __init__(self, text: str) -> None:
        self._inner = io.StringIO(text)
        self.consumed = 0

    def read(self, size: int = -1) -> str:
        data = self._inner.read(size)
        self.consumed += len(data)
        return data

    def readline(self) -> str:
        data = self._inner.readline()
        self.consumed += len(data)
        return data

    def __iter__(self) -> CountingStream:
        return self

    def __next__(self) -> str:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


def test_reading_the_header_does_not_consume_the_whole_file() -> None:
    """The sniff must push its sample back, not slurp the file to do it."""
    stream = CountingStream("a,b\n" + "1,2\n" * 50_000)
    table = read_table(stream, ",")  # type: ignore[arg-type]
    assert table.header == ["a", "b"]
    assert stream.consumed <= SNIFF_BYTES + 32
    assert sum(1 for _ in table.rows) == 50_000


def test_an_empty_input_has_no_header_to_profile() -> None:
    with pytest.raises(DataError, match="no header"):
        read("")


def test_a_header_only_file_profiles_zero_rows() -> None:
    header, rows = read("a,b\n")
    assert header == ["a", "b"]
    assert rows == []


def test_duplicate_header_names_are_made_unique() -> None:
    """Two columns must not collapse into one lie about both of them."""
    assert dedupe_header(["a", "a", "a"]) == ["a", "a_2", "a_3"]


def test_deduping_avoids_colliding_with_a_name_already_present() -> None:
    assert dedupe_header(["a", "a_2", "a"]) == ["a", "a_2", "a_2_2"]


def test_blank_header_names_get_a_positional_name() -> None:
    assert dedupe_header(["", " ", "x"]) == ["column_1", "column_2", "x"]


def test_header_names_are_stripped() -> None:
    header, _ = read(" name , age \nAda,36\n", delimiter=",")
    assert header == ["name", "age"]


def test_sniffing_an_unsplittable_sample_falls_back_to_comma() -> None:
    assert sniff_delimiter("just one column\nvalue\n") == ","


def test_sniffing_blank_text_falls_back_to_comma() -> None:
    assert sniff_delimiter("   ") == ","


def test_sniffing_never_returns_a_letter() -> None:
    """The stdlib sniffer will happily pick a regularly spaced letter."""
    assert sniff_delimiter("axbxc\n1x2x3\n") in ",;\t|"


def test_open_input_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text(CSV, encoding="utf-8")
    with open_input(str(path)) as handle:
        assert handle.read().replace("\r\n", "\n") == CSV


def test_open_input_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot read"), open_input(
        str(tmp_path / "nope.csv")
    ):
        pass


@pytest.mark.parametrize("path", [None, "-"])
def test_open_input_falls_back_to_stdin(
    path: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(CSV))
    with open_input(path) as handle:
        assert handle.read() == CSV


def test_open_output_writes_a_file(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    with open_output(str(path)) as handle:
        handle.write("hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_open_output_reports_an_unwritable_path(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cannot write"), open_output(
        str(tmp_path / "missing" / "out.txt")
    ):
        pass


@pytest.mark.parametrize("path", [None, "-"])
def test_open_output_falls_back_to_stdout(
    path: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    with open_output(path) as handle:
        handle.write("hello")
    assert capsys.readouterr().out == "hello"

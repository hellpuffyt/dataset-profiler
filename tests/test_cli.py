"""The command line, end to end."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from dataset_profiler import __version__
from dataset_profiler.cli import (
    DRIFT_EXIT,
    build_parser,
    main,
    parse_columns,
    parse_null_tokens,
)
from dataset_profiler.errors import UsageError
from dataset_profiler.types import DEFAULT_NULL_TOKENS

SALES = (
    "order_id,region,amount\n"
    "1,north,10\n"
    "2,north,20\n"
    "3,south,30\n"
    "4,south,\n"
)
SHIFTED = (
    "order_id,region,amount\n"
    "5,west,910\n"
    "6,west,920\n"
    "7,west,930\n"
    "8,west,940\n"
)


@pytest.fixture()
def sales(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text(SALES, encoding="utf-8")
    return path


@pytest.fixture()
def baseline(tmp_path: Path, sales: Path) -> Path:
    path = tmp_path / "baseline.json"
    assert main(["profile", str(sales), "--format", "json", "-o", str(path)]) == 0
    return path


def test_profiling_a_file_writes_a_report(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales)]) == 0
    out = capsys.readouterr().out
    assert "rows: 4, columns: 3" in out
    assert "order_id" in out


def test_the_tsv_format_is_machine_readable(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "--format", "tsv"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[1].split("\t")[:3] == ["order_id", "integer", "0"]


def test_the_json_format_round_trips_through_the_baseline_reader(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "-f", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["format"] == "dataset-profile/1"
    assert data["row_count"] == 4


def test_output_can_go_to_a_file(sales: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.txt"
    assert main(["profile", str(sales), "-o", str(out)]) == 0
    assert "rows: 4" in out.read_text(encoding="utf-8")


def test_profiling_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(SALES))
    assert main(["profile", "-"]) == 0
    assert "dataset: (stdin)" in capsys.readouterr().out


def test_a_subset_of_columns_can_be_profiled(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "-f", "tsv", "--columns", "region"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("region\t")


def test_asking_for_a_column_that_is_not_there_fails(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "--columns", "nope"]) == 1
    assert "no such column 'nope'" in capsys.readouterr().err


def test_a_columns_flag_that_selects_nothing_fails(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "--columns", " , "]) == 1
    assert "selected no columns" in capsys.readouterr().err


def test_null_tokens_can_be_replaced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "d.csv"
    path.write_text("v\nmissing\nNA\n", encoding="utf-8")
    assert main(["profile", str(path), "-f", "tsv", "--null-tokens", "missing"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[1].split("\t")[2] == "1"


def test_a_missing_input_file_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(tmp_path / "nope.csv")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_an_empty_input_file_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert main(["profile", str(path)]) == 1
    assert "no header row" in capsys.readouterr().err


def test_a_bad_option_value_is_reported(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "--bins", "0"]) == 1
    assert "--bins must be positive" in capsys.readouterr().err


def test_an_explicit_delimiter_is_honoured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "semi.csv"
    path.write_text("a;b\n1;2\n", encoding="utf-8")
    assert main(["profile", str(path), "-f", "tsv", "-d", ";"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[1].startswith("a\t")


def test_a_bad_delimiter_is_reported(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(sales), "-d", "::"]) == 1
    assert "one character" in capsys.readouterr().err


def test_a_named_encoding_is_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "latin.csv"
    path.write_bytes("name\ncaf\xe9\n".encode("latin-1"))
    assert main(["profile", str(path), "--encoding", "latin-1"]) == 0
    assert "rows: 1" in capsys.readouterr().out


def test_comparing_a_file_against_its_own_baseline_finds_nothing(
    sales: Path, baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", str(sales), "--baseline", str(baseline)]) == 0
    assert "no drift detected" in capsys.readouterr().out


def test_comparing_a_changed_file_reports_drift(
    tmp_path: Path, baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "shifted.csv"
    path.write_text(SHIFTED, encoding="utf-8")
    assert main(["compare", str(path), "-b", str(baseline), "-f", "tsv"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    statuses = {line.split("\t")[0]: line.split("\t")[1] for line in lines[1:]}
    assert statuses["amount"] == "drifted"
    assert statuses["region"] == "drifted"


def test_fail_on_drift_changes_the_exit_status(
    tmp_path: Path, baseline: Path
) -> None:
    """This is the flag a scheduled job actually depends on."""
    path = tmp_path / "shifted.csv"
    path.write_text(SHIFTED, encoding="utf-8")
    argv = ["compare", str(path), "-b", str(baseline), "--fail-on-drift"]
    assert main(argv) == DRIFT_EXIT


def test_fail_on_drift_is_quiet_when_nothing_moved(
    sales: Path, baseline: Path
) -> None:
    argv = ["compare", str(sales), "-b", str(baseline), "--fail-on-drift"]
    assert main(argv) == 0


def test_the_psi_threshold_is_configurable(
    sales: Path, baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["compare", str(sales), "-b", str(baseline), "--psi-threshold", "0.5"]
    assert main(argv) == 0
    assert "psi >= 0.5" in capsys.readouterr().out


def test_a_negative_null_threshold_is_rejected(
    sales: Path, baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["compare", str(sales), "-b", str(baseline), "--null-threshold", "-1"]
    assert main(argv) == 1
    assert "--null-threshold" in capsys.readouterr().err


def test_a_missing_baseline_is_reported(
    sales: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["compare", str(sales), "-b", str(tmp_path / "nope.json")]
    assert main(argv) == 1
    assert "cannot read baseline" in capsys.readouterr().err


def test_a_baseline_that_is_not_json_is_reported(
    sales: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["compare", str(sales), "-b", str(path)]) == 1
    assert "invalid JSON" in capsys.readouterr().err


def test_a_baseline_that_is_some_other_json_is_reported(
    sales: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    assert main(["compare", str(sales), "-b", str(path)]) == 1
    assert "not a dataset profile" in capsys.readouterr().err


def test_the_drift_report_can_be_json(
    sales: Path, baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["compare", str(sales), "-b", str(baseline), "-f", "json"]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["has_drift"] is False


def test_compare_requires_a_baseline(sales: Path) -> None:
    with pytest.raises(SystemExit):
        main(["compare", str(sales)])


def test_an_unknown_format_is_rejected_by_the_parser(sales: Path) -> None:
    with pytest.raises(SystemExit):
        main(["profile", str(sales), "-f", "xml"])


def test_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "COMMAND" in capsys.readouterr().out


def test_the_version_flag_reports_the_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == __version__


def test_the_parser_exposes_both_subcommands() -> None:
    parser = build_parser()
    text = parser.format_help()
    assert "profile" in text
    assert "compare" in text


def test_parse_columns_splits_and_strips() -> None:
    assert parse_columns(" a , b ") == ["a", "b"]
    assert parse_columns(None) is None


def test_parse_columns_rejects_an_empty_selection() -> None:
    with pytest.raises(UsageError):
        parse_columns(",,")


def test_parse_null_tokens_defaults_and_replaces() -> None:
    assert parse_null_tokens(None) == DEFAULT_NULL_TOKENS
    assert parse_null_tokens("Missing,GONE") == frozenset({"", "missing", "gone"})


def test_the_empty_string_is_always_a_null_token() -> None:
    """A blank cell is missing in every dataset that has ever existed."""
    assert "" in parse_null_tokens("weird")


def test_the_seed_is_accepted_and_reproducible(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["profile", str(sales), "-f", "json", "--seed", "5", "--sample-size", "2"]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    assert capsys.readouterr().out == first


def test_the_committed_example_profiles_as_documented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same assertion the CI dogfood job makes, run locally too."""
    root = Path(__file__).resolve().parent.parent
    assert main(["profile", str(root / "examples" / "sales.csv"), "-f", "tsv"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    expected = (root / "examples" / "expected-profile.tsv").read_text(
        encoding="utf-8"
    )
    assert lines == expected.strip().splitlines()


def test_the_committed_example_drifts_as_documented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parent.parent
    argv = [
        "compare",
        str(root / "examples" / "sales-q2.csv"),
        "-b",
        str(root / "examples" / "baseline.json"),
        "-f",
        "tsv",
        "--fail-on-drift",
    ]
    assert main(argv) == DRIFT_EXIT
    lines = capsys.readouterr().out.strip().splitlines()
    expected = (root / "examples" / "expected-drift.tsv").read_text(encoding="utf-8")
    assert lines == expected.strip().splitlines()

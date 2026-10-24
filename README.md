# dataset-profiler

Profile a tabular dataset — inferred types, cardinality, null patterns and
numeric shape — then store that profile and use it as a baseline to detect
drift the next time the data arrives.

Zero runtime dependencies. Standard library only.

This project is not published on PyPI. **`pip install dataset-profiler` installs an
unrelated third-party package with the same name** — install from source instead
(Python 3.10+, no runtime dependencies):

```bash
git clone https://github.com/hellpuffyt/dataset-profiler
cd dataset-profiler
python -m venv .venv && . .venv/bin/activate
pip install .
```

```bash
# What is in this file?
dataset-profiler profile sales.csv

# Keep this month as the baseline.
dataset-profiler profile sales.csv --format json -o baseline.json

# Next month: has anything moved?
dataset-profiler compare sales-q2.csv --baseline baseline.json --fail-on-drift
```

## Why this exists

Data pipelines fail loudly when a file is malformed and quietly when it is
merely *different*. A region column that gains a new value, a price column
that shifts by an order of magnitude, an id column that stops being unique,
a field that goes from 0% null to 40% null — none of those raise anything.
They just make every number downstream wrong.

The usual answer is a notebook that somebody ran once. This is the same idea
made into a file you can commit and a command a scheduler can run:

```
$ dataset-profiler profile examples/sales.csv
dataset: examples/sales.csv
rows: 12, columns: 6

column     type     nulls        distinct  detail
---------  -------  -----------  --------  --------------------------------------------
order_id   integer  0 (0.0%)     12        unique; range 1001..1012; mean 1006.5
region     string   0 (0.0%)     3         top 'north' x5; len 4..5
amount     float    1 (8.3%)     11        range 12.5..640; mean 191.681
signed_up  date     0 (0.0%)     11        top '2024-01-04' x2; len 10..10
active     boolean  2 (16.7%)    2         top 'true' x7; len 4..5; null spelt 'N/A' x2
notes      empty    12 (100.0%)  0         never populated
```

```
$ dataset-profiler compare examples/sales-q2.csv --baseline examples/baseline.json
baseline: examples/sales.csv (12 rows)
current:  examples/sales-q2.csv (12 rows)
thresholds: psi >= 0.2, null rate delta > 5.0%

column     status        severity  psi     notes
---------  ------------  --------  ------  --------------------------------------------------------
order_id   drifted       major     11.051  distribution PSI 11.051 (major)
region     ok            none      0.000
amount     drifted       major     12.384  distribution PSI 12.384 (major)
signed_up  ok            unknown   -       values are near-unique; frequency comparison skipped
active     ok            none      0.000
notes      type_changed  unknown   -       type changed: empty -> string; null rate 100.0% -> 83.3%

3 of 6 column(s) changed
```

Both of those are the committed `examples/` fixtures, and
[the CI dogfood job](.github/workflows/ci.yml) asserts the real console
script still produces exactly that output.

## What it reports

**Types.** Every CSV field is text, so a type is a claim about what the text
consistently looks like: `empty`, `boolean`, `integer`, `float`, `date`,
`datetime`, `string`. Per-value guesses are folded together with a widening
lattice — `integer` and `float` make `float`, `date` and `datetime` make
`datetime`, anything else that disagrees makes `string`. Unification is
commutative and associative (both asserted in the tests), so the inferred
type never depends on row order.

An all-null column comes out as `empty`, not `string`. "We never saw a
value" is a different fact from "we saw text", and only the first one should
stop you trusting the column.

**Cardinality.** Exact distinct count, plus derived flags: a column where
every value differs and none is missing is a candidate key; one distinct
value is a constant nobody meant to ship.

**Null patterns.** Not just how many, but *how they were spelt*. A file with
three blanks, two `N/A` and one `-` in the same column came from three
different producers, and that is worth seeing:

```
active  boolean  2 (16.7%)  2  top 'true' x7; null spelt 'N/A' x2
```

The null vocabulary is `"" NA N/A NaN null none nil - ?`, case-insensitive
and whitespace-trimmed, and `--null-tokens` replaces it entirely.

**Numeric shape.** Count, min, max, mean, standard deviation, the p01/p25/
p50/p75/p99 quantiles, and a 10-bin histogram. The mean and standard
deviation use Welford, not `sum(x*x) - n*mean**2`; a column of Unix
timestamps is enough to make the textbook formula return a negative
variance, and a profiler that reports a negative variance is worse than one
that reports nothing.

## How drift is measured

Two profiles of the same dataset a month apart do not share bin edges, do
not share a category list, and may not even share columns. So:

- **Numeric columns** are compared by re-expressing the current histogram on
  the *baseline* bin edges and taking the
  [PSI](https://en.wikipedia.org/wiki/Population_stability_index). The
  baseline defines the grid — that is what a baseline is for. Mass that has
  moved outside the old range is clamped into the outer bins rather than
  dropped, so a shift off the end shows up as a large PSI instead of as
  missing data.
- **Everything else** is compared as a frequency table over the union of
  both value sets.
- **Columns that changed type** get no PSI at all. A distance between text
  and numbers would be a fabricated number, which is the worst possible
  thing for this tool to produce.
- **Near-unique text columns** get no PSI either. Comparing the counts of
  values that each occur once measures nothing except which particular ids
  landed in each extract.
- **Nulls are tracked separately** from the value distribution throughout. A
  column that goes from 0% to 40% null has a serious problem that a PSI over
  its non-null values will not show.

PSI is read the conventional way: under 0.1 stable, 0.1 to 0.25 moderate,
above 0.25 a material shift. A column counts as drifted at or above
`--psi-threshold` (default 0.2), or when its null rate moves by more than
`--null-threshold` (default 0.05), or when it changes type, or when it stops
being unique.

## Using it in a scheduled job

`compare --fail-on-drift` exits **2** when anything moved, **1** when the
input or the baseline was unusable, and **0** when the data is fine — so the
three cases are distinguishable in a shell:

```bash
dataset-profiler compare today.csv --baseline baseline.json \
  --format json --fail-on-drift -o drift.json
case $? in
  0) echo "clean" ;;
  2) echo "drift"; cat drift.json ;;
  *) echo "could not check"; exit 1 ;;
esac
```

## What it does not keep in memory

The reader never loads the file. It reads 8 KB to sniff the dialect,
finishes the line that sample stopped inside, and hands the CSV reader that
sample followed by the rest of the stream — so reading the header consumes
at most 8 KB plus one line no matter how large the file is, which
[a test asserts](tests/test_io.py) by counting characters taken from the
stream.

What memory the profiler does use is set by three options, not by the number
of rows: `--max-distinct` (exact cardinality is counted up to this many
values), `--max-categories` (the frequency table kept for drift), and
`--sample-size` (a bounded, *seeded* reservoir for quantiles and the
histogram — the sample is random with respect to the data but identical for
identical input, so two runs over the same file always agree).

Measured on an Apple M5 (10 cores, 16 GB, macOS 27), CPython 3.12, peak Python
heap via `tracemalloc` around `main(["profile", FILE, "--format", "json"])`
(tracing slows the run), five columns of a generated CSV:

| Rows | File | Time | Peak heap (defaults) | Peak heap, `--max-distinct 1000` |
| --- | --- | --- | --- | --- |
| 1,000 | 0.03 MB | 0.08 s | 0.79 MB | 0.79 MB |
| 1,000,000 | 35.6 MB | 38.8 s | 20.6 MB | 2.21 MB |

The first version of the reader pushed the sniffed sample back by reading
the whole file into a string; the same 1,000,000-row measurement peaked at
**180.7 MB**. That is the difference the streaming reader makes, and it is
why the test above counts characters rather than trusting the code to look
right.

Every limit that was actually hit is recorded *in the profile* —
`distinct_truncated`, `categories_truncated`, `sampled` — so a consumer can
tell "this is exact" from "this is the best a bounded profiler could do".
The drift report says so too, in the notes.

## Options

Shared by both subcommands:

| Flag | Effect |
| --- | --- |
| `-f, --format {text,tsv,json}` | human table, one line per column, or full JSON |
| `-o, --output PATH` | write here instead of stdout |
| `-d, --delimiter CHAR` | override the sniffed delimiter |
| `--encoding NAME` | text encoding (default utf-8) |
| `--columns A,B` | profile only these columns |
| `--null-tokens A,B` | replace the null vocabulary |
| `--max-distinct N` | exact cardinality limit (default 100000) |
| `--max-categories N` | frequency table size (default 50) |
| `--sample-size N` | numeric reservoir size (default 20000) |
| `--seed N` | reservoir seed (default 0) |
| `--bins N` | histogram bins (default 10) |

`compare` adds `-b/--baseline`, `--psi-threshold`, `--null-threshold` and
`--fail-on-drift`.

Input that cannot be read — wrong `--encoding`, an unknown encoding name, or
a single field over Python's 128 KiB `csv` limit — exits **1** with a message.

The input may be `-` for stdin. Dialects (`,` `;` tab `|`) are sniffed, a
spreadsheet BOM is stripped, duplicate and blank header names are made
unique (`a`, `a_2`, `column_3`), and ragged rows are padded or truncated and
counted rather than raised on — a profile that says "3 rows did not match
the header width" is useful, and a traceback is not.

## As a library

```python
from dataset_profiler.drift import compare
from dataset_profiler.io import read_table
from dataset_profiler.profile import DatasetProfile, profile_table

with open("sales.csv", newline="") as handle:
    current = profile_table(read_table(handle), source="sales.csv")

baseline = DatasetProfile.from_json(open("baseline.json").read())
report = compare(baseline, current, psi_threshold=0.15)

for column in report.changes:
    print(column.name, column.status, column.severity, column.notes)
```

A `DatasetProfile` is a value: it compares by equality and survives a round
trip through JSON unchanged. That is not a nicety — a baseline that does not
round-trip is a printout, not a baseline, and there is a test that says so.

## Development

```bash
pip install -e ".[dev]"                                   # inside the venv above
pytest -q                                                 # 304 tests
ruff check .                                              # lint
mypy                                                      # strict types
pytest -q --cov=dataset_profiler --cov-report=term-missing
```

Coverage is 99% of `src/dataset_profiler`; the only uncovered lines are the
four in the `python -m dataset_profiler` shim.

## Security

See [SECURITY.md](SECURITY.md); audit evidence is in
[docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

## License

MIT — see [LICENSE](LICENSE).

# Contributing

Thanks for taking a look.

## Getting set up

```bash
git clone https://github.com/hellpuffyt/dataset-profiler
cd dataset-profiler
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

All four must pass; CI runs the same commands.

```bash
pytest -q
ruff check .
mypy
pytest -q --cov=dataset_profiler --cov-report=term-missing
```

## Design constraints

These are the rules the whole project is built around. A change that breaks
one of them is not a trade-off to discuss — it is a different tool.

- **A profile never invents a number.** If a statistic cannot be computed
  honestly, the field is `None` and the report says why. This is the reason
  a column that changed type gets no PSI, a near-unique text column gets no
  PSI, and an empty histogram gets no PSI. Filling those in with something
  plausible would make every downstream alert untrustworthy.
- **Every limit that was hit is recorded in the profile.**
  `distinct_truncated`, `categories_truncated` and `sampled` exist so a
  consumer can tell an exact answer from a bounded one. If you add a new
  bounded resource, add the flag that admits to it in the same change.
- **Memory is a function of the options, not of the row count.** Nothing may
  accumulate per row. The reader must not read the file to push its sniff
  sample back; `tests/test_io.py` counts the characters taken from the
  stream, and `tests/test_profile.py` asserts a peak-heap ceiling under
  `tracemalloc`.
- **A profile must survive a JSON round trip unchanged.** A baseline that
  does not round-trip is a printout. Floats are rounded once, at
  construction, so the in-memory object equals the reloaded one.
- **No runtime dependencies.** The tool has to be installable anywhere
  without argument. Development dependencies are fine.

## Adding a type

1. Add the kind to `src/dataset_profiler/types.py`, give it a parser, and
   put the check in `classify` **in the right order** — the ordering is the
   specification. `date` is checked before `datetime` because the datetime
   parser also accepts a bare date.
2. Extend `unify` if it should widen into something. Then run the existing
   commutativity and associativity tests, which cover every pair and triple
   of kinds automatically and will fail if the lattice stopped being one.
3. Add it to the type list in `README.md` and a changelog entry.

## Adding a drift signal

1. Compute it in `_compare_column` in `src/dataset_profiler/drift.py` and
   append a note describing it in words a person can act on.
2. Decide what it does to `status`. A signal that only ever adds a note is
   documentation; a signal that can set `DRIFTED` changes the exit status of
   somebody's scheduled job, so it needs a threshold and a flag.
3. Add a test that the signal fires **and** a test that it does not fire on
   an unchanged dataset. The second one is the one that matters: a drift
   detector that cries wolf gets turned off, and then it detects nothing.

## Changing the output format

`tsv` is the machine format and the CI dogfood job diffs it byte for byte
against `examples/expected-profile.tsv` and `examples/expected-drift.tsv`.
If you change it deliberately, regenerate both:

```bash
dataset-profiler profile examples/sales.csv --format tsv \
  -o examples/expected-profile.tsv
dataset-profiler compare examples/sales-q2.csv --baseline examples/baseline.json \
  --format tsv -o examples/expected-drift.tsv
```

If you change the profile JSON, regenerate `examples/baseline.json` the same
way and bump the `FORMAT` string in `profile.py`, so an old baseline fails
with "not a dataset profile" instead of comparing against a shape that no
longer means what it did.

## Reporting a bug

A small CSV that reproduces it is worth more than a description — ten rows
is usually plenty, and the profile of it (`--format json`) is worth more
still, since it says exactly what the tool believed about the data. If the
data is not yours to share, a file with the same shape reproduces almost
every real bug in this tool: the same delimiter, the same null spellings,
the same ragged rows.

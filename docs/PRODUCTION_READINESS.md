# Production readiness — dataset-profiler

Audit date: 2026-10-22. Tested commit: `bbd20bd` (branch `audit/production-readiness`, based on `main` @ `f854c07`).

## Scope

Stdlib-only CLI + library: profile a CSV (types via a widening lattice, exact-or-bounded cardinality,
null spellings, Welford moments, reservoir quantiles, histogram) to text/TSV/JSON, then `compare` a
new file against a stored JSON baseline (PSI on baseline bin edges for numerics, frequency PSI for
categoricals, separate null-rate tracking, type changes). Exit codes 0 clean / 1 unusable / 2 drift.

## Platforms and prerequisites

CPython 3.10–3.13, no runtime dependencies. Verified on macOS 27 (Apple M5, arm64), CPython 3.12.14.
Other OSes rely on the existing CI matrix (not re-run here).

## Clean-clone verification (commit `bbd20bd`)

```bash
git clone -b audit/production-readiness <local clone> dataset-profiler && cd dataset-profiler
uv venv --python 3.12 .venv && . .venv/bin/activate     # README: python -m venv (equivalent)
uv pip install .                                        # exit 0
# CI dogfood steps, run literally:
dataset-profiler profile examples/sales.csv --format tsv -o profile.tsv && diff examples/expected-profile.tsv profile.tsv   # identical
dataset-profiler compare examples/sales-q2.csv --baseline examples/baseline.json --format tsv --fail-on-drift -o drift.tsv  # exit 2, diff identical
dataset-profiler profile examples/sales.csv --format json -o fresh.json && diff examples/baseline.json fresh.json           # identical
dataset-profiler compare examples/sales.csv --baseline fresh.json --fail-on-drift   # "no drift detected"
uv pip install -e ".[dev]" && pytest -q && ruff check . && mypy                     # 304 passed / clean / clean
```

## Commands and results

| Command | Baseline (`f854c07`) | After (`bbd20bd`) |
| --- | --- | --- |
| `pytest -q` | exit 0 — 299 passed, 0 failed, 0 skipped | exit 0 — 304 passed, 0 failed, 0 skipped |
| `ruff check .` | exit 0 | exit 0 |
| `mypy` (strict) | exit 0 | exit 0 |
| `uvx pip-audit` | — | no known vulnerabilities |
| README `profile` / `compare` examples | output identical to README | identical |

CI status — default branch: success (`f854c07`, 2026-08-31); audit branch: pending until pushed.

## Math verification (independent of the project's own tests)

Against NumPy in a throwaway environment (NumPy is not a dependency), 300 random columns:
- Welford standard deviation on timestamp-scale values (mean 1.7e9, spread 0.1–1000): max relative
  error vs `numpy.std` 2.0e-10.
- Quantiles p01/p25/p50/p75/p99 vs `numpy.quantile` (linear): max relative error 1.4e-16.
- PSI vs `sum((a-e)·ln(a/e))` on normalised NumPy arrays: max absolute error 4.4e-16.
- `rebin` preserves total mass (1000 → 1000.0) and is the identity on identical edges.
- New test `test_psi_matches_a_hand_computed_value`: 50/50 → 25/75 equals 0.25·ln 3.

## Memory (bounded-memory claim)

Re-measured with `tracemalloc` (see README table): 1,000,000 rows / 35.6 MB → peak heap 20.6 MB with
defaults (dominated by the 100,000-value exact-distinct set) and 2.21 MB with `--max-distinct 1000`.
The original README figures were from a different machine (Windows, Python 3.13) and were replaced.

## Security review

- **Package-name confusion (Medium):** README told users to `pip install dataset-profiler`; that name
  on PyPI is an unrelated project (version 2.1.0, different author). Following the README installed
  third-party code. README now installs from source and warns explicitly.
- No network I/O; baseline JSON parsed with `json` and validated field by field (malformed baselines
  already produced clean errors). No `eval`/pickle.
- Secret scan of full history: no hits. `pip-audit`: no known vulnerabilities.
- Workflow permissions `contents: read`; actions tag-pinned (left as is).

## Bugs found and fixed

| Severity | Bug | Regression test |
| --- | --- | --- |
| Medium | README install instruction installed an unrelated PyPI package | docs (no code) |
| Low | `UnicodeDecodeError` (input or baseline), `LookupError` (bad `--encoding`), `csv.Error` (field > 128 KiB) were tracebacks, breaking the documented exit-1 contract | `tests/test_cli.py::test_undecodable_input_is_a_message_not_a_traceback`, `test_undecodable_baseline_is_a_message`, `test_unknown_encoding_is_a_message`, `test_oversized_field_is_a_message` |

## Known limitations

- Values Python cannot parse as finite numbers (`inf`, `1e400`) make a column `string`; numeric
  detail is then shown for the subset that did parse.
- Per-field size is capped at 128 KiB by the stdlib `csv` module.
- Only CSV-like input (no Excel/Parquet).

## Acceptance checklist

| # | Item | Expected | Command / test | Result |
| --- | --- | --- | --- | --- |
| 1 | README profile output reproduces | identical | dogfood diff | PASS |
| 2 | README drift output + exit 2 | identical, exit 2 | dogfood diff | PASS |
| 3 | Baseline round-trips | fresh profile == committed baseline | dogfood diff | PASS |
| 4 | Welford/quantile/PSI correct | match NumPy / hand value | section above | PASS |
| 5 | Bounded memory | heap set by options, not rows | tracemalloc run | PASS (2.21 MB at 1M rows) |
| 6 | Unusable input → exit 1 message | no traceback | 4 new CLI tests | PASS |
| 7 | Clean clone | install, tests, examples | section above | PASS |

## Classification proposal

**Production-Ready for Its Documented Scope** (proposal; lead to verify independently).

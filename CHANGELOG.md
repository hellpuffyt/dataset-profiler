# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Undecodable input or baseline files, unknown `--encoding` names and fields
  over the `csv` module's 128 KiB limit crashed with a traceback; they now exit
  1 with a message, as the exit-code contract promises.

### Changed

- README: install from source. The name `dataset-profiler` on PyPI belongs to an
  unrelated project, so the previous `pip install dataset-profiler` instruction
  installed someone else's code. Memory table re-measured on the audit machine.

### Added

- Hand-computed PSI regression test; `SECURITY.md`, `.editorconfig`,
  `docs/PRODUCTION_READINESS.md`.

## [0.1.0] — 2026-08-31

First release.

### Added

- `dataset-profiler profile` — inferred types, exact cardinality, null
  counts *and null spellings*, string lengths, and a numeric summary with
  mean, standard deviation, p01/p25/p50/p75/p99 quantiles and a histogram.
- Type inference over a widening lattice (`empty`, `boolean`, `integer`,
  `float`, `date`, `datetime`, `string`) that is commutative and
  associative, so the inferred type does not depend on row order. An
  all-null column is reported as `empty` rather than `string`.
- `dataset-profiler compare` — drift against a stored baseline profile.
  Numeric columns are compared by re-binning the current histogram onto the
  baseline bin edges and taking the Population Stability Index; everything
  else by a frequency table over the union of both value sets. Null rate,
  type change and loss of uniqueness are tracked as separate signals.
- Deliberate refusals to guess: no PSI for a column that changed type, for a
  near-unique text column, or for an empty histogram. The report says why
  instead of inventing a distance.
- Three output formats — an aligned `text` report, a `tsv` line per column
  for the next program in the pipe, and `json`, which is the same text
  `--baseline` reads back.
- Exit status 2 from `compare --fail-on-drift`, distinct from status 1 for
  an unusable input or baseline, so a scheduled job can tell the difference.
- Bounded, honest resource use: `--max-distinct`, `--max-categories` and a
  seeded `--sample-size` reservoir, with `distinct_truncated`,
  `categories_truncated` and `sampled` recorded in the profile whenever a
  limit was actually reached.
- Real-file handling: dialect sniffing (`,` `;` tab `|`) with the sample
  pushed back without reading the file, BOM stripping, unique-ified header
  names, ragged rows padded or truncated and counted, and a blank line in a
  one-column file treated as a value rather than as formatting.
- Welford accumulation for mean and standard deviation, so a column of
  timestamp-scale values does not produce a negative variance.
- `--columns`, `--null-tokens`, `--delimiter`, `--encoding`, `--seed`,
  `--bins`, stdin/stdout support, and `python -m dataset_profiler`.

[0.1.0]: https://github.com/hellpuffyt/dataset-profiler/releases/tag/v0.1.0

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `src/mortality.py` module with domain-specific analytics:
  - `cumulative_mortality_rate` and `annual_mortality_rate` (exponential
    decay model, `m = 1 - (N_t / N_0) ** (1/t)`).
  - `compute_cohort_metrics` for per-row cumulative and annualized mortality.
  - `survival_curve` for pooled or grouped Kaplan-Meier-style curves,
    weighted by cohort size.
  - `mortality_hotspots` to flag sites or species above a configurable
    annualized mortality threshold.
  - `cause_of_death_breakdown` aggregator with per-cause deaths and shares.
  - `summarize` for a single-dict portfolio overview.
- `demo/sample_data.csv` - 20-row sample dataset spanning 5 reforestation
  species, 9 sites, and 1-3 year assessment windows.
- `examples/mortality_analysis.py` runnable demo against the sample dataset.
- `tests/test_mortality.py` with 38 pytest cases covering math correctness
  (against hand-computed values), edge cases (`N_0 <= 0`, `N_t > N_0`,
  negative/zero time, `NaN` inputs, empty frames), and demo-data integration.
- Package-level re-exports in `src/__init__.py` for the new analytics API.

### Changed
- `README.md` rewritten around the actual analytics API, with install,
  quickstart using `demo/sample_data.csv`, and a data-format table.
- `.gitignore` now also ignores `.pytest_cache/`, `.coverage`, and `htmlcov/`.

## [0.1.0] - 2025-04-15

### Added
- Initial `MortalityTracker` class with CSV/Excel loading, generic
  preprocessing, and summary-statistics analysis.
- `src/data_generator.py` synthetic data generator.
- `examples/basic_usage.py` minimal usage example.

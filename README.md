# Tree Mortality Tracker

A small Python toolkit for tracking tree mortality rates across reforestation
project sites. Given per-cohort observations (planted count, surviving count,
dates, cause of death), it computes cumulative and annualized mortality,
Kaplan-Meier-style survival curves, hotspot sites, and cause-of-death breakdowns.

The annualized rate uses a constant-hazard exponential decay model:

    m_annual = 1 - (N_t / N_0) ** (1 / t)

## Features

- Load mortality observations from CSV or Excel (`MortalityTracker.load_data`).
- Per-cohort cumulative and annualized mortality (`compute_cohort_metrics`).
- Pooled or grouped survival curves (`survival_curve`).
- Hotspot detection by site or species above a configurable threshold
  (`mortality_hotspots`).
- Cause-of-death aggregation with counts and shares
  (`cause_of_death_breakdown`).
- Overall portfolio summary (`summarize`).
- Bundled demo dataset at `demo/sample_data.csv` (20 rows, 5 species, 9 sites).

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ is required.

## Quick Start

Run the bundled demo analysis end-to-end:

```bash
python examples/mortality_analysis.py
```

Or programmatically:

```python
import pandas as pd
from src.mortality import (
    compute_cohort_metrics,
    mortality_hotspots,
    survival_curve,
    summarize,
)

df = pd.read_csv("demo/sample_data.csv")

print(summarize(df))
# {'total_planted': 8100.0, 'total_alive': 6178.0, ...}

metrics = compute_cohort_metrics(df)
print(metrics[["site_id", "species", "annual_mortality"]].head())

hotspots = mortality_hotspots(df, by="site_id", threshold=0.15)
print(hotspots)

curve = survival_curve(df, group_by="species")
print(curve.head())
```

## Data Format

Required columns in the input CSV/Excel:

| column            | type   | notes                                    |
|-------------------|--------|------------------------------------------|
| `site_id`         | string | plot or site identifier                  |
| `species`         | string | tree species                             |
| `planted_date`    | date   | ISO format (`YYYY-MM-DD`)                |
| `initial_count`   | int    | number of seedlings planted (`N_0 > 0`)  |
| `alive_count`     | int    | number alive at assessment (`0 <= Nt <= N0`) |
| `assessment_date` | date   | ISO format                               |
| `cause_of_death`  | string | free-form; optional but recommended      |

See `demo/sample_data.csv` for a working example.

## Edge Case Handling

- `N_0 <= 0`, `NaN` counts, or non-positive elapsed time produce `NaN` metrics
  for that row (rest of the batch is unaffected).
- `alive_count > initial_count` raises `ValueError` - it's physically impossible.
- Empty DataFrames and missing required columns raise `ValueError`.
- Blank/`NaN` cause-of-death values are coerced to `"unknown"`.

## Tests

```bash
pytest tests/ -v
```

38 tests cover the math (against hand-computed values), edge cases, and the
bundled demo dataset.

## Project Structure

```
tree-mortality-tracker/
├── src/
│   ├── main.py             # Generic CSV/Excel loader + summary stats
│   ├── mortality.py        # Mortality analytics (this project's core)
│   └── data_generator.py   # Synthetic data generator
├── demo/
│   └── sample_data.csv     # 20-row example dataset
├── examples/
│   ├── basic_usage.py
│   └── mortality_analysis.py
├── tests/
│   └── test_mortality.py
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

## License

MIT License - free to use, modify, and distribute.

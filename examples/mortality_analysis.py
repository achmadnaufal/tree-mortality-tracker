"""Run mortality analytics against the bundled demo dataset.

Usage:
    python examples/mortality_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mortality import (  # noqa: E402
    cause_of_death_breakdown,
    compute_cohort_metrics,
    mortality_hotspots,
    summarize,
    survival_curve,
)


def main() -> None:
    demo = ROOT / "demo" / "sample_data.csv"
    df = pd.read_csv(demo)
    print(f"Loaded {len(df)} cohort observations from {demo.name}\n")

    overall = summarize(df)
    print("=== Overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v:,.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Per-cohort metrics (first 5) ===")
    metrics = compute_cohort_metrics(df)
    cols = [
        "site_id",
        "species",
        "years_elapsed",
        "cumulative_mortality",
        "annual_mortality",
    ]
    print(metrics[cols].head().to_string(index=False))

    print("\n=== Hotspots by site (threshold=0.15 annual) ===")
    hot = mortality_hotspots(df, by="site_id", threshold=0.15)
    print(hot.to_string(index=False))

    print("\n=== Survival curve (pooled) ===")
    print(survival_curve(df).to_string(index=False))

    print("\n=== Cause-of-death breakdown ===")
    print(cause_of_death_breakdown(df).to_string(index=False))


if __name__ == "__main__":
    main()

"""Package: tree-mortality-tracker."""

from .main import MortalityTracker
from .mortality import (
    annual_mortality_rate,
    cause_of_death_breakdown,
    compute_cohort_metrics,
    cumulative_mortality_rate,
    mortality_hotspots,
    summarize,
    survival_curve,
)

__all__ = [
    "MortalityTracker",
    "annual_mortality_rate",
    "cause_of_death_breakdown",
    "compute_cohort_metrics",
    "cumulative_mortality_rate",
    "mortality_hotspots",
    "summarize",
    "survival_curve",
]


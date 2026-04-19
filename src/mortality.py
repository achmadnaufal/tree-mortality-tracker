"""Mortality analytics for reforestation plots.

This module implements domain-specific metrics that complement the generic
summary statistics produced by :class:`src.main.MortalityTracker`.

Conventions
-----------
* ``N_0`` is the initial planting count (integer, strictly positive).
* ``N_t`` is the count alive after ``t`` years (integer, ``0 <= N_t <= N_0``).
* Cumulative mortality rate:   ``M = 1 - N_t / N_0``
* Annualized mortality rate (exponential decay):
      ``m_annual = 1 - (N_t / N_0) ** (1 / t)``
  which assumes a constant per-year hazard over the observation period.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "site_id",
    "species",
    "planted_date",
    "initial_count",
    "alive_count",
    "assessment_date",
)


def cumulative_mortality_rate(initial_count: int, alive_count: int) -> float:
    """Compute the cumulative mortality rate for a cohort.

    Args:
        initial_count: Number of seedlings planted (``N_0``), must be > 0.
        alive_count: Number of trees still alive at assessment (``N_t``).

    Returns:
        Fraction of the cohort that has died, in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``initial_count <= 0`` or ``alive_count`` is
            negative, or if ``alive_count > initial_count``.
    """
    if initial_count is None or alive_count is None:
        raise ValueError("initial_count and alive_count must not be None")
    if initial_count <= 0:
        raise ValueError("initial_count must be > 0 (no zero-planting cohorts)")
    if alive_count < 0:
        raise ValueError("alive_count must be >= 0")
    if alive_count > initial_count:
        raise ValueError(
            f"alive_count ({alive_count}) cannot exceed initial_count "
            f"({initial_count})"
        )
    return 1.0 - (alive_count / initial_count)


def annual_mortality_rate(
    initial_count: int, alive_count: int, years: float
) -> float:
    """Compute the annualized mortality rate using an exponential decay model.

    The model assumes a constant per-year hazard, so that the fraction
    surviving after ``t`` years is ``(1 - m_annual) ** t``.

    Args:
        initial_count: Number of seedlings planted (``N_0``), must be > 0.
        alive_count: Number of trees still alive at assessment (``N_t``).
        years: Observation length in years. Must be strictly positive.

    Returns:
        Annualized mortality rate in ``[0.0, 1.0]``.

    Raises:
        ValueError: For invalid counts or non-positive ``years``.
    """
    if years is None or not np.isfinite(years):
        raise ValueError("years must be a finite number")
    if years <= 0:
        raise ValueError("years must be > 0")
    survival = 1.0 - cumulative_mortality_rate(initial_count, alive_count)
    return 1.0 - survival ** (1.0 / years)


def _years_between(planted: pd.Series, assessed: pd.Series) -> pd.Series:
    """Return elapsed time in fractional years between two date series."""
    planted = pd.to_datetime(planted, errors="coerce")
    assessed = pd.to_datetime(assessed, errors="coerce")
    delta_days = (assessed - planted).dt.days
    return delta_days / 365.25


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_cohort_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cohort cumulative and annualized mortality.

    A *cohort* here is one row of the input frame: a (site, species,
    assessment) observation.

    Args:
        df: Frame with columns ``site_id``, ``species``, ``planted_date``,
            ``initial_count``, ``alive_count``, ``assessment_date``.

    Returns:
        New DataFrame (the input is not mutated) with added columns:
        ``years_elapsed``, ``cumulative_mortality``, ``annual_mortality``.
        Rows with invalid inputs (``NaN`` counts, ``N_0 <= 0``, or
        ``years <= 0``) carry ``NaN`` in the derived columns instead of
        raising, so a single bad row does not break a batch report. An
        impossible row (``alive_count > initial_count``) still raises.

    Raises:
        ValueError: If the frame is empty, required columns are missing,
            or any row has ``alive_count > initial_count``.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    _validate_columns(df)

    out = df.copy()
    out["years_elapsed"] = _years_between(
        out["planted_date"], out["assessment_date"]
    )

    cumulative: List[float] = []
    annual: List[float] = []
    for _, row in out.iterrows():
        n0 = row["initial_count"]
        nt = row["alive_count"]
        years = row["years_elapsed"]

        if pd.isna(n0) or pd.isna(nt) or pd.isna(years):
            cumulative.append(np.nan)
            annual.append(np.nan)
            continue
        if n0 <= 0 or years <= 0:
            cumulative.append(np.nan)
            annual.append(np.nan)
            continue
        if nt > n0:
            raise ValueError(
                f"Row with site_id={row.get('site_id')!r} has alive_count "
                f"({nt}) greater than initial_count ({n0})"
            )

        cm = cumulative_mortality_rate(int(n0), int(nt))
        cumulative.append(cm)
        annual.append(1.0 - (1.0 - cm) ** (1.0 / float(years)))

    out["cumulative_mortality"] = cumulative
    out["annual_mortality"] = annual
    return out


def survival_curve(
    df: pd.DataFrame, group_by: Optional[str] = None
) -> pd.DataFrame:
    """Build a Kaplan-Meier-style survival curve from cohort observations.

    For each time point ``t`` (measured in years since planting), the
    survival probability is the count-weighted average of ``N_t / N_0``
    across all cohorts observed at that time.

    Args:
        df: Cohort frame (see :func:`compute_cohort_metrics`).
        group_by: Optional column name to produce one curve per group
            (e.g. ``"species"`` or ``"site_id"``). When ``None``, a single
            pooled curve is returned.

    Returns:
        DataFrame with columns ``years_elapsed``, ``survival``, and -
        when ``group_by`` is set - the grouping column. Rows are sorted
        by group then by time, with ``survival = 1.0`` anchored at
        ``t = 0`` for each group.

    Raises:
        ValueError: If the frame is empty or required columns are missing.
    """
    metrics = compute_cohort_metrics(df)
    metrics = metrics.dropna(
        subset=["years_elapsed", "cumulative_mortality", "initial_count"]
    )

    if group_by is not None and group_by not in metrics.columns:
        raise ValueError(f"group_by column {group_by!r} not found in frame")

    def _curve_for(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["survival"] = 1.0 - sub["cumulative_mortality"]
        sub["_t"] = sub["years_elapsed"].round(3)
        sub["_weighted"] = sub["survival"] * sub["initial_count"]
        # Weighted mean of survival by initial cohort size at each time.
        agg = (
            sub.groupby("_t", as_index=False)
            .agg(weighted=("_weighted", "sum"), total=("initial_count", "sum"))
            .rename(columns={"_t": "years_elapsed"})
        )
        agg["survival"] = agg["weighted"] / agg["total"]
        agg = agg[["years_elapsed", "survival"]].sort_values("years_elapsed")
        anchor = pd.DataFrame({"years_elapsed": [0.0], "survival": [1.0]})
        return pd.concat([anchor, agg], ignore_index=True)

    if group_by is None:
        return _curve_for(metrics)

    parts = []
    for key, sub in metrics.groupby(group_by):
        curve = _curve_for(sub)
        curve[group_by] = key
        parts.append(curve)
    if not parts:
        return pd.DataFrame(
            columns=["years_elapsed", "survival", group_by]
        )
    return pd.concat(parts, ignore_index=True)[
        [group_by, "years_elapsed", "survival"]
    ]


def mortality_hotspots(
    df: pd.DataFrame,
    by: str = "site_id",
    threshold: float = 0.30,
) -> pd.DataFrame:
    """Identify groups whose annualized mortality exceeds a threshold.

    Args:
        df: Cohort frame.
        by: Column to aggregate over (e.g. ``"site_id"`` or ``"species"``).
        threshold: Annualized mortality rate above which a group is
            flagged as a hotspot. Must be in ``[0.0, 1.0]``.

    Returns:
        DataFrame sorted by annualized mortality (descending) containing
        ``by``, ``planted``, ``alive``, ``cumulative_mortality``,
        ``annual_mortality``, and ``is_hotspot`` columns.

    Raises:
        ValueError: If inputs are invalid or ``by`` column is missing.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.0, 1.0]")

    metrics = compute_cohort_metrics(df)
    if by not in metrics.columns:
        raise ValueError(f"group column {by!r} not found in frame")

    metrics = metrics.dropna(subset=["years_elapsed", "initial_count", "alive_count"])
    grouped = (
        metrics.groupby(by)
        .agg(
            planted=("initial_count", "sum"),
            alive=("alive_count", "sum"),
            years=("years_elapsed", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["planted"] > 0].copy()
    grouped["cumulative_mortality"] = 1.0 - grouped["alive"] / grouped["planted"]
    grouped["annual_mortality"] = 1.0 - (
        1.0 - grouped["cumulative_mortality"]
    ) ** (1.0 / grouped["years"].where(grouped["years"] > 0, np.nan))
    grouped["is_hotspot"] = grouped["annual_mortality"] > threshold
    return grouped.sort_values("annual_mortality", ascending=False).reset_index(
        drop=True
    )


def cause_of_death_breakdown(
    df: pd.DataFrame, column: str = "cause_of_death"
) -> pd.DataFrame:
    """Aggregate recorded causes of death across cohorts.

    Args:
        df: Cohort frame containing a cause-of-death column.
        column: Name of the cause column. Missing/blank values are
            treated as ``"unknown"``.

    Returns:
        DataFrame with ``cause``, ``deaths``, and ``share`` columns,
        sorted by deaths descending. ``deaths`` counts the implied number
        of dead trees ``(initial_count - alive_count)`` per row, grouped
        by cause.

    Raises:
        ValueError: If the frame is empty or required columns missing.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    _validate_columns(df)
    if column not in df.columns:
        raise ValueError(f"cause column {column!r} not found in frame")

    work = df.copy()
    work[column] = work[column].fillna("unknown").replace("", "unknown")
    work["_dead"] = (work["initial_count"] - work["alive_count"]).clip(lower=0)
    grouped = (
        work.groupby(column)["_dead"].sum().rename("deaths").reset_index()
    )
    grouped = grouped.rename(columns={column: "cause"})
    total = grouped["deaths"].sum()
    grouped["share"] = grouped["deaths"] / total if total > 0 else 0.0
    return grouped.sort_values("deaths", ascending=False).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """Return a compact overall mortality summary.

    Args:
        df: Cohort frame.

    Returns:
        Dictionary with ``total_planted``, ``total_alive``,
        ``cumulative_mortality``, and ``avg_annual_mortality`` keys.
    """
    metrics = compute_cohort_metrics(df).dropna(
        subset=["years_elapsed", "initial_count", "alive_count"]
    )
    planted = float(metrics["initial_count"].sum())
    alive = float(metrics["alive_count"].sum())
    if planted <= 0:
        return {
            "total_planted": 0.0,
            "total_alive": 0.0,
            "cumulative_mortality": float("nan"),
            "avg_annual_mortality": float("nan"),
        }
    cm = 1.0 - alive / planted
    mean_years = float(metrics["years_elapsed"].mean())
    annual = (
        1.0 - (1.0 - cm) ** (1.0 / mean_years) if mean_years > 0 else float("nan")
    )
    return {
        "total_planted": planted,
        "total_alive": alive,
        "cumulative_mortality": cm,
        "avg_annual_mortality": annual,
    }

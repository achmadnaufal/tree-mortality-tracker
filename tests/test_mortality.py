"""Tests for src.mortality - annualized rates, survival, hotspots, causes."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mortality import (
    annual_mortality_rate,
    cause_of_death_breakdown,
    compute_cohort_metrics,
    cumulative_mortality_rate,
    mortality_hotspots,
    summarize,
    survival_curve,
)

DEMO_CSV = Path(__file__).resolve().parent.parent / "demo" / "sample_data.csv"


# ---------------------------------------------------------------------------
# cumulative_mortality_rate
# ---------------------------------------------------------------------------
class TestCumulativeMortality:
    def test_half_die_in_one_year(self):
        # 100 planted, 50 survive -> mortality rate 0.5
        assert cumulative_mortality_rate(100, 50) == pytest.approx(0.5)

    def test_all_survive(self):
        assert cumulative_mortality_rate(200, 200) == pytest.approx(0.0)

    def test_all_die(self):
        assert cumulative_mortality_rate(120, 0) == pytest.approx(1.0)

    def test_zero_initial_raises(self):
        with pytest.raises(ValueError, match="initial_count must be > 0"):
            cumulative_mortality_rate(0, 0)

    def test_alive_exceeds_initial_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            cumulative_mortality_rate(100, 150)

    def test_negative_alive_raises(self):
        with pytest.raises(ValueError, match="alive_count must be >= 0"):
            cumulative_mortality_rate(100, -5)

    def test_none_inputs_raise(self):
        with pytest.raises(ValueError):
            cumulative_mortality_rate(None, 50)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# annual_mortality_rate
# ---------------------------------------------------------------------------
class TestAnnualMortality:
    def test_one_year_matches_cumulative(self):
        # Over exactly 1 year the annual rate equals the cumulative rate.
        assert annual_mortality_rate(100, 50, 1.0) == pytest.approx(0.5)

    def test_exponential_decay_two_years(self):
        # 100 -> 25 over 2 years: annual = 1 - (0.25)^(1/2) = 1 - 0.5 = 0.5
        assert annual_mortality_rate(100, 25, 2.0) == pytest.approx(0.5)

    def test_exponential_decay_three_years(self):
        # 1000 -> 512 over 3 years: survival per year = 0.8, so m = 0.2
        assert annual_mortality_rate(1000, 512, 3.0) == pytest.approx(0.2)

    def test_zero_mortality(self):
        assert annual_mortality_rate(100, 100, 5.0) == pytest.approx(0.0)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="years must be > 0"):
            annual_mortality_rate(100, 50, -1.0)

    def test_zero_time_raises(self):
        with pytest.raises(ValueError, match="years must be > 0"):
            annual_mortality_rate(100, 50, 0.0)

    def test_nan_time_raises(self):
        with pytest.raises(ValueError, match="finite"):
            annual_mortality_rate(100, 50, float("nan"))


# ---------------------------------------------------------------------------
# compute_cohort_metrics
# ---------------------------------------------------------------------------
def _frame(rows):
    return pd.DataFrame(rows)


class TestCohortMetrics:
    def test_basic_computation(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=50,
                    assessment_date="2024-01-01",
                ),
            ]
        )
        out = compute_cohort_metrics(df)
        assert out["cumulative_mortality"].iloc[0] == pytest.approx(0.5)
        # Two years: annual = 1 - sqrt(0.5) ~ 0.2929
        assert out["annual_mortality"].iloc[0] == pytest.approx(
            1 - math.sqrt(0.5), abs=1e-3
        )
        assert out["years_elapsed"].iloc[0] == pytest.approx(2.0, abs=0.01)

    def test_input_not_mutated(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        snapshot = df.copy()
        compute_cohort_metrics(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_cohort_metrics(pd.DataFrame())

    def test_missing_column_raises(self):
        df = _frame([dict(site_id="A", species="X")])
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_cohort_metrics(df)

    def test_alive_gt_initial_raises(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=10,
                    alive_count=50,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        with pytest.raises(ValueError, match="greater than initial_count"):
            compute_cohort_metrics(df)

    def test_zero_initial_yields_nan(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=0,
                    alive_count=0,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        out = compute_cohort_metrics(df)
        assert math.isnan(out["cumulative_mortality"].iloc[0])
        assert math.isnan(out["annual_mortality"].iloc[0])

    def test_nan_counts_yield_nan(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=np.nan,
                    alive_count=10,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        out = compute_cohort_metrics(df)
        assert math.isnan(out["cumulative_mortality"].iloc[0])

    def test_non_positive_elapsed_yields_nan(self):
        # assessment before planting -> years <= 0
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2024-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        out = compute_cohort_metrics(df)
        assert math.isnan(out["annual_mortality"].iloc[0])


# ---------------------------------------------------------------------------
# survival_curve
# ---------------------------------------------------------------------------
class TestSurvivalCurve:
    def _two_cohort_frame(self):
        return _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2023-01-01",
                ),
                dict(
                    site_id="B",
                    species="Shorea leprosula",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=60,
                    assessment_date="2024-01-01",
                ),
            ]
        )

    def test_pooled_curve_has_anchor(self):
        curve = survival_curve(self._two_cohort_frame())
        assert curve.iloc[0]["years_elapsed"] == 0.0
        assert curve.iloc[0]["survival"] == pytest.approx(1.0)
        # Monotonically non-increasing survival.
        survs = curve["survival"].tolist()
        assert all(earlier >= later for earlier, later in zip(survs, survs[1:]))

    def test_weighted_mean_survival(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=200,
                    alive_count=100,
                    assessment_date="2023-01-01",
                ),
                dict(
                    site_id="B",
                    species="Acacia mangium",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        curve = survival_curve(df)
        # Weighted survival: (200*0.5 + 100*0.8) / 300 = 180/300 = 0.6
        row = curve[curve["years_elapsed"] > 0].iloc[0]
        assert row["survival"] == pytest.approx(0.6)

    def test_group_by_species(self):
        curve = survival_curve(self._two_cohort_frame(), group_by="species")
        assert set(curve["species"].unique()) == {
            "Acacia mangium",
            "Shorea leprosula",
        }
        # Each group gets the t=0 anchor.
        anchors = curve[curve["years_elapsed"] == 0.0]
        assert len(anchors) == 2

    def test_group_by_unknown_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            survival_curve(self._two_cohort_frame(), group_by="nope")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            survival_curve(pd.DataFrame())


# ---------------------------------------------------------------------------
# mortality_hotspots
# ---------------------------------------------------------------------------
class TestHotspots:
    def test_high_mortality_flagged(self):
        df = _frame(
            [
                dict(
                    site_id="hot",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=40,
                    assessment_date="2023-01-01",
                ),
                dict(
                    site_id="cool",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=95,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        result = mortality_hotspots(df, by="site_id", threshold=0.30)
        hot_row = result[result["site_id"] == "hot"].iloc[0]
        cool_row = result[result["site_id"] == "cool"].iloc[0]
        assert hot_row["is_hotspot"] is np.True_ or hot_row["is_hotspot"] is True
        assert bool(hot_row["is_hotspot"]) is True
        assert bool(cool_row["is_hotspot"]) is False

    def test_threshold_bounds(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=50,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        with pytest.raises(ValueError, match="threshold"):
            mortality_hotspots(df, threshold=1.5)

    def test_group_column_missing(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=50,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        with pytest.raises(ValueError, match="not found"):
            mortality_hotspots(df, by="zzz")

    def test_sorted_descending(self):
        df = _frame(
            [
                dict(
                    site_id="low",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=90,
                    assessment_date="2023-01-01",
                ),
                dict(
                    site_id="high",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=30,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        result = mortality_hotspots(df, by="site_id", threshold=0.5)
        assert result.iloc[0]["site_id"] == "high"
        assert result.iloc[-1]["site_id"] == "low"


# ---------------------------------------------------------------------------
# cause_of_death_breakdown
# ---------------------------------------------------------------------------
class TestCauseBreakdown:
    def test_counts_dead_trees_per_cause(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=70,  # 30 dead
                    assessment_date="2023-01-01",
                    cause_of_death="drought",
                ),
                dict(
                    site_id="B",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=90,  # 10 dead
                    assessment_date="2023-01-01",
                    cause_of_death="pest",
                ),
                dict(
                    site_id="C",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,  # 20 dead
                    assessment_date="2023-01-01",
                    cause_of_death="drought",
                ),
            ]
        )
        result = cause_of_death_breakdown(df)
        drought = result[result["cause"] == "drought"].iloc[0]
        pest = result[result["cause"] == "pest"].iloc[0]
        assert drought["deaths"] == 50
        assert pest["deaths"] == 10
        assert drought["share"] == pytest.approx(50 / 60)

    def test_blank_cause_becomes_unknown(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2023-01-01",
                    cause_of_death=None,
                ),
            ]
        )
        result = cause_of_death_breakdown(df)
        assert "unknown" in result["cause"].tolist()

    def test_missing_cause_column_raises(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2023-01-01",
                ),
            ]
        )
        with pytest.raises(ValueError, match="cause column"):
            cause_of_death_breakdown(df)


# ---------------------------------------------------------------------------
# summarize & demo data integration
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_summary_keys(self):
        df = _frame(
            [
                dict(
                    site_id="A",
                    species="X",
                    planted_date="2022-01-01",
                    initial_count=100,
                    alive_count=80,
                    assessment_date="2024-01-01",
                ),
            ]
        )
        s = summarize(df)
        assert set(s) == {
            "total_planted",
            "total_alive",
            "cumulative_mortality",
            "avg_annual_mortality",
        }
        assert s["total_planted"] == 100
        assert s["total_alive"] == 80
        assert s["cumulative_mortality"] == pytest.approx(0.2)


class TestDemoDataIntegration:
    def test_demo_csv_loads(self):
        assert DEMO_CSV.exists(), f"demo sample not found: {DEMO_CSV}"
        df = pd.read_csv(DEMO_CSV)
        assert len(df) >= 15
        expected = {
            "site_id",
            "species",
            "planted_date",
            "initial_count",
            "alive_count",
            "assessment_date",
            "cause_of_death",
        }
        assert expected.issubset(df.columns)

    def test_demo_cohort_metrics_realistic(self):
        df = pd.read_csv(DEMO_CSV)
        metrics = compute_cohort_metrics(df)
        # Cumulative mortality should land in the 10-45% range per row.
        cm = metrics["cumulative_mortality"].dropna()
        assert (cm >= 0.08).all()
        assert (cm <= 0.45).all()

    def test_demo_hotspots_runs(self):
        df = pd.read_csv(DEMO_CSV)
        result = mortality_hotspots(df, by="site_id", threshold=0.15)
        assert len(result) > 0
        assert "annual_mortality" in result.columns

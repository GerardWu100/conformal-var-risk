"""Regression tests for Value-at-Risk and Expected Shortfall summary metrics."""

from __future__ import annotations

import pandas as pd
import pytest
from conformal_var_risk.config import EvaluationConfig, EvaluationPeriodConfig
from conformal_var_risk.evaluation.metrics import (
    clopper_pearson_interval,
    compute_summary_metrics,
)


def test_clopper_pearson_interval_quantifies_zero_breach_uncertainty() -> None:
    """Zero events in 153 trials should still allow a nonzero event rate."""
    lower, upper = clopper_pearson_interval(
        successes=0,
        trials=153,
        confidence_level=0.95,
    )

    assert lower == 0.0
    assert upper == pytest.approx(0.0238219913)


def test_es_backtest_is_undefined_when_predicted_es_is_zero() -> None:
    """Zero ES forecasts should yield missing diagnostics, not infinities."""
    backtest_results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "asset": ["portfolio"] * 2,
            "model": ["historical"] * 2,
            "alpha": [0.05] * 2,
            "predicted_lower_quantile": [0.01, 0.01],
            "predicted_var": [0.0, 0.0],
            "predicted_es": [0.0, 0.0],
            "interval_lower": [0.01, 0.01],
            "interval_upper": [0.02, 0.02],
            "actual_return": [0.0, 0.02],
            "violation": [True, False],
        }
    )
    evaluation_config = EvaluationConfig(
        es_backtest_confidence=0.95,
        periods=[
            EvaluationPeriodConfig(
                name="toy",
                start_date="2020-01-01",
                end_date="2020-01-02",
            )
        ],
    )

    metric_row = compute_summary_metrics(
        backtest_results=backtest_results,
        evaluation_config=evaluation_config,
    ).iloc[0]

    assert pd.isna(metric_row["es_z1_statistic"])
    assert pd.isna(metric_row["es_z2_statistic"])
    assert metric_row["es_backtest_status"] == "insufficient_tail_data"


def test_summary_metrics_include_expected_shortfall_backtests() -> None:
    """ES backtest statistics should be present and finite on a toy sample."""
    backtest_results = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
            ),
            "asset": ["portfolio"] * 4,
            "model": ["historical"] * 4,
            "alpha": [0.05] * 4,
            "predicted_lower_quantile": [-0.02, -0.02, -0.02, -0.02],
            "predicted_var": [0.02, 0.02, 0.02, 0.02],
            "predicted_es": [0.03, 0.03, 0.03, 0.03],
            "interval_lower": [-0.02, -0.02, -0.02, -0.02],
            "interval_upper": [0.02, 0.02, 0.02, 0.02],
            "actual_return": [-0.03, -0.01, -0.04, 0.01],
            "violation": [True, False, True, False],
        }
    )
    evaluation_config = EvaluationConfig(
        es_backtest_confidence=0.95,
        periods=[
            EvaluationPeriodConfig(
                name="toy",
                start_date="2020-01-01",
                end_date="2020-01-06",
            )
        ],
    )

    metrics_frame = compute_summary_metrics(
        backtest_results=backtest_results,
        evaluation_config=evaluation_config,
    )

    metric_row = metrics_frame.iloc[0]
    assert metric_row["period"] == "toy"
    assert metric_row["es_z1_statistic"] < 0.0
    assert metric_row["es_z2_statistic"] < 0.0
    assert metric_row["avg_predicted_es"] >= metric_row["avg_predicted_var"]


def test_summary_metrics_include_quantile_loss_and_skip_conformal_es_status() -> None:
    """Conformal rows should expose quantile loss but not ES status ranking."""
    backtest_results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "asset": ["portfolio"] * 3,
            "model": ["conformal"] * 3,
            "alpha": [0.05] * 3,
            "predicted_lower_quantile": [-0.02, -0.02, -0.02],
            "predicted_var": [0.02, 0.02, 0.02],
            "predicted_es": [0.03, 0.03, 0.03],
            "interval_lower": [-0.02, -0.02, -0.02],
            "interval_upper": [0.02, 0.02, 0.02],
            "actual_return": [-0.03, -0.01, -0.04],
            "violation": [True, False, True],
        }
    )
    evaluation_config = EvaluationConfig(
        es_backtest_confidence=0.95,
        periods=[
            EvaluationPeriodConfig(
                name="toy",
                start_date="2020-01-01",
                end_date="2020-01-03",
            )
        ],
    )

    metrics_frame = compute_summary_metrics(
        backtest_results=backtest_results,
        evaluation_config=evaluation_config,
    )

    metric_row = metrics_frame.iloc[0]
    assert "avg_quantile_loss" in metrics_frame.columns
    assert metric_row["avg_quantile_loss"] == pytest.approx(0.009666666666666667)
    assert metric_row["es_backtest_status"] == "not_reported"

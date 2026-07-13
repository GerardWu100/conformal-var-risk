"""Regression tests for the VaR model interface and rolling backtest loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_var_risk.evaluation.backtest import run_backtest
from conformal_var_risk.models import garch as garch_module
from conformal_var_risk.models.base import VaRModel
from conformal_var_risk.models.conformal import AdaptiveConformalVaRModel
from conformal_var_risk.models.fhs import FilteredHistoricalSimulationVaRModel
from conformal_var_risk.models.garch import GarchVaRModel
from conformal_var_risk.models.historical import HistoricalSimulationVaRModel


def test_historical_simulation_uses_lower_quantile_contract_for_var() -> None:
    """Historical simulation should derive VaR from the lower-tail quantile."""
    returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=float)
    model = HistoricalSimulationVaRModel(window=5)

    model.fit(returns)

    predicted_var = model.predict_var(alpha=0.2)
    lower_quantile = model.predict_lower_quantile(alpha=0.2)
    predicted_es = model.predict_es(alpha=0.2)
    lower_bound, upper_bound = model.predict_interval(alpha=0.2)

    assert predicted_var > 0.0
    assert predicted_es >= predicted_var
    assert lower_bound < upper_bound
    assert np.isclose(predicted_var, max(-lower_quantile, 0.0))


def test_garch_uses_lower_quantile_contract_for_var() -> None:
    """GARCH should derive VaR from the lower-tail quantile."""
    returns = np.linspace(-0.03, 0.03, num=120, dtype=float)
    model = GarchVaRModel(distribution="normal")

    model.fit(returns)

    predicted_var = model.predict_var(alpha=0.2)
    lower_quantile = model.predict_lower_quantile(alpha=0.2)
    predicted_es = model.predict_es(alpha=0.2)
    lower_bound, upper_bound = model.predict_interval(alpha=0.2)

    assert predicted_var >= 0.0
    assert predicted_es >= predicted_var
    assert lower_bound < upper_bound
    assert np.isclose(predicted_var, max(-lower_quantile, 0.0))


def test_failed_garch_refit_discards_stale_prior_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rolling fit should use current-window fallback statistics."""
    model = GarchVaRModel(distribution="normal")
    model._last_result = object()

    def raise_fit_error(*args: object, **kwargs: object) -> object:
        """Raise a deterministic fitting error for the fallback-path test."""
        del args
        del kwargs
        raise RuntimeError("forced rolling-fit failure")

    monkeypatch.setattr(garch_module, "arch_model", raise_fit_error)
    returns = np.linspace(-0.01, 0.02, num=40, dtype=float)

    model.fit(returns)

    assert model._last_result is None
    assert np.isfinite(model.predict_lower_quantile(alpha=0.05))


def test_adaptive_conformal_var_matches_requested_lower_quantile_contract() -> None:
    """Adaptive conformal VaR should derive from the public lower-tail forecast."""
    returns = np.linspace(-0.02, 0.02, num=80, dtype=float)
    model = AdaptiveConformalVaRModel(window=60, mean_window=10, learning_rate=0.05)

    model.fit(returns)

    lower_quantile = model.predict_lower_quantile(alpha=0.05)
    predicted_var = model.predict_var(alpha=0.05)

    assert np.isclose(predicted_var, max(-lower_quantile, 0.0))


def test_adaptive_conformal_uses_finite_sample_corrected_score_rank() -> None:
    """Conformal score selection should use the corrected order statistic."""
    scores = np.arange(1.0, 11.0)

    selected_score = AdaptiveConformalVaRModel._finite_sample_upper_quantile(
        scores=scores,
        alpha=0.2,
    )

    # ceil((10 + 1) * (1 - 0.2)) = 9, so the ninth sorted score is selected.
    assert selected_score == 9.0


def test_adaptive_conformal_non_breach_recovers_alpha_toward_target() -> None:
    """A quiet period after a breach should relax alpha back toward the target."""
    returns = np.linspace(-0.03, 0.03, num=80, dtype=float)
    model = AdaptiveConformalVaRModel(window=60, mean_window=10, learning_rate=0.05)

    model.fit(returns)

    lower_quantile = model.predict_lower_quantile(alpha=0.05)
    breach_return = lower_quantile - 0.01

    model.observe(realized_return=breach_return)

    breached_alpha = model.current_alpha
    recovery_return = max(lower_quantile + 0.01, 0.01)

    model.observe(realized_return=recovery_return)

    assert model.current_alpha > breached_alpha


def test_adaptive_conformal_lower_tail_breach_tightens_alpha() -> None:
    """A realized return below the lower-tail forecast should tighten alpha."""
    returns = np.linspace(-0.02, 0.02, num=80, dtype=float)
    model = AdaptiveConformalVaRModel(window=60, mean_window=10, learning_rate=0.05)

    model.fit(returns)

    initial_alpha = model.current_alpha
    lower_quantile = model.predict_lower_quantile(alpha=0.05)
    realized_return = lower_quantile - 0.01

    assert realized_return < lower_quantile

    model.observe(realized_return=realized_return)

    assert model.current_alpha < initial_alpha


def test_backtest_runs_all_rows_for_one_asset_and_model() -> None:
    """The rolling backtest should emit one row per evaluation day and alpha."""
    dates = pd.date_range("2021-01-01", periods=40, freq="B")
    returns = np.linspace(-0.03, 0.03, num=40, dtype=float)
    frame = pd.DataFrame({"asset_a": returns}, index=dates)

    results = run_backtest(
        returns_by_asset=frame,
        alphas=[0.05],
        calibration_window=20,
        model_factories={
            "historical": lambda alpha: HistoricalSimulationVaRModel(window=10)
        },
    )

    assert len(results) == 20
    assert list(results["date"]) == list(dates[20:])
    assert set(results["model"]) == {"historical"}
    assert set(results["asset"]) == {"asset_a"}
    assert results["predicted_var"].ge(0.0).all()
    assert results["predicted_es"].ge(results["predicted_var"]).all()
    assert results["predicted_lower_quantile"].le(results["interval_upper"]).all()
    assert results["violation"].equals(
        results["actual_return"] < results["predicted_lower_quantile"]
    )


class RecordingModel(VaRModel):
    """Minimal test double that records the last training window."""

    def __init__(self) -> None:
        """Initialize the capture buffer used by the timing regression tests."""
        self.last_window = np.empty(0, dtype=float)

    def fit(self, returns: np.ndarray) -> None:
        """Record the most recent trailing calibration window."""
        self.last_window = returns.copy()

    def predict_var(self, alpha: float) -> float:
        """Return a fixed positive VaR for deterministic assertions."""
        del alpha
        return 0.02

    def predict_lower_quantile(self, alpha: float) -> float:
        """Return the lower-tail return threshold used by the backtest."""
        del alpha
        return -0.02

    def predict_es(self, alpha: float) -> float:
        """Return a fixed positive ES that stays above the VaR."""
        del alpha
        return 0.03

    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return a symmetric interval whose lower bound is the test threshold."""
        del alpha
        return -0.02, 0.02


class PositiveQuantileModel(RecordingModel):
    """Test double whose forecast quantile is a small positive return."""

    def predict_lower_quantile(self, alpha: float) -> float:
        """Return a positive quantile to test quantile-score consistency."""
        del alpha
        return 0.02

    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return an interval containing the positive lower quantile."""
        del alpha
        return 0.02, 0.04


def test_backtest_forecasts_next_business_day_without_skipping() -> None:
    """Backtest rows should be stamped on the realized date, not a future one."""
    dates = pd.date_range("2021-01-01", periods=6, freq="B")
    frame = pd.DataFrame({"asset_a": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}, index=dates)

    results = run_backtest(
        returns_by_asset=frame,
        alphas=[0.05],
        calibration_window=2,
        model_factories={"recording": lambda alpha: RecordingModel()},
    )

    assert list(results["date"]) == list(dates[2:])
    assert len(results) == 4


def test_backtest_does_not_truncate_positive_return_quantile_for_scoring() -> None:
    """Coverage and pinball loss should score the model's actual quantile."""
    dates = pd.date_range("2021-01-01", periods=3, freq="B")
    frame = pd.DataFrame({"asset_a": [0.03, 0.03, 0.01]}, index=dates)

    results = run_backtest(
        returns_by_asset=frame,
        alphas=[0.05],
        calibration_window=2,
        model_factories={"positive": lambda alpha: PositiveQuantileModel()},
    )

    row = results.iloc[0]
    assert row["predicted_lower_quantile"] == 0.02
    assert row["predicted_var"] == 0.0
    assert bool(row["violation"])


def test_filtered_historical_uses_lower_quantile_contract_for_var() -> None:
    """Filtered historical simulation should derive VaR from lower quantile."""
    returns = np.linspace(-0.03, 0.03, num=120, dtype=float)
    rng = np.random.default_rng(7)
    model = FilteredHistoricalSimulationVaRModel(simulations=2_000, rng=rng)

    model.fit(returns)

    predicted_var = model.predict_var(alpha=0.2)
    lower_quantile = model.predict_lower_quantile(alpha=0.2)
    predicted_es = model.predict_es(alpha=0.2)
    lower_bound, upper_bound = model.predict_interval(alpha=0.2)

    assert predicted_var >= 0.0
    assert predicted_es >= predicted_var
    assert lower_bound < upper_bound
    assert np.isclose(predicted_var, max(-lower_quantile, 0.0))

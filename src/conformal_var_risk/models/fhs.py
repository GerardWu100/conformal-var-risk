"""Filtered historical simulation Value-at-Risk model.

This benchmark combines a GARCH(1,1) volatility forecast with an empirical
bootstrap over standardized shocks. The GARCH layer estimates the next-day
conditional mean and volatility, while the historical bootstrap preserves the
observed non-Gaussian shock shape.
"""

from __future__ import annotations

import logging

from arch import arch_model
import numpy as np

from conformal_var_risk.models.base import VaRModel

MIN_SIGMA = 1e-8
PERCENT_SCALE = 100.0


class FilteredHistoricalSimulationVaRModel(VaRModel):
    """One-step-ahead GARCH-scaled empirical-shock Value-at-Risk benchmark."""

    def __init__(
        self,
        simulations: int = 10_000,
        p: int = 1,
        q: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store the simulation settings and persistent random-number generator."""
        self.simulations = simulations
        self.p = p
        self.q = q
        self.rng = rng or np.random.default_rng()
        self._last_result = None
        self._fallback_mean = 0.0
        self._fallback_sigma = 0.01
        self._standardized_shocks = np.empty(0, dtype=float)
        self._simulated_returns = np.empty(0, dtype=float)

    def fit(self, returns: np.ndarray) -> None:
        """Estimate the GARCH state and bootstrap next-day return scenarios.

        Parameters
        ----------
        returns
            One-dimensional array of daily log returns with shape
            ``(n_observations,)`` in decimal units.
        """
        clean_returns = np.asarray(returns, dtype=float)
        self._fallback_mean = float(np.mean(clean_returns))
        self._fallback_sigma = float(max(np.std(clean_returns, ddof=1), MIN_SIGMA))
        self._last_result = None
        self._standardized_shocks = self._compute_fallback_standardized_shocks(
            clean_returns
        )

        try:
            fitted_model = arch_model(
                clean_returns * PERCENT_SCALE,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist="normal",
                rescale=False,
            )
            self._last_result = fitted_model.fit(disp="off", show_warning=False)
            self._standardized_shocks = self._extract_standardized_shocks(clean_returns)
        except Exception as error:
            logging.warning(
                "Filtered historical GARCH fit failed; using fallback state: %s", error
            )

        forecast_mean, forecast_sigma = self._forecast_location_scale()
        resampled_shocks = self.rng.choice(
            self._standardized_shocks, size=self.simulations, replace=True
        )
        self._simulated_returns = forecast_mean + forecast_sigma * resampled_shocks

    def predict_lower_quantile(self, alpha: float) -> float:
        """Return the empirical lower-tail quantile from simulated returns."""
        return float(np.quantile(self._simulated_returns, alpha))

    def predict_es(self, alpha: float) -> float:
        """Return the empirical Expected Shortfall from the simulated tail."""
        lower_bound = self.predict_lower_quantile(alpha=alpha)
        tail_returns = self._simulated_returns[self._simulated_returns <= lower_bound]
        if len(tail_returns) == 0:
            return self.predict_var(alpha=alpha)

        tail_loss = -float(np.mean(tail_returns))
        return max(tail_loss, self.predict_var(alpha=alpha))

    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return lower and upper quantiles from the simulated return sample."""
        lower_bound = float(np.quantile(self._simulated_returns, alpha))
        upper_bound = float(np.quantile(self._simulated_returns, 1.0 - alpha))
        return lower_bound, upper_bound

    def _extract_standardized_shocks(self, returns: np.ndarray) -> np.ndarray:
        """Standardize returns by the fitted conditional mean and volatility."""
        fitted_result = self._last_result
        if fitted_result is None:
            return self._compute_fallback_standardized_shocks(returns)

        mean_percent = float(fitted_result.params["mu"])
        conditional_sigma_percent = np.asarray(
            fitted_result.conditional_volatility, dtype=float
        )
        safe_sigma_percent = np.maximum(conditional_sigma_percent, MIN_SIGMA)
        standardized_shocks = (
            (returns * PERCENT_SCALE) - mean_percent
        ) / safe_sigma_percent

        finite_shocks = standardized_shocks[np.isfinite(standardized_shocks)]
        if len(finite_shocks) == 0:
            return self._compute_fallback_standardized_shocks(returns)
        return finite_shocks

    def _forecast_location_scale(self) -> tuple[float, float]:
        """Extract the one-step-ahead mean and volatility forecast in return units."""
        fitted_result = self._last_result
        if fitted_result is None:
            return self._fallback_mean, self._fallback_sigma

        forecast = fitted_result.forecast(horizon=1, reindex=False)
        mean_percent = float(forecast.mean.iloc[-1, 0])
        variance_percent = float(forecast.variance.iloc[-1, 0])
        mean_return = mean_percent / PERCENT_SCALE
        sigma_return = max(
            np.sqrt(max(variance_percent, 0.0)) / PERCENT_SCALE, MIN_SIGMA
        )
        return mean_return, sigma_return

    def _compute_fallback_standardized_shocks(self, returns: np.ndarray) -> np.ndarray:
        """Build a stable bootstrap sample when the GARCH fit is unavailable."""
        demeaned_returns = returns - self._fallback_mean
        standardized_shocks = demeaned_returns / self._fallback_sigma
        finite_shocks = standardized_shocks[np.isfinite(standardized_shocks)]
        if len(finite_shocks) == 0:
            return np.array([0.0], dtype=float)
        return finite_shocks

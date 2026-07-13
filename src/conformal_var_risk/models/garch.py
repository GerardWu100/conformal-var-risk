"""Parametric GARCH Value-at-Risk models."""

from __future__ import annotations

import logging

from arch import arch_model
import numpy as np
from scipy.stats import norm, t

from conformal_var_risk.models.base import VaRModel

MIN_SIGMA = 1e-8


class GarchVaRModel(VaRModel):
    """One-step-ahead GARCH(1,1) Value-at-Risk forecast under a chosen innovation law."""

    def __init__(self, distribution: str, p: int = 1, q: int = 1) -> None:
        """Store the distribution family and lag orders."""
        self.distribution = distribution
        self.p = p
        self.q = q
        self._last_result = None
        self._fallback_mean = 0.0
        self._fallback_sigma = 0.01

    def fit(self, returns: np.ndarray) -> None:
        """Estimate the GARCH model on percent-scaled daily log returns."""
        clean_returns = np.asarray(returns, dtype=float)
        self._fallback_mean = float(np.mean(clean_returns))
        self._fallback_sigma = float(max(np.std(clean_returns, ddof=1), MIN_SIGMA))
        # A failed rolling refit must not reuse parameters from the prior date.
        self._last_result = None

        distribution_name = "normal" if self.distribution == "normal" else "t"
        try:
            fitted_model = arch_model(
                clean_returns * 100.0,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=distribution_name,
                rescale=False,
            )
            self._last_result = fitted_model.fit(disp="off", show_warning=False)
        except Exception as error:
            logging.warning("GARCH fit failed; using fallback state: %s", error)

    def predict_lower_quantile(self, alpha: float) -> float:
        """Return the GARCH lower-tail return quantile for the next day."""
        mu, sigma = self._forecast_location_scale()
        return mu + self._innovation_quantile(alpha=alpha) * sigma

    def predict_es(self, alpha: float) -> float:
        """Return the positive Expected Shortfall implied by the innovation law."""
        mu, sigma = self._forecast_location_scale()
        standardized_tail_mean = self._standardized_tail_mean(alpha=alpha)
        tail_mean = mu + sigma * standardized_tail_mean
        return max(-tail_mean, self.predict_var(alpha=alpha))

    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return lower and upper predictive quantiles for the next-day return."""
        mu, sigma = self._forecast_location_scale()
        lower_bound = mu + self._innovation_quantile(alpha=alpha) * sigma
        upper_bound = mu + self._innovation_quantile(alpha=1.0 - alpha) * sigma
        return lower_bound, upper_bound

    def _forecast_location_scale(self) -> tuple[float, float]:
        """Extract the one-step-ahead conditional mean and volatility forecast."""
        if self._last_result is None:
            return self._fallback_mean, self._fallback_sigma

        forecast = self._last_result.forecast(horizon=1, reindex=False)
        mean_percent = float(forecast.mean.iloc[-1, 0])
        variance_percent = float(forecast.variance.iloc[-1, 0])
        mean_return = mean_percent / 100.0
        sigma_return = max(np.sqrt(max(variance_percent, 0.0)) / 100.0, MIN_SIGMA)
        return mean_return, sigma_return

    def _student_t_degrees_of_freedom(self) -> float:
        """Read fitted tail heaviness when available, otherwise use a stable default."""
        if self._last_result is not None and "nu" in self._last_result.params.index:
            return float(self._last_result.params["nu"])
        return 8.0

    def _innovation_quantile(self, alpha: float) -> float:
        """Read the standardized innovation quantile for the requested distribution."""
        clipped_alpha = float(np.clip(alpha, 1e-12, 1.0 - 1e-12))
        if self.distribution == "normal":
            return float(norm.ppf(clipped_alpha))

        degrees_of_freedom = self._student_t_degrees_of_freedom()
        scale_adjustment = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        return float(t.ppf(clipped_alpha, df=degrees_of_freedom) * scale_adjustment)

    def _standardized_tail_mean(self, alpha: float) -> float:
        """Return E[Z | Z <= q_alpha] for the standardized innovation Z."""
        clipped_alpha = float(np.clip(alpha, 1e-12, 1.0 - 1e-12))
        if self.distribution == "normal":
            quantile = float(norm.ppf(clipped_alpha))
            return float(-norm.pdf(quantile) / clipped_alpha)

        degrees_of_freedom = self._student_t_degrees_of_freedom()
        raw_quantile = float(t.ppf(clipped_alpha, df=degrees_of_freedom))
        raw_density = float(t.pdf(raw_quantile, df=degrees_of_freedom))
        raw_tail_mean = (
            -(
                (degrees_of_freedom + raw_quantile**2)
                / ((degrees_of_freedom - 1.0) * clipped_alpha)
            )
            * raw_density
        )
        scale_adjustment = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        return float(scale_adjustment * raw_tail_mean)

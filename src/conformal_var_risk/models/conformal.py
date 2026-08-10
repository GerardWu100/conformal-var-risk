"""Adaptive conformal Value-at-Risk model with one-sided lower-tail updates."""

from __future__ import annotations

import numpy as np

from conformal_var_risk.models.base import VaRModel

MIN_ALPHA = 0.001
MAX_ALPHA = 0.999
MIN_DIAGNOSTIC_INTERVAL_WIDTH = 1e-12


class AdaptiveConformalVaRModel(VaRModel):
    """Adaptive one-sided conformal forecaster for one-step-ahead lower tails."""

    def __init__(
        self,
        window: int = 500,
        mean_window: int = 20,
        learning_rate: float = 0.005,
        initial_alpha: float = 0.05,
    ) -> None:
        """Store calibration, base-predictor, and adaptation hyperparameters."""
        self.window = window
        self.mean_window = mean_window
        self.learning_rate = learning_rate
        self.current_alpha = initial_alpha
        self._target_alpha = initial_alpha
        self._center = 0.0
        self._scores = np.empty(0, dtype=float)
        self._pseudo_returns = np.empty(0, dtype=float)
        self._last_lower_quantile = 0.0

    def fit(self, returns: np.ndarray) -> None:
        """Recompute one-sided lower-tail calibration scores."""
        trailing_returns = np.asarray(returns[-self.window :], dtype=float)
        if len(trailing_returns) <= self.mean_window:
            raise ValueError("Conformal model needs more returns than mean_window.")

        # One-sided scores compare each rolling-mean forecast to the realized return.
        rolling_centers = self._compute_rolling_mean(trailing_returns)
        realized_segment = trailing_returns[self.mean_window :]
        self._center = float(np.mean(trailing_returns[-self.mean_window :]))
        raw_shortfalls = rolling_centers - realized_segment
        self._scores = np.maximum(raw_shortfalls, 0.0)
        self._pseudo_returns = self._center - self._scores
        self._last_lower_quantile = 0.0

    def predict_var(self, alpha: float) -> float:
        """Estimate VaR from the public lower-tail quantile forecast."""
        lower_quantile = self.predict_lower_quantile(alpha=alpha)
        return max(-lower_quantile, 0.0)

    def predict_lower_quantile(self, alpha: float) -> float:
        """Return the conformal one-sided lower-tail return quantile."""
        self._target_alpha = alpha
        effective_alpha = float(np.clip(self.current_alpha, MIN_ALPHA, MAX_ALPHA))
        calibration_adjustment = self._finite_sample_upper_quantile(
            scores=self._scores,
            alpha=effective_alpha,
        )
        lower_quantile = self._center - calibration_adjustment
        self._last_lower_quantile = lower_quantile
        return lower_quantile

    def predict_es(self, alpha: float) -> float:
        """Estimate ES from the empirical lower tail implied by one-sided scores."""
        lower_quantile = self.predict_lower_quantile(alpha=alpha)
        tail_returns = self._pseudo_returns[self._pseudo_returns <= lower_quantile]
        predicted_var = max(-lower_quantile, 0.0)
        if len(tail_returns) == 0:
            return predicted_var
        return max(-float(np.mean(tail_returns)), predicted_var)

    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return a diagnostic interval built from the lower-tail forecast."""
        lower_bound = self.predict_lower_quantile(alpha=alpha)
        half_width = max(abs(self._center - lower_bound), MIN_DIAGNOSTIC_INTERVAL_WIDTH)
        upper_bound = self._center + half_width
        return lower_bound, upper_bound

    def observe(self, realized_return: float, alpha: float | None = None) -> None:
        """Update the adaptive tail level with a one-sided target-seeking rule."""
        del alpha
        # Lower alpha after a breach, which selects a higher score rank and a more
        # conservative next boundary; raise it after a quiet day.
        breach_indicator = float(realized_return < self._last_lower_quantile)
        updated_alpha = self.current_alpha + self.learning_rate * (
            self._target_alpha - breach_indicator
        )
        self.current_alpha = float(np.clip(updated_alpha, MIN_ALPHA, MAX_ALPHA))

    def _compute_rolling_mean(self, returns: np.ndarray) -> np.ndarray:
        """Compute one-step-ahead rolling-mean forecasts over the input sample."""
        rolling_means: list[float] = []
        for row_number in range(self.mean_window, len(returns)):
            history = returns[row_number - self.mean_window : row_number]
            rolling_means.append(float(np.mean(history)))
        return np.asarray(rolling_means, dtype=float)

    @staticmethod
    def _finite_sample_upper_quantile(scores: np.ndarray, alpha: float) -> float:
        """Return the split-conformal finite-sample upper score quantile.

        Parameters
        ----------
        scores
            One-dimensional calibration nonconformity scores.
        alpha
            Effective lower-tail error probability in ``(0, 1)``.

        Returns
        -------
        float
            Order statistic at rank ``ceil((n + 1) * (1 - alpha))``, clipped
            to the largest available score when the requested rank exceeds
            the calibration sample size ``n``.

        Notes
        -----
        The correction is the standard split-conformal finite-sample rank.
        Financial time-series scores are not exchangeable, so using this rank
        does not by itself establish a finite-sample coverage guarantee here.
        """
        if len(scores) == 0:
            raise ValueError("Conformal calibration scores must not be empty.")

        sample_size = len(scores)
        rank = int(np.ceil((sample_size + 1) * (1.0 - alpha)))
        clipped_rank = int(np.clip(rank, 1, sample_size))
        return float(np.partition(scores, clipped_rank - 1)[clipped_rank - 1])

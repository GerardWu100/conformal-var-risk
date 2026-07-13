"""Historical simulation Value-at-Risk model."""

from __future__ import annotations

import numpy as np

from conformal_var_risk.models.base import VaRModel


class HistoricalSimulationVaRModel(VaRModel):
    """Empirical-quantile Value-at-Risk over a trailing fixed window."""

    def __init__(self, window: int = 250) -> None:
        """Store the trailing-window length used for empirical quantiles."""
        self.window = window
        self._window_returns = np.empty(0, dtype=float)

    def fit(self, returns: np.ndarray) -> None:
        """Keep the most recent observations needed for historical simulation."""
        self._window_returns = np.asarray(returns[-self.window :], dtype=float)

    def predict_lower_quantile(self, alpha: float) -> float:
        """Return the empirical lower-tail return quantile itself."""
        return self._compute_quantile(alpha=alpha)

    def predict_es(self, alpha: float) -> float:
        """Average the empirical tail losses beyond the VaR cutoff."""
        lower_bound = self._compute_quantile(alpha=alpha)
        tail_returns = self._window_returns[self._window_returns <= lower_bound]
        if len(tail_returns) == 0:
            return self.predict_var(alpha=alpha)
        return max(-float(np.mean(tail_returns)), self.predict_var(alpha=alpha))

    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return the central empirical interval with total mass ``1 - 2 * alpha``."""
        lower_bound = self._compute_quantile(alpha=alpha)
        upper_bound = self._compute_quantile(alpha=1.0 - alpha)
        return lower_bound, upper_bound

    def _compute_quantile(self, alpha: float) -> float:
        """Read one empirical quantile from the trailing return sample."""
        return float(np.quantile(self._window_returns, alpha))

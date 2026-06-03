"""Shared interface for one-step-ahead lower-tail risk forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VaRModel(ABC):
    """Common stateful interface for one-day-ahead lower-tail risk forecasting.

    The interface includes an ``observe`` hook because adaptive conformal
    prediction must update its internal error-rate target after each realized
    return. Static benchmark models keep the default no-op implementation.
    """

    @abstractmethod
    def fit(self, returns: np.ndarray) -> None:
        """Fit the model on a one-dimensional array of daily log returns."""

    @abstractmethod
    def predict_lower_quantile(self, alpha: float) -> float:
        """Return the forecast lower-tail return quantile for the next day."""

    def predict_var(self, alpha: float) -> float:
        """Return the positive loss threshold implied by the lower quantile."""
        lower_quantile = self.predict_lower_quantile(alpha=alpha)
        return max(-lower_quantile, 0.0)

    @abstractmethod
    def predict_es(self, alpha: float) -> float:
        """Return the positive Expected Shortfall at the requested tail level."""

    @abstractmethod
    def predict_interval(self, alpha: float) -> tuple[float, float]:
        """Return the one-day diagnostic interval bounds."""

    def observe(self, realized_return: float, alpha: float | None = None) -> None:
        """Update model state after observing the next-day realized return."""
        del realized_return
        del alpha

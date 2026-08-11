"""Model registry for the conformal Value-at-Risk pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conformal_var_risk.config import ProjectConfig
from conformal_var_risk.models.base import VaRModel
from conformal_var_risk.models.conformal import AdaptiveConformalVaRModel
from conformal_var_risk.models.fhs import FilteredHistoricalSimulationVaRModel
from conformal_var_risk.models.garch import GarchVaRModel
from conformal_var_risk.models.historical import HistoricalSimulationVaRModel


def build_model_factories(
    config: ProjectConfig,
) -> dict[str, Callable[[float], VaRModel]]:
    """Build the alpha-aware registry used by the rolling backtest.

    Each factory receives the requested left-tail probability so adaptive
    models can initialize their internal coverage target before the first
    forecast.
    """
    return {
        "historical": lambda alpha: HistoricalSimulationVaRModel(
            window=config.historical.window
        ),
        "garch_normal": lambda alpha: GarchVaRModel(
            distribution="normal",
            p=config.garch.p,
            q=config.garch.q,
        ),
        "garch_student_t": lambda alpha: GarchVaRModel(
            distribution="student_t",
            p=config.garch.p,
            q=config.garch.q,
        ),
        "filtered_historical": lambda alpha: FilteredHistoricalSimulationVaRModel(
            simulations=config.filtered_historical.simulations,
            p=config.garch.p,
            q=config.garch.q,
            # Tie bootstrap draws to alpha so multi-alpha runs stay reproducible.
            rng=np.random.default_rng(config.run.seed + int(alpha * 100_000) + 17),
        ),
        "conformal": lambda alpha: AdaptiveConformalVaRModel(
            window=config.conformal.window,
            mean_window=config.conformal.mean_window,
            learning_rate=config.conformal.learning_rate,
            initial_alpha=alpha,
        ),
    }

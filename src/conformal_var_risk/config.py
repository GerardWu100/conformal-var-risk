"""Typed configuration loading for the offline conformal VaR pipeline.

All runtime choices live in one TOML file so dataset scope, preprocessing
assumptions, feature choices, and model settings stay explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class RunConfig:
    """Top-level runtime settings.

    Attributes
    ----------
    start_date
        Inclusive analysis start date in `YYYY-MM-DD` format.
    end_date
        Inclusive analysis end date in `YYYY-MM-DD` format.
    output_dir
        Output folder for derived parquet artifacts.
    seed
        Random seed used by stochastic model components.
    """

    start_date: str
    end_date: str
    output_dir: str
    seed: int


@dataclass(frozen=True)
class RawDataConfig:
    """Raw parquet input settings.

    Attributes
    ----------
    raw_dir
        Folder that stores tracked raw parquet files.
    file_pattern
        Glob pattern used to discover raw parquet parts.
    timezone
        Source timezone convention for `ts`. Expected value is `UTC`.
    regular_session_only
        Whether runtime preprocessing should enforce regular-session filtering.
    """

    raw_dir: str
    file_pattern: str
    timezone: str
    regular_session_only: bool


@dataclass(frozen=True)
class PortfolioConfig:
    """Configured symbols used in the study universe.

    Attributes
    ----------
    symbols
        Ordered list of symbols used for per-asset modeling.
    """

    symbols: list[str]


@dataclass(frozen=True)
class BacktestConfig:
    """Rolling backtest settings.

    Attributes
    ----------
    calibration_window
        Number of trailing observations used for model fitting.
    alphas
        Left-tail probabilities for Value-at-Risk evaluation.
    """

    calibration_window: int
    alphas: list[float]


@dataclass(frozen=True)
class EvaluationPeriodConfig:
    """Named date range used in summary reporting."""

    name: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings.

    Attributes
    ----------
    es_backtest_confidence
        Confidence level used for one-sided Expected Shortfall status labels.
    periods
        Named calendar splits used for summary tables.
    """

    es_backtest_confidence: float
    periods: list[EvaluationPeriodConfig]


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering settings."""

    include_realized_return_volatility_5d: bool


@dataclass(frozen=True)
class HistoricalConfig:
    """Historical simulation model settings."""

    window: int


@dataclass(frozen=True)
class GarchConfig:
    """GARCH model order settings."""

    p: int
    q: int


@dataclass(frozen=True)
class FilteredHistoricalConfig:
    """Filtered historical simulation settings."""

    simulations: int


@dataclass(frozen=True)
class ConformalConfig:
    """Adaptive conformal model settings."""

    window: int
    mean_window: int
    learning_rate: float


@dataclass(frozen=True)
class ProjectConfig:
    """Full typed runtime configuration object."""

    run: RunConfig
    raw_data: RawDataConfig
    portfolio: PortfolioConfig
    backtest: BacktestConfig
    evaluation: EvaluationConfig
    features: FeatureConfig
    historical: HistoricalConfig
    garch: GarchConfig
    filtered_historical: FilteredHistoricalConfig
    conformal: ConformalConfig


def load_config(config_path: Path) -> ProjectConfig:
    """Load the project TOML file into typed dataclasses.

    Parameters
    ----------
    config_path
        Path to the root `config.toml` file.

    Returns
    -------
    ProjectConfig
        Fully typed config object used across preprocessing, modeling, and
        evaluation modules.
    """
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    evaluation_periods = [
        EvaluationPeriodConfig(**period)
        for period in raw_config["evaluation"]["periods"]
    ]

    return ProjectConfig(
        run=RunConfig(**raw_config["run"]),
        raw_data=RawDataConfig(**raw_config["raw_data"]),
        portfolio=PortfolioConfig(**raw_config["portfolio"]),
        backtest=BacktestConfig(**raw_config["backtest"]),
        evaluation=EvaluationConfig(
            es_backtest_confidence=raw_config["evaluation"]["es_backtest_confidence"],
            periods=evaluation_periods,
        ),
        features=FeatureConfig(**raw_config["features"]),
        historical=HistoricalConfig(**raw_config["historical"]),
        garch=GarchConfig(**raw_config["garch"]),
        filtered_historical=FilteredHistoricalConfig(
            **raw_config["filtered_historical"]
        ),
        conformal=ConformalConfig(**raw_config["conformal"]),
    )

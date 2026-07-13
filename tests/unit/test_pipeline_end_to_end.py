"""End-to-end regression tests for the offline raw-parquet pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conformal_var_risk.pipeline import run_pipeline


def _write_toy_raw_cache(raw_dir: Path) -> None:
    """Write a small valid raw parquet file for offline integration tests."""
    rows: list[dict[str, object]] = []
    symbols = ["AAPL", "JPM", "TSLA", "SPY"]

    demo_days = pd.bdate_range("2020-01-02", periods=35)
    for day_index, day in enumerate(demo_days):
        for symbol_index, symbol in enumerate(symbols):
            base_price = 100.0 + 10.0 * symbol_index
            day_shift = 0.3 * float(day_index)
            day_text = day.strftime("%Y-%m-%d")
            rows.extend(
                [
                    {
                        "symbol": symbol,
                        "ts": f"{day_text}T14:30:00+00:00",
                        "close": base_price + day_shift,
                    },
                    {
                        "symbol": symbol,
                        "ts": f"{day_text}T14:31:00+00:00",
                        "close": base_price + day_shift + 0.4,
                    },
                    {
                        "symbol": symbol,
                        "ts": f"{day_text}T14:32:00+00:00",
                        "close": base_price + day_shift + 0.8,
                    },
                ]
            )

    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(raw_dir / "minute_bars.parquet", index=False)


def _write_test_config(config_path: Path) -> None:
    """Write a compact config file that uses local raw parquet only."""
    config_path.write_text(
        "\n".join(
            [
                "# Runtime settings.",
                "[run]",
                'start_date = "2020-01-01"',
                'end_date = "2020-03-31"',
                'output_dir = "outputs"',
                "seed = 7",
                "",
                "# Raw data input settings.",
                "[raw_data]",
                'raw_dir = "data/raw"',
                'file_pattern = "*.parquet"',
                'timezone = "UTC"',
                "regular_session_only = true",
                "",
                "# Portfolio symbols.",
                "[portfolio]",
                'symbols = ["AAPL", "JPM", "TSLA", "SPY"]',
                "",
                "# Backtest settings.",
                "[backtest]",
                "calibration_window = 20",
                "alphas = [0.05]",
                "",
                "# Evaluation settings.",
                "[evaluation]",
                "es_backtest_confidence = 0.95",
                "",
                "[[evaluation.periods]]",
                'name = "full"',
                'start_date = "2020-01-01"',
                'end_date = "2020-03-31"',
                "",
                "# Feature settings.",
                "[features]",
                "include_realized_return_volatility_5d = false",
                "",
                "# Historical simulation settings.",
                "[historical]",
                "window = 20",
                "",
                "# GARCH settings.",
                "[garch]",
                "p = 1",
                "q = 1",
                "",
                "# Filtered historical simulation settings.",
                "[filtered_historical]",
                "simulations = 100",
                "",
                "# Adaptive conformal settings.",
                "[conformal]",
                "window = 20",
                "mean_window = 5",
                "learning_rate = 0.02",
            ]
        ),
        encoding="utf-8",
    )


def test_pipeline_runs_from_raw_parquet_and_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    """A full offline run should start from data/raw and write parquet outputs."""
    config_path = tmp_path / "config.toml"
    _write_test_config(config_path=config_path)
    _write_toy_raw_cache(raw_dir=tmp_path / "data" / "raw")

    artifacts = run_pipeline(config_path=config_path)

    assert artifacts.daily_returns_path.exists()
    assert artifacts.realized_variance_path.exists()
    assert artifacts.feature_table_path.exists()
    assert artifacts.backtest_results_path.exists()
    assert artifacts.metrics_path.exists()

    feature_table = pd.read_parquet(artifacts.feature_table_path)
    backtest_results = pd.read_parquet(artifacts.backtest_results_path)
    metrics = pd.read_parquet(artifacts.metrics_path)

    assert not feature_table.empty
    assert not backtest_results.empty
    assert not metrics.empty
    assert "rv_lag_1" in feature_table.columns
    assert "predicted_var" in backtest_results.columns
    assert "avg_quantile_loss" in metrics.columns

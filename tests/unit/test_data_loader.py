"""Regression tests for offline raw-parquet preprocessing steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformal_var_risk.data.daily_panel import build_daily_log_return_panel
from conformal_var_risk.data.raw_cache import load_raw_minute_bars
from conformal_var_risk.data.realized_variance import (
    build_daily_realized_variance_panel,
)


def _toy_minute_frame() -> pd.DataFrame:
    """Build a tiny two-day minute sample for hand-checkable assertions."""
    rows = [
        {"symbol": "AAPL", "ts": "2020-01-02T14:30:00+00:00", "close": 100.0},
        {"symbol": "AAPL", "ts": "2020-01-02T14:31:00+00:00", "close": 101.0},
        {"symbol": "AAPL", "ts": "2020-01-02T14:32:00+00:00", "close": 102.0},
        {"symbol": "AAPL", "ts": "2020-01-03T14:30:00+00:00", "close": 103.0},
        {"symbol": "AAPL", "ts": "2020-01-03T14:31:00+00:00", "close": 104.0},
        {"symbol": "AAPL", "ts": "2020-01-03T14:32:00+00:00", "close": 105.0},
        {"symbol": "JPM", "ts": "2020-01-02T14:30:00+00:00", "close": 50.0},
        {"symbol": "JPM", "ts": "2020-01-02T14:31:00+00:00", "close": 50.5},
        {"symbol": "JPM", "ts": "2020-01-02T14:32:00+00:00", "close": 51.0},
        {"symbol": "JPM", "ts": "2020-01-03T14:30:00+00:00", "close": 51.5},
        {"symbol": "JPM", "ts": "2020-01-03T14:31:00+00:00", "close": 52.0},
        {"symbol": "JPM", "ts": "2020-01-03T14:32:00+00:00", "close": 52.5},
    ]
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def test_missing_raw_directory_raises_clear_error(tmp_path: Path) -> None:
    """The loader should fail loudly when `data/raw/` does not exist."""
    missing_dir = tmp_path / "raw"
    with pytest.raises(FileNotFoundError) as exception_info:
        _ = load_raw_minute_bars(
            raw_dir=missing_dir,
            file_pattern="*.parquet",
            required_symbols=["AAPL", "JPM"],
            start_date="2020-01-01",
            end_date="2020-01-31",
        )

    assert "data/raw" in str(exception_info.value) or "raw" in str(exception_info.value)


def test_raw_contract_requires_symbol_ts_close_columns(tmp_path: Path) -> None:
    """The loader should reject parquet files with missing required columns."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    bad_frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": [pd.Timestamp("2020-01-02T14:30:00+00:00")],
            "open": [100.0],
        }
    )
    bad_frame.to_parquet(raw_dir / "minute_bars.parquet", index=False)

    with pytest.raises(ValueError) as exception_info:
        _ = load_raw_minute_bars(
            raw_dir=raw_dir,
            file_pattern="*.parquet",
            required_symbols=["AAPL"],
            start_date="2020-01-01",
            end_date="2020-01-31",
        )

    assert "close" in str(exception_info.value)


def test_daily_log_return_panel_matches_toy_close_to_close_math() -> None:
    """Daily panel should use regular-session last close per symbol and log returns."""
    minute_frame = _toy_minute_frame()

    daily_returns = build_daily_log_return_panel(
        minute_bars=minute_frame,
        symbols=["AAPL", "JPM"],
    )

    expected_aapl = np.log(105.0 / 102.0)
    expected_jpm = np.log(52.5 / 51.0)
    expected_portfolio = 0.5 * (expected_aapl + expected_jpm)

    assert len(daily_returns) == 1
    assert daily_returns.index[0] == pd.Timestamp("2020-01-03")
    assert daily_returns.loc[pd.Timestamp("2020-01-03"), "AAPL"] == pytest.approx(
        expected_aapl
    )
    assert daily_returns.loc[pd.Timestamp("2020-01-03"), "JPM"] == pytest.approx(
        expected_jpm
    )
    assert daily_returns.loc[pd.Timestamp("2020-01-03"), "portfolio"] == pytest.approx(
        expected_portfolio
    )


def test_realized_variance_matches_hand_computation() -> None:
    """Daily realized variance should equal sum of squared intraday log returns."""
    minute_frame = _toy_minute_frame()

    realized_variance = build_daily_realized_variance_panel(
        minute_bars=minute_frame,
        symbols=["AAPL", "JPM"],
    )

    aapl_intraday_r1 = np.log(101.0 / 100.0)
    aapl_intraday_r2 = np.log(102.0 / 101.0)
    expected_aapl_day1 = aapl_intraday_r1**2 + aapl_intraday_r2**2

    jpm_intraday_r1 = np.log(50.5 / 50.0)
    jpm_intraday_r2 = np.log(51.0 / 50.5)
    expected_jpm_day1 = jpm_intraday_r1**2 + jpm_intraday_r2**2

    assert realized_variance.loc[pd.Timestamp("2020-01-02"), "AAPL"] == pytest.approx(
        expected_aapl_day1
    )
    assert realized_variance.loc[pd.Timestamp("2020-01-02"), "JPM"] == pytest.approx(
        expected_jpm_day1
    )
    assert realized_variance.loc[
        pd.Timestamp("2020-01-02"), "portfolio"
    ] == pytest.approx(0.5 * (expected_aapl_day1 + expected_jpm_day1))

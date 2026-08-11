"""Regression tests for feature-table timing and alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conformal_var_risk.features.feature_builder import build_feature_table


def test_feature_table_uses_strict_lagged_information() -> None:
    """Feature row at date t must only depend on information through date t."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.02, 0.03, len(dates)),
        },
        index=dates,
    )
    rv = pd.DataFrame(
        {
            "AAPL": np.linspace(0.0001, 0.0010, len(dates)),
        },
        index=dates,
    )

    feature_table = build_feature_table(
        daily_returns=returns,
        realized_variance=rv,
        include_realized_return_volatility_5d=False,
    )

    first_row = feature_table.iloc[0]
    feature_date = pd.Timestamp(first_row["date"])

    expected_rv_lag_1 = rv.loc[feature_date - pd.offsets.BDay(1), "AAPL"]
    expected_target = returns.loc[feature_date + pd.offsets.BDay(1), "AAPL"]

    assert first_row["rv_lag_1"] == expected_rv_lag_1
    assert first_row["target_next_day_return"] == expected_target


def test_feature_table_dropna_is_column_scoped_not_global() -> None:
    """Rows should not be dropped for irrelevant all-NaN extra columns."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.01, 0.02, len(dates)),
        },
        index=dates,
    )
    rv = pd.DataFrame(
        {
            "AAPL": np.linspace(0.0002, 0.0011, len(dates)),
            "UNUSED": np.nan,
        },
        index=dates,
    )

    feature_table = build_feature_table(
        daily_returns=returns,
        realized_variance=rv,
        include_realized_return_volatility_5d=False,
    )

    assert not feature_table.empty
    assert set(feature_table["asset"]) == {"AAPL"}

"""Feature-table construction for daily tail-risk forecasting.

All features are timed so a feature row at date `t` only uses information that
is available up to and including date `t`, and is used to forecast date `t+1`.
"""

from __future__ import annotations

import pandas as pd


def build_feature_table(
    daily_returns: pd.DataFrame,
    realized_variance: pd.DataFrame,
    include_realized_return_volatility_5d: bool,
) -> pd.DataFrame:
    """Build a long-form feature table with strict no-lookahead timing.

    Parameters
    ----------
    daily_returns
        Wide daily log-return panel indexed by date. Columns are assets.
    realized_variance
        Wide daily realized-variance panel indexed by date. Columns are assets.
    include_realized_return_volatility_5d
        Whether to include a 5-day rolling realized return volatility feature.

    Returns
    -------
    pd.DataFrame
        Long-form table with one row per date-asset pair. Includes target
        next-day return, current-day realized variance, and lagged features.
    """
    if not daily_returns.index.equals(realized_variance.index):
        raise ValueError(
            "Feature build requires aligned date index for returns and RV."
        )

    shared_assets = daily_returns.columns.intersection(realized_variance.columns)

    feature_rows: list[pd.DataFrame] = []
    for asset_name in shared_assets:
        asset_returns = daily_returns[asset_name].copy()
        asset_realized_variance = realized_variance[asset_name].copy()

        # Target at date t is next-day return R_{t+1}.
        target_next_day_return = asset_returns.shift(-1)

        asset_frame = pd.DataFrame(
            {
                "date": daily_returns.index,
                "asset": asset_name,
                "target_next_day_return": target_next_day_return,
                "rv_t": asset_realized_variance,
                "rv_lag_1": asset_realized_variance.shift(1),
                "rv_mean_5": asset_realized_variance.shift(1).rolling(5).mean(),
                "rv_mean_21": asset_realized_variance.shift(1).rolling(21).mean(),
                "abs_return_lag_1": asset_returns.abs().shift(1),
            }
        )
        if include_realized_return_volatility_5d:
            asset_frame["return_volatility_5"] = asset_returns.shift(1).rolling(5).std()

        feature_rows.append(asset_frame)

    feature_table = pd.concat(feature_rows, ignore_index=True)

    required_feature_columns = [
        "rv_lag_1",
        "rv_mean_5",
        "rv_mean_21",
        "abs_return_lag_1",
        "target_next_day_return",
    ]
    if include_realized_return_volatility_5d:
        required_feature_columns.append("return_volatility_5")

    # Drop rows where required model inputs are unavailable, but do it only on
    # required feature columns to avoid accidental global dropna behavior.
    feature_table = feature_table.dropna(subset=required_feature_columns).reset_index(
        drop=True
    )
    return feature_table

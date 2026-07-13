"""Daily realized variance construction from intraday log returns.

Realized variance for day `t` is defined as:

`RV_t = sum_i r_{t,i}^2`

where `r_{t,i}` is intraday log return for interval `i` on day `t`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_var_risk.data.daily_panel import (
    _assign_new_york_trading_date,
    _filter_regular_session,
)


def build_daily_realized_variance_panel(
    minute_bars: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """Build daily realized variance panel per symbol and portfolio.

    Parameters
    ----------
    minute_bars
        Long-form raw frame with columns `symbol`, `ts`, and `close`.
    symbols
        Ordered output symbols.

    Returns
    -------
    pd.DataFrame
        Daily realized variance panel indexed by date. Columns are symbols plus
        equal-weight `portfolio` realized variance.
    """
    regular_session = _filter_regular_session(minute_bars=minute_bars)
    working = _assign_new_york_trading_date(
        minute_bars=regular_session.sort_values(["symbol", "ts"])
    )

    # Sum squared intraday log returns within each symbol-day.
    working["intraday_log_return"] = working.groupby(["symbol", "date"], sort=False)[
        "close"
    ].transform(lambda close_series: np.log(close_series / close_series.shift(1)))
    working["squared_intraday_log_return"] = working["intraday_log_return"] ** 2

    realized_variance_long = (
        working.groupby(["date", "symbol"], as_index=False)[
            "squared_intraday_log_return"
        ]
        .sum(min_count=1)
        .rename(columns={"squared_intraday_log_return": "rv"})
    )
    realized_variance_panel = (
        realized_variance_long.pivot(index="date", columns="symbol", values="rv")
        .reindex(columns=symbols)
        .sort_index()
    )
    realized_variance_panel.index.name = "date"

    # Portfolio realized variance includes cross-asset covariance. Averaging
    # constituent variances would omit covariance and apply the wrong weights.
    intraday_return_panel = working.pivot_table(
        index=["date", "ts"],
        columns="symbol",
        values="intraday_log_return",
        aggfunc="last",
    ).reindex(columns=symbols)
    complete_constituent_rows = intraday_return_panel.notna().all(axis=1)
    equal_weight_intraday_return = intraday_return_panel.mean(axis=1).where(
        complete_constituent_rows
    )
    portfolio_realized_variance = (
        equal_weight_intraday_return.pow(2).groupby(level="date").sum(min_count=1)
    )

    realized_variance_panel["portfolio"] = portfolio_realized_variance.reindex(
        realized_variance_panel.index
    )
    return realized_variance_panel

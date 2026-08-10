"""Daily close-to-close return panel construction from minute bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16


def _assign_new_york_trading_date(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Add a calendar ``date`` column from UTC timestamps in New York session time."""
    dated = minute_bars.copy()
    dated["date"] = (
        dated["ts"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return dated


def build_daily_log_return_panel(
    minute_bars: pd.DataFrame,
    symbols: list[str],
    regular_session_only: bool = True,
) -> pd.DataFrame:
    """Build daily close-to-close log returns from raw minute bars.

    Parameters
    ----------
    minute_bars
        Long-form raw frame with columns `symbol`, `ts`, and `close`. The
        `ts` column must be timezone-aware timestamps.
    symbols
        Ordered symbol list for output columns.
    regular_session_only
        Whether to drop bars outside the 09:30-16:00 New York session before
        picking each daily close.

    Returns
    -------
    pd.DataFrame
        Daily log-return panel indexed by date. Columns are configured symbols
        plus the equal-weight `portfolio` series.
    """
    session_bars = (
        _filter_regular_session(minute_bars=minute_bars)
        if regular_session_only
        else minute_bars
    )
    dated_bars = _assign_new_york_trading_date(minute_bars=session_bars)

    # One close per symbol-day: last print of the retained session, then wide panel.
    daily_close = (
        dated_bars.sort_values(["symbol", "ts"])
        .groupby(["date", "symbol"], as_index=False)["close"]
        .last()
        .pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )

    ordered_close = daily_close.reindex(columns=symbols)
    daily_log_returns = np.log(ordered_close / ordered_close.shift(1))
    daily_log_returns = daily_log_returns.dropna(how="all")
    daily_log_returns.index.name = "date"

    return _append_portfolio_column(
        panel=daily_log_returns,
        constituent_symbols=symbols,
    )


def _filter_regular_session(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Restrict bars to regular US equity session in New York time."""
    new_york_times = minute_bars["ts"].dt.tz_convert("America/New_York")
    hours = new_york_times.dt.hour
    minutes = new_york_times.dt.minute

    after_open = (hours > MARKET_OPEN_HOUR) | (
        (hours == MARKET_OPEN_HOUR) & (minutes >= MARKET_OPEN_MINUTE)
    )
    before_close = hours < MARKET_CLOSE_HOUR
    exactly_close = (hours == MARKET_CLOSE_HOUR) & (minutes == 0)

    return minute_bars.loc[after_open & (before_close | exactly_close)].copy()


def _append_portfolio_column(
    panel: pd.DataFrame,
    constituent_symbols: list[str],
) -> pd.DataFrame:
    """Append equal-weight portfolio only where all constituents exist."""
    augmented = panel.copy()
    constituent_returns = augmented.reindex(columns=constituent_symbols)
    complete_rows = constituent_returns.notna().all(axis=1)
    augmented["portfolio"] = constituent_returns.mean(axis=1).where(complete_rows)
    return augmented

"""Raw parquet ingestion utilities for offline preprocessing.

The default runtime path uses only tracked parquet files in `data/raw/`.
This module validates the raw-cache contract before any model code runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_RAW_COLUMNS = ["symbol", "ts", "close"]


def load_raw_minute_bars(
    raw_dir: Path,
    file_pattern: str,
    required_symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load and validate raw minute bars from local parquet files.

    Parameters
    ----------
    raw_dir
        Directory that stores tracked raw parquet files.
    file_pattern
        Pattern used to discover one file or a small number of parquet parts.
    required_symbols
        Symbols that must be present in the raw data contract.
    start_date
        Inclusive lower date bound in `YYYY-MM-DD` format.
    end_date
        Inclusive upper date bound in `YYYY-MM-DD` format.

    Returns
    -------
    pd.DataFrame
        Long-form minute-bar frame with columns `symbol`, `ts`, and `close`.

    Raises
    ------
    FileNotFoundError
        If the raw directory or parquet files are missing.
    ValueError
        If required columns, symbols, or timestamp conventions are violated.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            "Missing required raw directory. Expected tracked offline input at "
            f"{raw_dir}. Create/populate data/raw before running the pipeline."
        )

    parquet_paths = sorted(raw_dir.glob(file_pattern))
    if not parquet_paths:
        raise FileNotFoundError(
            "No raw parquet files found in data/raw. Expected at least one file "
            f"matching pattern {file_pattern!r} under {raw_dir}."
        )

    frames = [pd.read_parquet(path) for path in parquet_paths]
    raw_frame = pd.concat(frames, ignore_index=True)

    # Enforce the offline raw-cache contract before any preprocessing runs.
    missing_columns = [
        column_name
        for column_name in REQUIRED_RAW_COLUMNS
        if column_name not in raw_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Raw parquet contract violation: missing required columns "
            f"{missing_columns}. Required columns are {REQUIRED_RAW_COLUMNS}."
        )

    raw_frame = raw_frame[REQUIRED_RAW_COLUMNS].copy()
    raw_frame["ts"] = pd.to_datetime(raw_frame["ts"], utc=False, errors="coerce")
    if raw_frame["ts"].isna().any():
        raise ValueError("Raw parquet contract violation: invalid timestamps in `ts`.")

    if raw_frame["ts"].dt.tz is None:
        raise ValueError(
            "Raw parquet contract violation: `ts` must be timezone-aware UTC timestamps."
        )

    observed_symbols = set(raw_frame["symbol"].unique())
    missing_symbols = sorted(set(required_symbols) - observed_symbols)
    if missing_symbols:
        raise ValueError(
            "Raw parquet contract violation: missing required symbols "
            f"{missing_symbols}."
        )

    # Inclusive calendar bounds in UTC, with end-of-day on the configured end date.
    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_timestamp = (
        pd.Timestamp(end_date, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    bounded = raw_frame.loc[
        (raw_frame["ts"] >= start_timestamp) & (raw_frame["ts"] <= end_timestamp)
    ].copy()
    if bounded.empty:
        raise ValueError(
            "Raw parquet contract violation: no rows remain after applying "
            f"date range [{start_date}, {end_date}]."
        )

    bounded = bounded.sort_values(["symbol", "ts"]).reset_index(drop=True)
    for symbol in required_symbols:
        symbol_slice = bounded.loc[bounded["symbol"] == symbol, "ts"]
        if not symbol_slice.is_monotonic_increasing:
            raise ValueError(
                "Raw parquet contract violation: timestamps are not sorted for "
                f"symbol {symbol}."
            )

    return bounded

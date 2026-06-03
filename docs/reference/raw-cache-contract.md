# Raw Cache Contract

This document defines the required offline raw-data contract for this repository.

## Purpose

The default runtime path starts from tracked raw parquet files in `data/raw/`.
No database access is required after those files are present.

The canonical pipeline order is:

1. raw parquet minute bars
2. regular-session filtering and daily close construction
3. daily log-return panel construction
4. realized variance construction
5. feature table construction
6. model training and backtest

`data/raw/` is the only tracked raw-data home. Runtime preprocessing starts from
those files only.

## Required Folder

- required folder: `data/raw/`

## Allowed Files

One of the following layouts is supported:

- one file: `data/raw/minute_bars.parquet`
- or a small number of parts: `data/raw/minute_bars_*.parquet`

No other raw input location is part of the default contract.

## Required Columns

Every parquet part must contain exactly these required columns:

- `symbol` (string ticker)
- `ts` (timestamp)
- `close` (numeric close price)

Additional columns are allowed but discouraged for portability.

## Timestamp Convention

- `ts` must be timezone-aware and represent UTC timestamps.
- The pipeline converts `ts` into `America/New_York` for session filtering.

## Session Convention

- Rows may include full-session minute bars.
- The runtime pipeline always applies regular-session filtering internally
  (`09:30` to `16:00` New York time, inclusive of the close print).

## Bounded Demo Scope

The tracked demo dataset is bounded to keep clone-and-run simple:

- symbols: `AAPL`, `JPM`, `TSLA`, `SPY`
- derived series: equal-weight `portfolio` (constructed in code)
- date span: `2019-01-01` through `2023-12-31`

## Required Manifest Fields

`data/raw/README.md` must declare:

- file list
- symbols
- row counts
- first timestamp
- last timestamp
- source note (for example, whether a one-time ClickHouse export was used)

## Validation Rules Used By The Loader

The loader fails early with actionable messages when:

- `data/raw/` is missing
- no parquet files match the configured raw-file pattern
- required columns are missing
- configured symbols are missing
- timestamps are missing timezone information
- timestamps are not sorted within each symbol

These checks protect offline reproducibility and interview-defensible results.

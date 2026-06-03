# Raw Demo Dataset Manifest

This folder stores the tracked offline raw input for the default runtime path.

## Files

- `minute_bars.parquet`

## Dataset Scope

- symbols: `AAPL`, `JPM`, `TSLA`, `SPY`
- date span: `2019-01-01` to `2023-12-31` (business days)
- timestamp timezone: UTC in column `ts`
- fields: `symbol`, `ts`, `close`

## Size And Row Count

- total rows: 2,034,240
- compressed parquet size: see `du -sh data/raw`

## Source Note

This bounded demo cache was generated once for offline portability and tracked
in git. The project runtime does not require any database access once these
files are present.

# ClickHouse Raw-Cache Refresh (Optional)

This project runs offline by default from tracked parquet files in `data/raw/`.
ClickHouse is optional and only used when you intentionally refresh the raw
cache.

## When You Need This

Use this workflow only when you want to rebuild `data/raw/minute_bars.parquet`
from a trusted upstream database snapshot.

## Required Output Contract

Your export must keep only these columns:

- `symbol`
- `ts`
- `close`

Keep the bounded study universe:

- symbols: `AAPL`, `JPM`, `TSLA`, `SPY`
- date range: `2019-01-01` through `2023-12-31`

## One-Time Refresh Steps

1. Connect to ClickHouse from a machine with credentials.
2. Query minute bars for the bounded symbols and date range.
3. Export parquet with only `symbol`, `ts`, and `close`.
4. Write output to `data/raw/minute_bars.parquet`.
5. Update `data/raw/README.md` with row count, date bounds, and source note.
6. Run the offline pipeline locally to verify compatibility.

## Important Rule

Do not make ClickHouse part of the default runtime path. The default path must
remain local-parquet-only for portability.

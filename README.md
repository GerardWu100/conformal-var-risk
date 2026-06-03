# Conformal VaR Risk

This project is an offline-first quantitative risk research pipeline for
one-day-ahead lower-tail Value-at-Risk, abbreviated VaR, with Expected
Shortfall, abbreviated ES, as a secondary diagnostic.

The repository is intentionally scoped as a resume project: small, linear, and
interview-defensible. The full runtime starts from tracked raw parquet files,
builds daily returns and realized variance, constructs a compact feature table,
runs a model comparison backtest, and finishes with a teaching notebook.

## Why This Is Interview-Defensible

- It starts from transparent raw minute bars, not hidden preprocessed tensors.
- It makes data timing explicit to avoid lookahead bias.
- It compares conformal and benchmark models on the same walk-forward protocol.
- It keeps the model set narrow and explainable.

## Offline Input Contract

Required raw input lives in `data/raw/` only.

- required columns: `symbol`, `ts`, `close`
- bounded symbols: `AAPL`, `JPM`, `TSLA`, `SPY`
- bounded date span: `2019-01-01` through `2023-12-31`

See `docs/reference/raw-cache-contract.md` for full details.

## Run

1. `uv sync`
2. `uv run python -m pytest`
3. `uv run python scripts/run_pipeline.py --config config.toml`
4. `uv run python -m nbconvert --to notebook --execute --inplace notebooks/demo.ipynb`

## Outputs

The pipeline writes derived parquet artifacts under `outputs/runs/`:

- `outputs/runs/daily_returns.parquet`
- `outputs/runs/daily_realized_variance.parquet`
- `outputs/runs/feature_table.parquet`
- `outputs/runs/backtest_results.parquet`
- `outputs/runs/summary_metrics.parquet`

## Optional ClickHouse Refresh

ClickHouse is not required for normal runs. It is only an optional one-time
refresh path when you want to regenerate `data/raw/` from a database source.

See `docs/reference/clickhouse-refresh.md`.

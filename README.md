# Conformal VaR Risk

Offline-first research pipeline that compares an adaptive conformal
one-day-ahead lower-tail Value-at-Risk (VaR) model against standard
benchmarks on a small equity and portfolio panel. It is a compact,
interview-defensible risk-model validation project, not a production trading
system.

## What it does

- Reads tracked raw minute-bar parquet from `data/raw/` (no database needed
  by default).
- Builds a daily close-to-close log-return panel and daily realized
  variance, including cross-asset covariance for the equal-weight portfolio
  series.
- Builds a compact, no-lookahead daily feature table.
- Runs a rolling walk-forward backtest comparing five models: Historical
  Simulation, GARCH(1,1) with normal innovations, GARCH(1,1) with Student-$t$
  innovations, Filtered Historical Simulation, and an adaptive conformal
  lower-tail model.
- Scores each model with violation rate, coverage rate, quantile (pinball)
  loss, Christoffersen coverage tests, and exact Clopper-Pearson breach-rate
  intervals.
- Universe: `AAPL`, `JPM`, `TSLA`, `SPY`, plus a derived equal-weight
  `portfolio` series, over `2019-01-01` to `2023-12-31`.

The conformal model builds one-sided downside scores around a rolling mean
and picks the sorted score at rank $\min(n, \lceil (n+1)(1-a_t) \rceil)$,
where $n$ is the number of calibration scores and $a_t$ is an internal tail
probability that is lowered after each breach (making the next boundary more
conservative). This is the standard finite-sample split-conformal
order-statistic rule, but rolling equity returns are not exchangeable, so the
code treats conformal theory as a construction method, not a coverage
guarantee — coverage is instead measured empirically in the walk-forward
backtest. See `docs/user/project-brief.md` for the full methodology note.

## Requirements

- Python 3.13
- `uv`
- Tracked raw parquet in `data/raw/` (included in this repo)

No external service is required for the default run. ClickHouse is only used
for an optional, manual one-time refresh of `data/raw/`; the repository code
does not read the `CLICKHOUSE_*` variables in `.env.example` itself — that
connection is made by hand when regenerating the raw cache. See
`docs/reference/clickhouse-refresh.md`.

## Setup

```
uv sync
```

## Usage

- `uv run python -m pytest` — run the unit test suite.
- `uv run python scripts/run_pipeline.py --config config.toml` — run the full
  pipeline: build the panel, features, and backtest, and write outputs.
- `uv run python -m nbconvert --to notebook --execute --inplace notebooks/demo.ipynb`
  — execute the teaching notebook end to end.

## Configuration

Runtime settings live in `config.toml`:

- `[run]` — date range, output directory, random seed.
- `[raw_data]` — raw folder, file pattern, timezone, `regular_session_only`
  session filter.
- `[portfolio]` — the symbol universe.
- `[backtest]` — `calibration_window` (trailing returns per model fit) and
  left-tail `alphas`.
- `[evaluation]` — ES backtest confidence level and named evaluation periods
  (`full`, `covid`, `rate_shock`, `post_covid`).
- `[historical]`, `[garch]`, `[filtered_historical]`, `[conformal]` —
  per-model settings (windows, simulation counts, learning rate).

With the default 1,150-return calibration window, only 153 out-of-sample
forecasts per series fall in 2023, so the configured `covid` and
`rate_shock` evaluation periods have no rows under this setting.

## Layout

- `src/conformal_var_risk/` — the package: data loading, features, models,
  evaluation, pipeline orchestration.
- `scripts/run_pipeline.py` — thin CLI entry point.
- `data/raw/` — tracked raw minute-bar parquet input.
- `outputs/runs/` — generated parquet artifacts (gitignored).
- `notebooks/demo.ipynb` — teaching walkthrough of the full pipeline.
- `tests/unit/` — unit tests.
- `docs/reference/` — raw-data contract, offline workflow, and ClickHouse
  refresh notes.
- `docs/user/project-brief.md` — methodology and design-decision note.

## Output

The pipeline writes derived parquet artifacts to `outputs/runs/`:

- `daily_returns.parquet`
- `daily_realized_variance.parquet` — squared decimal-return units per day,
  not annualized.
- `feature_table.parquet`
- `backtest_results.parquet`
- `summary_metrics.parquet`

## License

All rights reserved. See [LICENSE](LICENSE).

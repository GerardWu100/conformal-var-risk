# Offline Workflow

This reference describes the default clone-and-run path.

## Requirements

- Python 3.13
- `uv`
- tracked raw parquet files in `data/raw/`

No database access is required for the standard workflow.

## Setup

1. Install dependencies:
   - `uv sync`
2. Confirm raw files exist:
   - `ls data/raw`

## Run Commands

1. Run tests:
   - `uv run python -m pytest`
2. Run pipeline:
   - `uv run python scripts/run_pipeline.py --config config.toml`
3. Execute teaching notebook:
   - `uv run python -m nbconvert --to notebook --execute --inplace notebooks/demo.ipynb`

## Output Artifacts

The pipeline writes derived parquet outputs under `outputs/runs/`:

- `daily_returns.parquet`
- `daily_realized_variance.parquet`
- `feature_table.parquet`
- `backtest_results.parquet`
- `summary_metrics.parquet`

These are derived artifacts and can be rebuilt from `data/raw/`.

## Common Failure Modes

- `data/raw/` missing:
  - restore tracked raw parquet files
- missing required raw columns:
  - ensure raw files contain `symbol`, `ts`, `close`
- symbol coverage mismatch:
  - ensure all symbols from `config.toml` are present in raw data
- notebook execution error after code changes:
  - rerun pipeline first to regenerate derived outputs

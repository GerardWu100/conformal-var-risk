# GUIDE_src.md -- Source Package Guide

## Package Purpose

`src/conformal_var_risk/` contains the full offline runtime implementation.

## Subfolder Responsibilities

- `config.py`
  - typed TOML schema and loader
- `data/`
  - raw parquet contract checks
  - daily return panel construction
  - realized variance construction
- `features/`
  - no-lookahead feature-table construction
- `models/`
  - benchmark and conformal VaR models
- `evaluation/`
  - rolling backtest and summary metrics
- `pipeline/`
  - orchestration and artifact writing

## Runtime Contract

The package assumes:

- raw tracked input exists under `data/raw/`
- required raw columns are `symbol`, `ts`, `close`
- derived artifacts are written under `outputs/runs/`

## Data Objects

- daily return panel: wide DataFrame, index=`date`, columns=assets + `portfolio`
- realized variance panel: same shape as return panel
- feature table: long DataFrame with date, asset, target, and feature columns
- backtest results: long DataFrame with per-model daily forecasts and outcomes
- summary metrics: grouped DataFrame with period-level diagnostics

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

## Model and Evaluation Logic

For a target lower-tail probability $\alpha$, every model forecasts a return
quantile $q_{t,\alpha}$. A violation occurs when the realized return is below
that quantile. Reported VaR is the nonnegative loss
$\max(-q_{t,\alpha},0)$, but coverage and quantile loss always use the raw
quantile so scoring and adaptive updates cannot disagree.

The conformal model builds downside scores
$s_u=\max(\hat{\mu}_u-r_u,0)$ around a rolling mean $\hat{\mu}_u$. For $n$
calibration scores and internal tail level $a_t$, it selects sorted score rank
$\min(n,\lceil(n+1)(1-a_t)\rceil)$. The adaptive update lowers $a_t$ after a
breach, selecting a larger score and a more conservative next boundary.

Summary metrics include violation frequency, pinball loss, Christoffersen
coverage tests, and exact 95% Clopper-Pearson intervals. The exact interval is
valid for one Bernoulli forecast series under its sampling assumptions; do not
apply it to pooled assets as though correlated violations were independent.

## Short Journal

- 2026-07-13: Preserved raw quantiles for scoring, added finite-sample conformal rank selection, and made failed rolling GARCH fits clear stale state before using the fallback forecast.
- 2026-07-13: ES backtest ratios now return missing diagnostics when predicted ES is zero instead of propagating infinities.

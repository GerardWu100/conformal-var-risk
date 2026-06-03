# GUIDE_ROOT.md -- Root Navigation Guide

## Purpose

This repository is a compact offline-first research pipeline for one problem:
compare one-day-ahead lower-tail VaR models on a shared return panel built from
tracked raw minute-bar parquet files.

The root stays intentionally thin:

- one runtime config file (`config.toml`)
- one CLI entrypoint under `scripts/`
- one package under `src/conformal_var_risk/`
- unit tests under `tests/unit/`
- one teaching notebook

## Workflow

The runtime order is:

1. Read raw minute bars from `data/raw/`.
2. Build daily close-to-close log returns.
3. Build daily realized variance.
4. Build a compact daily feature table.
5. Run rolling model backtests.
6. Write summary metrics and derived parquet artifacts.
7. Teach the full flow in `notebooks/demo.ipynb`.

## Root Files

- `README.md`
  - project story, offline contract, and run commands
- `config.toml`
  - dataset bounds, preprocessing assumptions, model settings
- `scripts/run_pipeline.py`
  - thin CLI shell for the package pipeline
- `pyproject.toml`
  - dependency and packaging metadata
- `GUIDE_OVERVIEW.md`
  - concise architecture map

## Important Rule

Default execution must succeed offline when `data/raw/` is present.
ClickHouse is optional and used only for one-time raw-cache refresh.

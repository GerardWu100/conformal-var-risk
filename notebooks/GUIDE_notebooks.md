# GUIDE_notebooks.md -- Notebook Guide

## Purpose

`demo.ipynb` is the main teaching artifact for this repository.

Unlike the previous dashboard-reader notebook style, this notebook now teaches
the full offline research pipeline from local raw parquet input to model
evaluation interpretation.

## Execution Rule

The notebook is designed to run top-to-bottom with no hidden state using:

- `uv run python -m nbconvert --to notebook --execute --inplace notebooks/demo.ipynb`

## Required Teaching Sections

1. project framing and offline contract
2. raw parquet inspection
3. regular-session filtering and daily close construction
4. daily log-return construction
5. realized variance construction
6. feature building
7. training and backtest setup
8. model execution
9. evaluation metrics
10. interpretation and interview narrative

## Structure Rule

Notebook cells should alternate Markdown and code throughout so theory and
implementation stay aligned step by step.

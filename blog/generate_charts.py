"""Generate the evidence charts used by the conformal VaR blog post.

The default mode reads frozen comma-separated value (CSV) files under
``blog/data``. Pass ``--refresh-data`` after running the project pipeline to
rebuild those frozen inputs from ``outputs/runs`` before drawing the figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOG_ROOT = Path(__file__).resolve().parent
DATA_DIR = BLOG_ROOT / "data"
IMAGE_DIR = BLOG_ROOT / "images"
PIPELINE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs"
MODEL_ORDER = [
    "Historical",
    "GARCH normal",
    "GARCH Student-t",
    "Filtered historical",
    "Adaptive conformal",
]
MODEL_LABELS = {
    "historical": "Historical",
    "garch_normal": "GARCH normal",
    "garch_student_t": "GARCH Student-t",
    "filtered_historical": "Filtered historical",
    "conformal": "Adaptive conformal",
}
TAIL_COLORS = {0.05: "#16a6a1", 0.01: "#d85a54"}


def refresh_frozen_data() -> None:
    """Create compact, post-specific CSV inputs from pipeline parquet results.

    Returns
    -------
    None
        Writes an aggregate model summary and a dated SPY forecast path under
        ``blog/data``.
    """
    metrics = pd.read_parquet(PIPELINE_OUTPUT_DIR / "summary_metrics.parquet")
    backtest = pd.read_parquet(PIPELINE_OUTPUT_DIR / "backtest_results.parquet")

    full_sample = metrics.loc[metrics["period"] == "full"].copy()
    aggregate = (
        full_sample.groupby(["model", "alpha"], as_index=False)
        .agg(
            observations=("observations", "sum"),
            violations=("violations", "sum"),
            violation_rate=("violation_rate", "mean"),
            avg_quantile_loss=("avg_quantile_loss", "mean"),
            avg_predicted_var=("avg_predicted_var", "mean"),
            avg_predicted_es=("avg_predicted_es", "mean"),
            assets_passing_uc_5pct=("uc_p_value", lambda values: int((values >= 0.05).sum())),
            assets_passing_cc_5pct=("cc_p_value", lambda values: int((values >= 0.05).sum())),
        )
        .sort_values(["alpha", "model"])
    )
    aggregate.to_csv(DATA_DIR / "model_summary.csv", index=False)

    spy_path = backtest.loc[
        (backtest["asset"] == "SPY")
        & (backtest["model"] == "conformal")
        & np.isclose(backtest["alpha"], 0.05),
        ["date", "actual_return", "predicted_lower_quantile", "violation"],
    ].copy()
    spy_path.to_csv(DATA_DIR / "spy_conformal_5pct_path.csv", index=False)


def plot_model_comparison(summary: pd.DataFrame) -> None:
    """Plot violation rates and quantile losses for every model and tail level.

    Parameters
    ----------
    summary
        Frozen model summary with one row per model and tail probability.

    Returns
    -------
    None
        Writes two high-resolution Portable Network Graphics (PNG) files.
    """
    chart_data = summary.assign(label=summary["model"].map(MODEL_LABELS))
    positions = np.arange(len(MODEL_ORDER))
    width = 0.36

    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for offset, alpha in [(-width / 2, 0.05), (width / 2, 0.01)]:
        subset = chart_data.loc[np.isclose(chart_data["alpha"], alpha)].set_index("label")
        values = subset.reindex(MODEL_ORDER)["violation_rate"].to_numpy() * 100.0
        axis.bar(
            positions + offset,
            values,
            width,
            label=f"Nominal {alpha:.0%} tail",
            color=TAIL_COLORS[alpha],
        )
        axis.axhline(alpha * 100.0, color=TAIL_COLORS[alpha], linestyle="--", linewidth=1.4)
    axis.set_title("Observed VaR violation rate, 31 May to 29 December 2023")
    axis.set_ylabel("Violation rate (%)")
    axis.set_xticks(positions, MODEL_ORDER, rotation=18, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(
        IMAGE_DIR / "01_violation_rates.png", dpi=220, facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for offset, alpha in [(-width / 2, 0.05), (width / 2, 0.01)]:
        subset = chart_data.loc[np.isclose(chart_data["alpha"], alpha)].set_index("label")
        values = subset.reindex(MODEL_ORDER)["avg_quantile_loss"].to_numpy() * 10_000.0
        axis.bar(
            positions + offset,
            values,
            width,
            label=f"{alpha:.0%} tail",
            color=TAIL_COLORS[alpha],
        )
    axis.set_title("Average pinball loss across five forecast series")
    axis.set_ylabel("Pinball loss (basis points of return)")
    axis.set_xticks(positions, MODEL_ORDER, rotation=18, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(
        IMAGE_DIR / "02_quantile_loss.png", dpi=220, facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)


def plot_spy_path(path_data: pd.DataFrame) -> None:
    """Plot SPY returns against the adaptive conformal 5% lower quantile.

    Parameters
    ----------
    path_data
        Dated SPY return forecasts with breach indicators.

    Returns
    -------
    None
        Writes one high-resolution PNG file under ``blog/images``.
    """
    dates = pd.to_datetime(path_data["date"])
    realized = path_data["actual_return"] * 100.0
    lower_quantile = path_data["predicted_lower_quantile"] * 100.0
    breaches = path_data["violation"].astype(str).str.lower().eq("true")

    fig, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    axis.plot(dates, realized, color="#8291a5", linewidth=1.1, label="SPY daily log return")
    axis.plot(dates, lower_quantile, color="#18aaa3", linewidth=2.0, label="Adaptive conformal 5% quantile")
    axis.scatter(
        dates.loc[breaches],
        realized.loc[breaches],
        color="#d84f49",
        edgecolor="white",
        linewidth=0.6,
        s=42,
        zorder=3,
        label="Violation",
    )
    axis.axhline(0.0, color="#1c2734", linewidth=0.8, alpha=0.6)
    axis.set_title("SPY adaptive conformal VaR forecast path")
    axis.set_ylabel("Daily log return (%)")
    axis.set_xlabel("Forecast date")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.2)
    fig.savefig(
        IMAGE_DIR / "03_spy_forecast_path.png", dpi=220, facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for data refresh and chart generation.

    Returns
    -------
    argparse.Namespace
        Parsed ``refresh_data`` Boolean flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Rebuild frozen CSV inputs from outputs/runs before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    """Optionally refresh frozen data, then regenerate all technical charts."""
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_data:
        refresh_frozen_data()

    summary = pd.read_csv(DATA_DIR / "model_summary.csv")
    path_data = pd.read_csv(DATA_DIR / "spy_conformal_5pct_path.csv")
    plot_model_comparison(summary=summary)
    plot_spy_path(path_data=path_data)


if __name__ == "__main__":
    main()

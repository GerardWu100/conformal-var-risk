"""Script entrypoint for the offline conformal Value-at-Risk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from conformal_var_risk.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse the small command-line surface for one pipeline run."""
    parser = argparse.ArgumentParser(
        description="Run the offline conformal Value-at-Risk backtest pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.toml",
        help="Path to the pipeline config TOML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline and print generated artifact paths."""
    arguments = parse_args()
    artifacts = run_pipeline(config_path=arguments.config)
    print(f"daily_returns={artifacts.daily_returns_path}")
    print(f"daily_realized_variance={artifacts.realized_variance_path}")
    print(f"feature_table={artifacts.feature_table_path}")
    print(f"backtest_results={artifacts.backtest_results_path}")
    print(f"metrics={artifacts.metrics_path}")


if __name__ == "__main__":
    main()

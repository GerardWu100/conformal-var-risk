"""Rolling backtest engine for one-step-ahead Value-at-Risk forecasts."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from conformal_var_risk.models.base import VaRModel


def run_backtest(
    returns_by_asset: pd.DataFrame,
    alphas: list[float],
    calibration_window: int,
    model_factories: dict[str, Callable[[float], VaRModel]],
) -> pd.DataFrame:
    """Run a rolling-origin backtest across assets, models, and tail levels.

    Parameters
    ----------
    returns_by_asset
        DataFrame indexed by business date. Each column stores one daily log-return
        series.
    alphas
        Left-tail probabilities. For example, ``0.05`` means 95% Value-at-Risk.
    calibration_window
        Number of trailing observations passed into each model fit.
    model_factories
        Mapping from model name to an alpha-aware constructor. The constructor
        receives the target left-tail probability so adaptive models can align
        their initial internal state with the requested coverage level.

    Returns
    -------
    pd.DataFrame
        Long-form panel with one forecast row per asset, model, alpha, and
        evaluation date.

    Notes
    -----
    The loop is intentionally explicit so each stage is easy to trace:
    choose one asset, walk one rolling window at a time, emit one row per
    model-alpha pair, then call ``observe`` with the realized return.
    """
    forecast_rows: list[dict[str, object]] = []
    sorted_returns = returns_by_asset.sort_index()

    for asset_name in sorted_returns.columns:
        asset_series = sorted_returns[asset_name].dropna()

        for model_name, model_factory in model_factories.items():
            # One model instance per alpha so adaptive state does not leak across levels.
            models_by_alpha = {alpha: model_factory(alpha) for alpha in alphas}

            for evaluation_index in range(calibration_window, len(asset_series)):
                # Fit on the trailing window, then score the next realized return.
                training_returns = asset_series.iloc[
                    evaluation_index - calibration_window : evaluation_index
                ].to_numpy(dtype=float)
                evaluation_date = asset_series.index[evaluation_index]
                realized_return = float(asset_series.iloc[evaluation_index])

                for alpha, model in models_by_alpha.items():
                    model.fit(training_returns)
                    lower_quantile = model.predict_lower_quantile(alpha=alpha)

                    # Score the model's actual return quantile. VaR remains a
                    # nonnegative loss amount even when that quantile is positive.
                    lower_bound, upper_bound = model.predict_interval(alpha=alpha)
                    predicted_var = max(-lower_quantile, 0.0)
                    predicted_es = model.predict_es(alpha=alpha)
                    is_violation = realized_return < lower_quantile

                    forecast_rows.append(
                        {
                            "date": evaluation_date,
                            "asset": asset_name,
                            "model": model_name,
                            "alpha": alpha,
                            "predicted_var": predicted_var,
                            "predicted_es": predicted_es,
                            "predicted_lower_quantile": lower_quantile,
                            "interval_lower": lower_bound,
                            "interval_upper": upper_bound,
                            "actual_return": realized_return,
                            "violation": is_violation,
                        }
                    )

                    model.observe(realized_return=realized_return, alpha=alpha)

    return pd.DataFrame(forecast_rows)

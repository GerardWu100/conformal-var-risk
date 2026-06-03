"""Coverage and sharpness metrics for lower-tail VaR and ES backtests.

This module keeps the statistical summaries close to the financial definitions.
Value-at-Risk, abbreviated VaR, is backtested through violation frequency and
the Christoffersen coverage tests. Expected Shortfall, abbreviated ES, is
backtested through Acerbi-Szekely style statistics that evaluate the realized
severity of VaR breaches against the predicted ES levels. Lower-tail forecast
accuracy is measured with quantile loss, also called pinball loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from conformal_var_risk.config import EvaluationConfig


def compute_summary_metrics(
    backtest_results: pd.DataFrame,
    evaluation_config: EvaluationConfig,
) -> pd.DataFrame:
    """Compute VaR and ES backtest summaries across configured periods.

    Parameters
    ----------
    backtest_results
        Long-form backtest panel. Expected columns are `date`, `asset`, `model`,
        `alpha`, `predicted_lower_quantile`, `predicted_var`, `predicted_es`,
        `interval_lower`, `interval_upper`, `actual_return`, and `violation`.
    evaluation_config
        Config-driven period definitions and ES backtest labeling settings.

    Returns
    -------
    pd.DataFrame
        Summary metrics for each configured period.
    """
    dated_results = backtest_results.copy()
    dated_results["date"] = pd.to_datetime(dated_results["date"])

    summary_frames: list[pd.DataFrame] = []
    period_frames = _slice_configured_periods(
        dated_results=dated_results,
        evaluation_config=evaluation_config,
    )
    for period_name, period_frame in period_frames.items():
        if period_frame.empty:
            continue
        summary_frames.append(
            _summarize_period(
                period_frame=period_frame,
                period_name=period_name,
                evaluation_config=evaluation_config,
            )
        )

    return pd.concat(summary_frames, ignore_index=True)


def _slice_configured_periods(
    dated_results: pd.DataFrame,
    evaluation_config: EvaluationConfig,
) -> dict[str, pd.DataFrame]:
    """Build one DataFrame slice per configured named calendar period."""
    period_frames: dict[str, pd.DataFrame] = {}
    for period_config in evaluation_config.periods:
        start_timestamp = pd.Timestamp(period_config.start_date)
        end_timestamp = pd.Timestamp(period_config.end_date)
        in_period = (dated_results["date"] >= start_timestamp) & (
            dated_results["date"] <= end_timestamp
        )
        period_frames[period_config.name] = dated_results.loc[in_period].copy()
    return period_frames


def _summarize_period(
    period_frame: pd.DataFrame,
    period_name: str,
    evaluation_config: EvaluationConfig,
) -> pd.DataFrame:
    """Aggregate one configured period into one summary row per asset-model-alpha."""
    rows: list[dict[str, object]] = []
    grouped = period_frame.groupby(["asset", "model", "alpha"], sort=True)
    for (asset_name, model_name, alpha), group in grouped:
        # VaR coverage tests use the breach indicator; ES tests use breach severity.
        violation_series = group["violation"].astype(int).to_numpy(dtype=int)
        interval_width = group["interval_upper"] - group["interval_lower"]
        uc_statistic, uc_p_value = christoffersen_unconditional_coverage(
            violations=violation_series,
            alpha=float(alpha),
        )
        cc_statistic, cc_p_value = christoffersen_conditional_coverage(
            violations=violation_series,
            alpha=float(alpha),
        )
        es_z1_statistic = acerbi_szekely_z1_statistic(group=group)
        es_z2_statistic = acerbi_szekely_z2_statistic(group=group, alpha=float(alpha))
        es_z1_p_value = acerbi_szekely_z1_p_value(group=group)
        es_z2_p_value = acerbi_szekely_z2_p_value(group=group, alpha=float(alpha))
        rows.append(
            {
                "period": period_name,
                "asset": asset_name,
                "model": model_name,
                "alpha": float(alpha),
                "observations": int(len(group)),
                "violations": int(violation_series.sum()),
                "violation_rate": float(violation_series.mean()),
                "coverage_rate": float(1.0 - violation_series.mean()),
                "avg_quantile_loss": float(
                    quantile_loss(
                        actual_returns=group["actual_return"],
                        predicted_lower_quantiles=group["predicted_lower_quantile"],
                        alpha=float(alpha),
                    ).mean()
                ),
                "avg_predicted_var": float(group["predicted_var"].mean()),
                "avg_predicted_es": float(group["predicted_es"].mean()),
                "realized_tail_loss_mean": realized_tail_loss_mean(group=group),
                "es_z1_statistic": es_z1_statistic,
                "es_z1_p_value": es_z1_p_value,
                "es_z2_statistic": es_z2_statistic,
                "es_z2_p_value": es_z2_p_value,
                "es_backtest_status": classify_es_backtest(
                    model_name=str(model_name),
                    es_z2_p_value=es_z2_p_value,
                    confidence_level=evaluation_config.es_backtest_confidence,
                ),
                "avg_interval_width": float(interval_width.mean()),
                "uc_statistic": uc_statistic,
                "uc_p_value": uc_p_value,
                "cc_statistic": cc_statistic,
                "cc_p_value": cc_p_value,
            }
        )
    return pd.DataFrame(rows)


def quantile_loss(
    actual_returns: pd.Series,
    predicted_lower_quantiles: pd.Series,
    alpha: float,
) -> pd.Series:
    """Return the lower-tail quantile pinball loss for each forecast row."""
    tail_indicator = (actual_returns < predicted_lower_quantiles).astype(float)
    forecast_error = actual_returns - predicted_lower_quantiles
    return (alpha - tail_indicator) * forecast_error


def christoffersen_unconditional_coverage(
    violations: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Compute Christoffersen's unconditional-coverage test."""
    sample_size = int(len(violations))
    if sample_size == 0:
        return np.nan, np.nan

    violation_count = int(np.sum(violations))
    empirical_rate = np.clip(violation_count / sample_size, 1e-12, 1.0 - 1e-12)
    alpha = float(np.clip(alpha, 1e-12, 1.0 - 1e-12))

    log_likelihood_null = violation_count * np.log(alpha) + (
        sample_size - violation_count
    ) * np.log(1.0 - alpha)
    log_likelihood_alt = violation_count * np.log(empirical_rate) + (
        sample_size - violation_count
    ) * np.log(1.0 - empirical_rate)
    statistic = float(-2.0 * (log_likelihood_null - log_likelihood_alt))
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return statistic, p_value


def christoffersen_conditional_coverage(
    violations: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Compute Christoffersen's conditional-coverage test statistic and p-value."""
    if len(violations) < 2:
        return np.nan, np.nan

    previous_state = violations[:-1]
    current_state = violations[1:]
    n00 = int(np.sum((previous_state == 0) & (current_state == 0)))
    n01 = int(np.sum((previous_state == 0) & (current_state == 1)))
    n10 = int(np.sum((previous_state == 1) & (current_state == 0)))
    n11 = int(np.sum((previous_state == 1) & (current_state == 1)))

    total_transitions = n00 + n01 + n10 + n11
    if total_transitions == 0:
        return np.nan, np.nan

    independence_rate = np.clip((n01 + n11) / total_transitions, 1e-12, 1.0 - 1e-12)
    pi01 = np.clip(n01 / max(n00 + n01, 1), 1e-12, 1.0 - 1e-12)
    pi11 = np.clip(n11 / max(n10 + n11, 1), 1e-12, 1.0 - 1e-12)

    log_likelihood_independent = (n00 + n10) * np.log(1.0 - independence_rate) + (
        n01 + n11
    ) * np.log(independence_rate)
    log_likelihood_markov = (
        n00 * np.log(1.0 - pi01)
        + n01 * np.log(pi01)
        + n10 * np.log(1.0 - pi11)
        + n11 * np.log(pi11)
    )
    independence_statistic = float(
        -2.0 * (log_likelihood_independent - log_likelihood_markov)
    )
    unconditional_statistic, _ = christoffersen_unconditional_coverage(
        violations=violations,
        alpha=alpha,
    )
    conditional_coverage_statistic = unconditional_statistic + independence_statistic
    p_value = float(1.0 - chi2.cdf(conditional_coverage_statistic, df=2))
    return conditional_coverage_statistic, p_value


def realized_tail_loss_mean(group: pd.DataFrame) -> float:
    """Return the positive mean realized loss on VaR-breach days."""
    tail_returns = group.loc[group["violation"], "actual_return"]
    if tail_returns.empty:
        return np.nan
    return -float(tail_returns.mean())


def acerbi_szekely_z1_statistic(group: pd.DataFrame) -> float:
    """Compute the Acerbi-Szekely Z1 ES backtest statistic."""
    breach_frame = group.loc[group["violation"]].copy()
    if breach_frame.empty:
        return np.nan

    breach_contributions = breach_frame["actual_return"] / breach_frame["predicted_es"]
    return float(breach_contributions.mean() + 1.0)


def acerbi_szekely_z2_statistic(group: pd.DataFrame, alpha: float) -> float:
    """Compute the Acerbi-Szekely Z2 ES backtest statistic."""
    scaled_tail_contributions = (
        group["actual_return"] * group["violation"].astype(float)
    ) / (alpha * group["predicted_es"])
    return float(scaled_tail_contributions.mean() + 1.0)


def acerbi_szekely_z1_p_value(group: pd.DataFrame) -> float:
    """Approximate a one-sided p-value for the Z1 mean-zero null hypothesis."""
    breach_frame = group.loc[group["violation"]].copy()
    if len(breach_frame) < 2:
        return np.nan
    z1_samples = breach_frame["actual_return"] / breach_frame["predicted_es"] + 1.0
    return one_sided_mean_zero_p_value(z1_samples.to_numpy(dtype=float))


def acerbi_szekely_z2_p_value(group: pd.DataFrame, alpha: float) -> float:
    """Approximate a one-sided p-value for the Z2 mean-zero null hypothesis."""
    z2_samples = (group["actual_return"] * group["violation"].astype(float)) / (
        alpha * group["predicted_es"]
    ) + 1.0
    return one_sided_mean_zero_p_value(z2_samples.to_numpy(dtype=float))


def one_sided_mean_zero_p_value(sample: np.ndarray) -> float:
    """Approximate a left-tailed p-value for mean(sample) = 0."""
    if len(sample) < 2:
        return np.nan

    sample_mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    if sample_std == 0.0:
        if sample_mean < 0.0:
            return 0.0
        if sample_mean > 0.0:
            return 1.0
        return 0.5

    test_statistic = np.sqrt(len(sample)) * sample_mean / sample_std
    return float(norm.cdf(test_statistic))


def classify_es_backtest(
    model_name: str,
    es_z2_p_value: float,
    confidence_level: float,
) -> str:
    """Label the ES backtest outcome using a configurable p-value threshold."""
    if model_name == "conformal":
        return "not_reported"
    if np.isnan(es_z2_p_value):
        return "insufficient_tail_data"
    if es_z2_p_value < (1.0 - confidence_level):
        return "reject_underestimated_es"
    return "do_not_reject"

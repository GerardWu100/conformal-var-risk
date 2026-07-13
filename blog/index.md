---
title: "Adaptive Conformal VaR: Calibration Has a Price"
description: "A walk-forward comparison of adaptive conformal Value-at-Risk with historical simulation, GARCH, and filtered historical simulation on four US assets and an equal-weight portfolio."
date: 2026-07-12
image: images/cover-conformal-var.png
categories: ["Quantitative Finance", "Risk Management"]
---

A one-day Value-at-Risk forecast is a line in the sand. If the model reports a 5% lower-tail boundary, returns should cross it on roughly five days out of one hundred. A boundary that is crossed too often understates risk. One that is never crossed may be safe, but it can also be too wide to guide a trading limit or capital decision.

I built this project to test that tension directly. The experiment compares adaptive conformal prediction with four familiar benchmarks, using the same data and the same walk-forward forecast dates. The result is less tidy than a model leaderboard: conformal forecasting was conservative in this sample, and that conservatism cost pinball loss.

![A lower-tail boundary adapting around market shocks](images/cover-conformal-var.png)

The image frames the problem: observations can fall through a risk boundary, and an adaptive rule changes that boundary after the miss. The empirical question is whether the extra protection is worth the added width.

## The forecast is only as honest as its clock

The tracked input contains regular-session minute bars for AAPL, JPM, TSLA, and SPY from 2019 through 2023. The pipeline takes the last close for each trading day and computes daily log returns. If $P_t$ is the closing price on day $t$ and $P_{t-1}$ is the previous close, the return is

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
$$

Here, $r_t$ is a decimal daily return. The pipeline also builds daily realized variance from intraday returns and lags its feature columns. Those features make the research table useful for later model extensions, although the five models in this comparison fit directly on trailing daily returns.

Every forecast follows the same order: fit on 1,150 past observations, predict the next day, record the realized return, then update the model. That last sequence matters. The realized return cannot influence the boundary used to judge it.

```python
training_returns = asset_series.iloc[
    evaluation_index - calibration_window : evaluation_index
].to_numpy(dtype=float)
realized_return = float(asset_series.iloc[evaluation_index])

model.fit(training_returns)
lower_quantile = min(model.predict_lower_quantile(alpha=alpha), 0.0)
is_violation = realized_return < lower_quantile
model.observe(realized_return=realized_return, alpha=alpha)
```

The four assets and an equal-weight portfolio produce five forecast series. Each series is evaluated at tail probabilities of 5% and 1% against historical simulation, Gaussian GARCH(1,1), Student-t GARCH(1,1), filtered historical simulation, and adaptive conformal VaR. GARCH means generalized autoregressive conditional heteroskedasticity, a model in which conditional variance changes through time.

## Building the conformal lower bound

The conformal model starts with a rolling-mean predictor. Let $m=20$ be the mean lookback and let $\hat{\mu}_t$ be the forecast center for day $t$:

$$
\hat{\mu}_t = \frac{1}{m}\sum_{j=1}^{m} r_{t-j}.
$$

For each calibration observation, the model keeps only downside forecast errors. Its one-sided nonconformity score is

$$
s_t = \max(\hat{\mu}_t-r_t,0),
$$

where $s_t$ measures how far the return fell below its rolling center and is zero when it did not. With $Q_p(s)$ denoting the empirical $p$-quantile of recent scores and $a_t$ denoting the model's internal tail level, the next lower return quantile is

$$
q_t(a_t)=\hat{\mu}_t-Q_{1-a_t}(s).
$$

The reported Value-at-Risk (VaR) is a positive loss amount:

$$
\operatorname{VaR}_{t,\alpha}=\max(-q_t(a_t),0),
$$

where $\alpha$ is the target tail probability. A violation occurs when $r_t<q_t(a_t)$.

After observing day $t$, the adaptive rule changes its internal tail level. Let $I_t=1$ after a violation and $I_t=0$ otherwise, and let $\gamma=0.005$ be the learning rate. The implementation uses

$$
a_{t+1}=\operatorname{clip}\left(a_t+\gamma(\alpha-I_t),0.001,0.999\right).
$$

A violation therefore lowers $a_t$, which selects a higher score quantile and pushes the next boundary downward. A quiet day raises $a_t$ gradually. The model responds to the direction of the latest coverage error without fitting a parametric return distribution.

```python
raw_shortfalls = rolling_centers - realized_segment
self._scores = np.maximum(raw_shortfalls, 0.0)

adjustment = np.quantile(self._scores, 1.0 - self.current_alpha)
lower_quantile = self._center - adjustment

breach = float(realized_return < self._last_lower_quantile)
self.current_alpha += self.learning_rate * (self._target_alpha - breach)
```

## Calibration and sharpness answer different questions

The available evaluation window runs from 31 May through 29 December 2023. There are 153 forecasts per asset or portfolio series, for 765 forecasts per model and tail level. That is enough for a compact comparison, but thin evidence for a 1% event: the expected count is only 7.65 violations after pooling all five series.

![Observed violation rates by model and tail probability](images/01_violation_rates.png)

Historical simulation came closest to the 5% target, with 42 violations and a 5.49% pooled rate. Adaptive conformal recorded 23 violations, or 3.01%. At the 1% level, conformal recorded no violations; historical simulation recorded five. The dashed lines show the nominal targets. Distance below a line is not free accuracy. It means the risk limit was wider than necessary if the sample is representative.

| Model | 5% violations | 5% rate | 1% violations | 1% rate |
|---|---:|---:|---:|---:|
| Historical simulation | 42 | 5.49% | 5 | 0.65% |
| Gaussian GARCH | 24 | 3.14% | 6 | 0.78% |
| Student-t GARCH | 24 | 3.14% | 6 | 0.78% |
| Filtered historical simulation | 27 | 3.53% | 4 | 0.52% |
| Adaptive conformal | 23 | 3.01% | 0 | 0.00% |

All five asset-level unconditional and conditional Christoffersen coverage tests had p-values above 5% for every model-tail pair. This should not be read as proof that every model is calibrated. With 153 observations per series, these tests have little power in the 1% tail. A failure to reject is weaker than positive evidence of good coverage.

Pinball loss adds the missing cost. For realized return $r_t$, predicted quantile $q_t$, and violation indicator $I_t$, the loss is

$$
L_{\alpha}(r_t,q_t)=(\alpha-I_t)(r_t-q_t).
$$

It penalizes a boundary that is too high when a loss breaches it, while still charging forecasts that sit needlessly far below ordinary returns.

![Average pinball loss by model and tail probability](images/02_quantile_loss.png)

At 5%, filtered historical simulation had the lowest average loss at 13.21 basis points of return; adaptive conformal had the highest at 14.01. At 1%, historical simulation led at 3.24 basis points, versus 3.68 for conformal. The gap is modest, but the ordering is consistent with the violation chart: conformal bought fewer breaches with a more conservative boundary.

## Watching one boundary move

The SPY path makes the mechanics easier to see. The teal line is the adaptive conformal 5% return quantile, the grey line is the realized daily log return, and red points mark violations.

SPY crossed the adaptive conformal 5% boundary five times in 153 forecasts, a 3.27% violation rate. The lower quantile is smoother than the daily return because it comes from a 20-day center and a 500-return score window. After a breach, the internal update makes the next forecast more conservative; quiet observations slowly reverse that move.

## What I would change before using it

The 1,150-day calibration requirement leaves only seven months of evaluation data. It also removes all configured COVID and 2022 rate-shock windows from the summary because those dates occur before the first forecast. A serious stress comparison needs either a longer raw history or a shorter calibration design justified out of sample.

The adaptive state also deserves a sensitivity study. The learning rate $\gamma=0.005$ is large relative to a 1% target, and clipping can matter after clustered violations. I would plot $a_t$ itself, test several learning rates, and compare rolling coverage before deciding that the update improves regime response.

Expected Shortfall (ES), the average loss conditional on entering the tail, is included as a secondary empirical diagnostic. The conformal construction targets a quantile, not a formal ES guarantee. That distinction should survive any production presentation.

The clean lesson is methodological. Coverage, sharpness, and sample size belong on the same page. In this run, adaptive conformal VaR reduced violations, but historical and filtered historical methods produced better quantile loss. A risk manager has to decide how much extra width is worth paying for before choosing the boundary.

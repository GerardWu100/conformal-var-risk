# Adaptive outline

## Archetype decision

- Problem: compare one-day lower-tail Value-at-Risk forecasts under a shared walk-forward protocol.
- Options considered: risk model, pricing model, and mixed risk-model/data-pipeline.
- Choice: risk model.
- Why: the repository's main analytical objects are breach frequency, quantile loss, Expected Shortfall diagnostics, and conditional coverage. The minute-bar pipeline matters because it fixes the information set, but it supports rather than defines the argument.
- Verify while drafting: check that the generated backtest results expose a real calibration-versus-sharpness comparison and do not imply that adaptive conformal prediction wins every metric.

## Section blueprint

1. The risk question: a one-day loss boundary that survives regime changes.
2. From minute bars to an honest forecast origin: regular-session prices, daily log returns, realized variance, and lagged features.
3. Five models on one walk-forward clock: historical simulation, Gaussian and Student-t GARCH, filtered historical simulation, and adaptive conformal.
4. How the conformal lower bound is built: rolling mean, one-sided shortfall scores, empirical score quantile, and adaptive tail probability.
5. How forecasts are judged: violation rate, pinball loss, Christoffersen tests, and Expected Shortfall as a secondary diagnostic.
6. What the experiment says: full-sample comparison, asset-level detail, and a dated forecast-path example.
7. Limits: short evaluation window caused by the 1,150-day calibration window, adaptive-state interpretation, feature/model separation, and no transaction or portfolio aggregation layer.

## Equations

- Daily log return: define close prices $P_{t}$ and $P_{t-1}$, then $r_t = \log(P_t/P_{t-1})$.
- Rolling center: define lookback $m$ and $\hat{\mu}_t = m^{-1}\sum_{j=1}^{m}r_{t-j}$.
- One-sided score: define $s_t = \max(\hat{\mu}_t-r_t,0)$.
- Conformal quantile: define current internal tail level $a_t$ and $q_t(a_t)=\hat{\mu}_t-Q_{1-a_t}(s)$.
- Finite-sample rank: define score count $n$ and $k_t=\min(n,\lceil(n+1)(1-a_t)\rceil)$.
- Equal-weight portfolio realized variance: derive it from synchronized intraday portfolio returns so covariance terms remain present.
- Adaptive update: define breach indicator $I_t$, target tail probability $\alpha$, and learning rate $\gamma$; use $a_{t+1}=\operatorname{clip}[a_t+\gamma(\alpha-I_t)]$ exactly as implemented.
- Value-at-Risk: $\operatorname{VaR}_{t,\alpha}=\max(-q_t,0)$.
- Pinball loss and empirical violation rate, with all symbols defined.

## Code excerpts

- The one-sided score construction and adaptive update from `models/conformal.py`.
- The finite-sample score order statistic from `models/conformal.py`.
- The explicit fit-predict-observe ordering from `evaluation/backtest.py` to show why realized returns cannot leak into forecasts.

## Graphs

1. Generated hero: a visual metaphor for a lower-tail boundary adapting after breach clusters.
2. Full-sample violation rate versus the nominal tail level by model. Takeaway: calibration must be read against the 5% and 1% targets, not by ranking raw breach counts alone.
3. Average pinball loss by model and tail level. Takeaway: sharper quantiles can outperform on loss while missing nominal coverage, so the two diagnostics answer different questions.
4. One asset/model forecast path, only if the evaluation sample is long enough to read. Takeaway: show the temporal relationship between realized returns, the forecast quantile, and breaches.

The final post will use at most three technical graphs; the hero is decorative rather than an evidence graph.

## Gaps and assumptions

- The tracked sample ends in 2023 and is a bounded four-asset research universe, not current market evidence.
- With a 1,150-observation calibration window, the backtest may have little or no overlap with the configured COVID and rate-shock subperiods. Empty periods must not be described as tested.
- The adaptive conformal update is reported as implemented. Its sign and behavioral effect deserve explicit scrutiny rather than a generic coverage claim.
- Classical finite-sample conformal coverage depends on exchangeability, which is not established for overlapping rolling equity-return scores.
- Pooled counts across assets and the portfolio are descriptive because their violations are cross-sectionally dependent.
- Portfolio realized variance must be calculated after combining synchronized constituent returns; averaging constituent variances omits covariance.
- Expected Shortfall is an empirical diagnostic here; the project does not claim a formal conformal guarantee for Expected Shortfall.
- Primary references will cover adaptive conformal inference, Christoffersen coverage tests, GARCH, filtered historical simulation, quantile loss, realized variance, and Basel market-risk rules.

## Scope note

The canonical workspace and only destination for this task is `<project>/blog/`. Per the user's instruction, no publish bundle will be copied to `~/projects/website`, Hugo will not be run there, and no website files or commits will be touched. The project-local blog package will be validated, committed, and pushed on the repository's current branch.

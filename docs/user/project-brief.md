## Project 6: Conformal prediction for distribution-free risk management

**Role target:** Risk quant, model validation, portfolio management
**Wow factor:** Applies a rapidly emerging statistical framework that almost no candidates know about, directly relevant to regulatory risk requirements

Conformal prediction provides **distribution-free uncertainty quantification**
with finite-sample coverage guarantees. For this repository, the practical
research question is narrower: can an adaptive conformal lower-tail method
improve one-day-ahead VaR calibration relative to standard market-risk
benchmarks on an equity return panel?

The current implementation has settled on this design:

- forecast the lower return quantile directly, then derive positive VaR from it
- compare adaptive conformal against historical simulation, GARCH-Normal,
  GARCH-Student-$t$, and filtered historical simulation
- evaluate methods with violation rate, coverage rate, quantile loss, and
  Christoffersen coverage tests
- report ES for all models, but do not rank conformal ES as a first-class
  calibrated output because its ES estimate is an empirical tail proxy

Two design ideas from the original topic note were intentionally dropped:

- Monte Carlo was replaced by filtered historical simulation because Gaussian
  Monte Carlo adds little incremental information beyond GARCH-Normal in this
  setup.
- `mapie` was not adopted because the implemented conformal method is a
  custom one-sided adaptive update rule tied directly to the rolling VaR
  backtest.

This makes the project stronger for model validation and bank risk work: the
story is now about lower-tail forecast calibration, walk-forward correctness,
and defensible benchmarking rather than a broad but loosely specified
"conformal finance" survey.

# EDA Summary

For a Monte Carlo + GBM case study, exploratory analysis focuses less on feature relationships and more on **return behavior, volatility, and correlation structure**.

## EDA Areas Reviewed

- Historical adjusted-close price levels by ticker
- Daily log-return distributions
- Annualized drift estimates by asset
- Covariance and cross-asset correlation behavior
- Missing-data handling after market-calendar alignment
- Sensitivity of outcomes to lookback-window selection

## Notes

Because this is a stochastic simulation workflow rather than a supervised learning pipeline, the most relevant exploratory work centers on whether the historical return process looks stable enough to justify a GBM baseline.

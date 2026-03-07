# Monte Carlo Simulation Summary

## Run Context
- Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL
- Lookback period: 3y
- Historical window: 2023-03-07 to 2026-03-06
- Forecast horizon: 252 trading days
- Simulation paths: 5000
- Data source: cache

## Portfolio Forecast
- Starting portfolio value: 271.19
- Expected terminal value: 375.68
- Median terminal value: 363.45
- 5th percentile terminal value: 247.97
- 95th percentile terminal value: 542.92
- Expected return: 38.53%
- Probability of loss: 10.90%
- Annualized portfolio volatility: 0.2391

## Method
This case study uses multivariate **Geometric Brownian Motion (GBM)** to simulate future asset prices. GBM models the continuously compounded return of each asset as a drift term plus a stochastic shock, while the covariance matrix preserves cross-asset correlation.

## Outputs
- `results/metrics.json`
- `results/summary.md`
- `results/portfolio_paths.png`
- `results/terminal_value_distribution.png`
# Monte Carlo Simulation Summary

## Run Context
- Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL
- Lookback period: 3y
- Historical window: 2023-03-21 to 2026-03-20
- Forecast horizon: 252 trading days
- Simulation paths: 5000
- Data source: yfinance

## Portfolio Forecast
- Starting portfolio value: 261.83
- Expected terminal value: 349.38
- Median terminal value: 338.24
- 5th percentile terminal value: 230.91
- 95th percentile terminal value: 503.68
- Expected return: 33.44%
- Probability of loss: 13.60%
- Annualized portfolio volatility: 0.2377

## Method
This case study uses multivariate **Geometric Brownian Motion (GBM)** to simulate future asset prices. GBM models the continuously compounded return of each asset as a drift term plus a stochastic shock, while the covariance matrix preserves cross-asset correlation.

## Outputs
- `results/metrics.json`
- `results/summary.md`
- `results/portfolio_paths.png`
- `results/terminal_value_distribution.png`
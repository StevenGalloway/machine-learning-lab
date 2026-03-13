# Experiment Log

## Baseline Experiment

- **Model family:** Monte Carlo simulation with multivariate Geometric Brownian Motion
- **Forecast horizon:** 180 days
- **Simulation paths:** 5,000
- **Portfolio construction:** Weighted 5-coin basket
- **Data source:** `yfinance`
- **Caching policy:** Refresh market data when cache is older than 24 hours or when `--refresh-cache` is passed

## Planned Follow-On Experiments

- Compare 1-year, 2-year, and 5-year lookback windows
- Evaluate sensitivity to weight concentration risk
- Test shrinkage covariance estimates
- Benchmark GBM against jump-diffusion extensions
- Add stress overlays for exchange outages and crash scenarios

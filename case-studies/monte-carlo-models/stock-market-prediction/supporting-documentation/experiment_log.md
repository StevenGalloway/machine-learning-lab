# Experiment Log

## Baseline Experiment

- **Model family:** Monte Carlo simulation with multivariate Geometric Brownian Motion
- **Forecast horizon:** 252 trading days
- **Simulation paths:** 5,000
- **Portfolio construction:** Equal-weight basket
- **Data source:** `yfinance`
- **Caching policy:** Refresh market data when cache is older than 24 hours or when `--refresh-cache` is passed

## Planned Follow-On Experiments

- Compare 1-year, 3-year, and 5-year lookback windows
- Evaluate sensitivity to ticker selection and concentration risk
- Test shrinkage covariance estimates
- Compare equal-weight vs custom portfolio weights
- Benchmark GBM against jump-diffusion or regime-switching extensions

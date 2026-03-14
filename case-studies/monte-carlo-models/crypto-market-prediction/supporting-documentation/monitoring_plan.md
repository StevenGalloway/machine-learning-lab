# Monitoring Plan

## Operational Monitoring

- Cache freshness in `data/cached_prices.csv`
- Script execution success/failure
- Runtime duration by simulation size
- Output artifact completeness in `results/`

## Analytical Monitoring

- Drift changes when cache refreshes
- Annualized volatility and covariance shifts
- Probability-of-loss swings across runs
- Large changes in percentile bands (p10 / p50 / p90)

## Portfolio Governance Monitoring

- Concentration risk by weight selection
- Changes in correlation structure between assets
- Review of whether the selected lookback window still represents current market conditions

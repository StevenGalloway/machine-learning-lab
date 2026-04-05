# Crypto Portfolio Monte Carlo Summary

## Portfolio Configuration
- Tickers: BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD
- Weights: {'BTC-USD': 0.4, 'ETH-USD': 0.25, 'SOL-USD': 0.15, 'BNB-USD': 0.1, 'XRP-USD': 0.1}
- Lookback period: 2y
- Forecast days: 180
- Simulation paths: 5000

## Simulation Results
- Starting portfolio value: 1.0000
- 10th percentile terminal value: 0.5603
- Median terminal value: 0.9197
- 90th percentile terminal value: 1.5238
- Expected return: -0.17%
- Probability of loss: 58.14%
- Annualized portfolio volatility: 56.41%

## Notes
- This is a stochastic simulation baseline, not a deterministic prediction engine.
- Results are sensitive to the selected lookback period, covariance estimate, and portfolio weights.

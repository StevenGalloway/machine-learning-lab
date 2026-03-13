# Crypto Portfolio Monte Carlo Summary

## Portfolio Configuration
- Tickers: BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD
- Weights: {'BTC-USD': 0.4, 'ETH-USD': 0.25, 'SOL-USD': 0.15, 'BNB-USD': 0.1, 'XRP-USD': 0.1}
- Lookback period: 2y
- Forecast days: 180
- Simulation paths: 5000

## Simulation Results
- Starting portfolio value: 1.0000
- 10th percentile terminal value: 0.4198
- Median terminal value: 0.7173
- 90th percentile terminal value: 1.2323
- Expected return: -21.35%
- Probability of loss: 78.64%
- Annualized portfolio volatility: 58.73%

## Notes
- This is a stochastic simulation baseline, not a deterministic prediction engine.
- Results are sensitive to the selected lookback period, covariance estimate, and portfolio weights.

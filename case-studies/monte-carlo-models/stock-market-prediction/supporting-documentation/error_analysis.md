# Error Analysis

Traditional prediction-error analysis does not map cleanly onto a Monte Carlo GBM workflow because the model does not produce a single supervised label prediction. Instead, error analysis is framed as **forecast realism analysis**.

## Main Failure Modes

- Historical drift is not representative of future market conditions
- Volatility regime shifts make the covariance estimate stale
- GBM underrepresents jumps, crashes, and structural breaks
- Equal-weight portfolio assumptions may not match investor intent
- Lookback windows that are too short or too long can distort inputs

## Practical Interpretation

The most important "error" in this case study is false confidence. A clean output distribution can still be misleading if the market moves into a regime that the historical sample did not represent.

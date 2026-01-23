# Success Metrics — Regression (CFB)

## Primary metrics
- **MAE**: average absolute error in points (easy to interpret)
- **RMSE**: penalizes large misses (useful if big misses are costly)
- **R²**: variance explained

## Relative error metric
- **SMAPE** (recommended): 0.236
> Note: MAPE can be unstable when actual values are near zero. This dataset includes occasional very low scores, so this case study also reports **SMAPE** as a more stable relative error metric.


## Current results (test set)
- Baseline MAE: 11.10 | RMSE: 14.46 | R²: -0.017
- LinearReg MAE: 8.72 | RMSE: 11.41 | R²: 0.367

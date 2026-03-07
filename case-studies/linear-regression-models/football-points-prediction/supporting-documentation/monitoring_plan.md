# Monitoring Plan

## Data integrity
- Schema check: FirstDowns numeric, within expected range
- Drift: FirstDowns distribution shift (mean/variance)

## Performance monitoring (requires labels)
- MAE/RMSE tracking by time window
- Bias: mean residual near 0
- Segment checks if later features exist (home/away, opponent tier)

## Drift triggers
- Sustained increase in MAE/RMSE
- Significant shift in FirstDowns distribution

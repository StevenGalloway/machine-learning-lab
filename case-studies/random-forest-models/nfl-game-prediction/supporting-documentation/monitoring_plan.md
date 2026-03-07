# Monitoring Plan

## What to monitor (weekly / per refresh)
- **Discrimination:** ROC AUC (rolling window)
- **Threshold metrics:** accuracy, precision, recall (at a fixed operating point)
- **Probability quality:** Brier score + calibration curve drift
- **Data drift:** distribution changes in `season`, `week_num`, team frequencies

## Alerts
- AUC drop > 0.05 vs trailing 8-week mean
- Brier increase > 0.02 vs trailing 8-week mean
- Missing data in required columns

## Logging
- Dataset SHA256 + schema version
- Model version + hyperparameters
- Summary metrics + confusion matrix

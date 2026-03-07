# Monitoring Plan

## Data integrity
- Schema checks (features present, numeric ranges)
- Missing values and outliers
- Distribution drift for each feature

## Model performance (requires labels)
- Precision (PPV) of approvals (primary)
- False approval rate (FPR)
- Approval rate shift (guardrail)
- Calibration (Brier)

## Fairness monitoring (sensitive)
- Age bucket metrics
- Income quartile metrics
- Disparity alerts:
  - approval-rate differences
  - precision differences
  - FPR differences

## Drift detection
- PSI / KS tests for feature drift
- Prediction drift (mean score shift, tail behavior)

## Retraining triggers
- Sustained increase in false approvals
- Distribution drift beyond threshold
- Quarterly policy review cycle

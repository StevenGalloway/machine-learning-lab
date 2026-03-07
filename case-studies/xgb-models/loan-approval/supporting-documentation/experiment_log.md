# Experiment Log — XGBoost

## Why XGBoost for underwriting?
Boosted trees are a strong choice for structured/tabular decisioning problems:
- Capture nonlinear interactions (e.g., credit score vs loan amount)
- Strong performance without heavy feature engineering
- Robust under mixed scales

## Model configuration
- n_estimators: 500
- learning_rate: 0.03
- max_depth: 3
- subsample: 0.9
- colsample_bytree: 0.9
- reg_lambda: 1.0
- eval_metric: logloss
- seed: 42

## Results (test set)
Default threshold (0.50):
- FP (false approvals): 41
- FN: 105
- Precision: 0.634
- Recall: 0.403
- ROC AUC: 0.730

Policy threshold (0.66):
- FP (false approvals): 12
- FN: 139
- Precision: 0.755
- Recall: 0.210
- ROC AUC: 0.730

## Notes
- Policy thresholding is where business risk posture is applied.
- In real systems, this policy is aligned with credit risk and portfolio targets and reviewed by governance.

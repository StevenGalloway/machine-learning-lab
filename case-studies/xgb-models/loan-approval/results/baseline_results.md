# Baseline Results — Logistic Regression

## Why baseline?
A simple, interpretable model provides a reference point and a stability check before introducing more complex models.

## Results (test set, threshold=0.50)
Confusion matrix (Approved = positive):
- TN: 391
- FP (false approvals): 33
- FN (missed approvals): 114
- TP: 62

Metrics:
- Accuracy: 0.755
- ROC AUC: 0.755
- Precision (PPV): 0.653
- Recall: 0.352
- Specificity: 0.922
- Brier: 0.170

## Interpretation
- Precision and FP count quantify “false approvals” risk directly.
- If false approvals are too high, we adjust threshold or move to a model with better ranking and separability.

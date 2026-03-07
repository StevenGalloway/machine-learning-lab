# Baseline Results - Logistic Regression

## Why a baseline?
A linear model provides:
- A strong sanity check
- Faster training
- A reference point for whether added complexity is worth it

## Results (test set, threshold = 0.50)
Confusion matrix (malignant = positive):

- TN: 71
- FP: 1
- FN: 3
- TP: 39

Key metrics:
- Accuracy: **0.965**
- ROC AUC: **0.996**
- Sensitivity (Recall): **0.929**
- Specificity: **0.986**
- PPV (Precision): **0.975**
- NPV: **0.959**
- Brier (calibration): **0.021**

## Interpretation
- Sensitivity is the headline metric: false negatives are the primary safety risk.
- If sensitivity is below the clinical target, we adjust the threshold or use a stronger model.

## Next step
Try a boosted-tree model to capture nonlinear interactions and improve ranking (ROC AUC) while preserving high sensitivity.

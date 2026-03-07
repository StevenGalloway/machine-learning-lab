# Experiment Log — Boosted Trees (XGBoost-style)

## Model choice rationale
Boosted trees are a strong default for structured/tabular clinical measurement data:
- Capture nonlinearities and feature interactions
- Often achieve strong ROC AUC
- Provide feature importance signals for review

## Model configuration
- Estimators (trees): 300
- Learning rate: 0.05
- Max depth: 3
- Subsample: 0.9
- Colsample by tree: 0.9
- Objective: binary logistic
- Evaluation metric: logloss
- Seed: 42

## Results (test set, threshold = 0.50)
Confusion matrix (malignant = positive):
- TN: 72
- FP: 0
- FN: 4
- TP: 38

Key metrics:
- Accuracy: **0.965**
- ROC AUC: **0.994**
- Sensitivity (Recall): **0.905**
- Specificity: **1.000**
- PPV (Precision): **1.000**
- NPV: **0.947**
- Brier (calibration): **0.023**

## Clinically oriented operating point (high sensitivity)
A threshold was selected to target sensitivity ≈ **0.976**:
- Chosen threshold: **0.193**
- Specificity at this threshold: **1.000**
- False negatives at this threshold: **1**
- False positives at this threshold: **0**

## Feature importance (top 10)
- worst perimeter: 0.3203
- worst radius: 0.1779
- worst concave points: 0.1207
- mean concave points: 0.1117
- concavity error: 0.0236
- mean concavity: 0.0218
- worst concavity: 0.0212
- worst area: 0.0196
- mean texture: 0.0172
- mean area: 0.0171

## Notes for a real clinical environment
- Perform external validation across sites/devices
- Evaluate calibration and consider Platt/Isotonic calibration
- Document intended use and failure modes before any clinical rollout

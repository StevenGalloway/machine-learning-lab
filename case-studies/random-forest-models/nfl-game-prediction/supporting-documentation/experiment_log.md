# Experiment Log

- Run type: offline training + evaluation
- Random seed: `42`
- Train/test split: **season-aware holdout** (most recent season used as test when available)
- Primary metric: **ROC AUC**
- Probability quality metric: **Brier score** (lower is better)

## Baseline
- Model: Logistic Regression
- Purpose: sanity-check and easy-to-explain baseline

## Candidate model
- Model: Random Forest + isotonic calibration
- Hyperparameters:
```json
{
  "n_estimators": 500,
  "max_depth": null,
  "min_samples_leaf": 2,
  "min_samples_split": 4,
  "class_weight": "balanced",
  "calibration": {
    "method": "isotonic",
    "cv": 3
  }
}
```

## Results (this run)
- Baseline ROC AUC: **0.930**
- Random Forest ROC AUC: **0.930**
- Random Forest Accuracy: **0.858**
- Random Forest Brier: **0.105**

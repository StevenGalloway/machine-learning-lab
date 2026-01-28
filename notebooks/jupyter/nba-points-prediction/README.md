# NBA Home Win Prediction (Random Forest)

This project predicts whether the **home team wins or loses** an NBA game using a **Random Forest classifier**.

It is designed as an **enterprise-style machine learning notebook**, focusing on:
- Reproducibility
- Proper evaluation
- Clean modeling patterns (no leakage, no ad-hoc preprocessing)

---

## Problem Statement

Given basic team performance statistics for a game,  
can we predict whether the **home team will win**?

This is framed as a **binary classification problem**:

- `WIN = 1` → Home team wins  
- `WIN = 0` → Home team loses

---

## Model Type

**Random Forest Classifier**

Why Random Forest:
- Strong baseline for tabular data
- Captures non-linear relationships
- Minimal feature engineering required
- Provides feature importance for explainability

---

## Dataset

Input features:

- `PTS_HOME`, `PTS_AWAY`
- `FG_PCT_HOME`, `FG_PCT_AWAY`
- `REB_HOME`, `REB_AWAY`
- `AST_HOME`, `AST_AWAY`

Target:

- `WIN` (1 if home points > away points)

---

## Methodology

Key enterprise modeling practices used:

- Reproducible train/test split
- `Pipeline` for preprocessing + model
- Input validation & schema checks
- No target leakage
- Probabilistic evaluation using ROC AUC

---

## Evaluation Metrics

The model reports:

- **Accuracy**
- **ROC AUC**
- **Precision / Recall / F1**
- Confusion Matrix
- ROC Curve

Example output:

```json
{
  "accuracy": 0.81,
  "roc_auc": 0.88,
  "precision": 0.83,
  "recall": 0.79
}
```

### Interpretation

- ROC AUC near **0.9** indicates strong discrimination ability.
- Accuracy above **80%** is very strong for sports outcome prediction.

---

## Explainability

Random Forest feature importance is generated, showing which inputs
contribute most to predictions (e.g. shooting percentage vs rebounds).

---

## Run Instructions

Open the notebook:

```
nba-points-prediction_ENTERPRISE.ipynb
```

The notebook generates:

```
results/metrics.json
```

on each run as a structured evaluation artifact.

---

## Why this project is enterprise-ready

This is not a toy notebook:

- Uses `Pipeline` (training-serving consistency)
- Produces structured metrics JSON
- No hardcoded paths
- No inline data mutation
- Proper classification framing

This mirrors real-world ML workflows used in:
- Analytics engineering teams
- Sports analytics
- Decision-support systems

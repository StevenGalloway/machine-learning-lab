# NFL Passing Yards Prediction (Linear Regression)

This project predicts **quarterback passing yards** using a **Linear Regression model**.

It is structured as an **enterprise-grade regression case study**, emphasizing:
- Correct problem framing
- Data leakage prevention
- Reproducible evaluation
- Interpretable modeling

---

## Problem Statement

Given pre-game factors such as:
- Passing volume
- Completion efficiency
- Defensive strength
- Game pace
- Weather

Can we predict **how many passing yards** a QB will throw for?

This is a **supervised regression problem**.

---

## Model Type

**Linear Regression**

Why Linear Regression:
- Appropriate for continuous targets
- Highly interpretable
- Strong baseline for forecasting problems
- Coefficients provide direct business insight

---

## Dataset

Synthetic but realistic dataset.

Features:
- `Pass_Attempts`
- `Completions`
- `Completion_Pct`
- `Defense_Rank`
- `Pace`
- `Wind_MPH`

Target:
- `Pass_Yards`

Dataset size:
- ~2,500 rows

---

## Methodology

Enterprise modeling practices:

- Explicit feature list
- Target leakage checks
- `Pipeline` for imputation + model
- Reproducible train/test split
- Diagnostic plots

---

## Evaluation Metrics

The model reports:

- **RMSE** – Root Mean Squared Error
- **MAE** – Mean Absolute Error
- **R²** – Variance explained

Example output:

```json
{
  "rmse": 28.9,
  "mae": 23.0,
  "r2": 0.886
}
```

### Interpretation

- **RMSE ≈ 29 yards** → typical prediction error ~2–3 completions
- **MAE ≈ 23 yards** → average miss per game
- **R² ≈ 0.89** → model explains ~89% of variance

This is **very strong performance** for a simple linear baseline.

---

## Interpretability

Linear regression coefficients are displayed, showing:

- Positive drivers (e.g. attempts, pace)
- Negative drivers (e.g. wind, strong defenses)

This makes the model **fully explainable** and business-friendly.

---

## Run Instructions

Open the notebook:

```
nfl-passing-yards-prediction_linear_reg_ENTERPRISE.ipynb
```

The notebook references:

```
data/nfl_passing_yards_dataset.csv
```

And generates:

```
results/metrics.json
```

on each run.

---

## Why this project is enterprise-ready

This notebook demonstrates real-world ML discipline:

- No target leakage
- Clean regression framing
- Reproducible metrics
- Interpretable coefficients
- Structured outputs

This is directly aligned with how forecasting models are built in:

- Finance
- Operations planning
- Capacity modeling
- Sports analytics

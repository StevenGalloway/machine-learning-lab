# Personal Loan Default Prediction (Random Forest)

Fintech startup **real-time underwriting** case study.

- Target: `default` (1=default, 0=no default)
- Primary metric: **ROC AUC**
- Most costly error: **false approvals** ⇒ FN (predict 0 but actual 1)
- Sensitive feature: `sex` (monitoring only)

## Artifacts (generated when you run the script)
- `results/metrics.json`
- `results/roc_curve.png`
- `results/confusion_matrix.png`
- `results/feature_importance_top20.png`

## Data
- `data/personal_loan_default_synthetic_1500.csv` (synthetic enterprise dataset)

## Run
```bash
python loan_default_random_forest.py
```
Generated: 2026-01-25

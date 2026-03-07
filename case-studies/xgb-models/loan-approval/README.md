# Loan Approval Decision Support (XGBoost)

> **Enterprise-style demo:** This case study is **for demonstration** but is written to mimic a real underwriting decision flow.
> **Human-in-the-loop:** The model provides a **risk/approval score** and a threshold-based recommendation. Final decisions remain with credit policy and underwriters.

## Decision supported
**Decision:** “Should this application be recommended for approval under current policy?”

## Cost posture (policy)
- **False approvals (approving someone who should be rejected)** are considered **more costly** than false rejections.
- We therefore evaluate and tune thresholds to **minimize false approvals**, prioritizing **precision** and low **false positive rate**.

## Quick results (test set)
Positive class = **Approved (1)**

| Model | Accuracy | ROC AUC | Precision | Recall | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.755 | 0.755 | 0.653 | 0.352 | 33 | 114 |
| XGBoost (threshold=0.50) | 0.757 | 0.730 | 0.634 | 0.403 | 41 | 105 |
| XGBoost (policy threshold=0.66) | 0.748 | 0.730 | 0.755 | 0.210 | 12 | 139 |

## Visuals
- ROC Curve: `results/roc_curve.png`
- Confusion Matrix (policy threshold): `results/confusion_matrix.png`

## How to run
```bash
python loan_approval_model.py
```

## Files
- `loan_approval_model.py` – enterprise-style training/evaluation script (baseline + XGBoost + policy threshold)
- `problem_statement.md` – underwriting framing and non-goals
- `success_metrics.md` – KPI definitions and threshold strategy
- `data_description.md` – dataset schema, distributions, and assumptions
- `feature_dictionary.md` – feature meanings and sensitive feature handling
- `eda_summary.md` – quick statistical findings
- `baseline_results.md` – baseline performance and interpretation
- `experiment_log.md` – XGBoost configuration and rationale
- `error_analysis.md` – analysis focused on false approvals
- `model_card.md` – intended use, limitations, fairness notes
- `deployment_plan.md` – assistive integration plan
- `monitoring_plan.md` – drift, fairness, and safety monitoring
- `risk_analysis.md` – governance, compliance, and misuse scenarios
- `stakeholders.md` – stakeholders and review checkpoints
- `results/metrics.json` – machine-readable metrics and slice metrics

---
Generated: 2026-01-23

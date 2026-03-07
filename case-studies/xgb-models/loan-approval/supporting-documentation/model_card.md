# Model Card — Loan Approval Decision Support

## Model details
- Model type: XGBoost boosted-tree classifier (binary logistic)
- Output: approval probability-like score (0–1)
- Intended to support human underwriting, not replace it

## Intended use
- Assistive recommendation (approve vs reject) under defined policy threshold
- Human-in-the-loop required

## Out of scope
- Autonomous approval without governance
- Use in jurisdictions or products without legal/policy review

## Performance (hold-out test)
Policy threshold: 0.66
- Precision (PPV): 0.755
- Recall: 0.210
- Specificity: 0.972
- ROC AUC: 0.730
- False approvals (FP): 12

## Fairness and sensitive attributes
Sensitive features in this demo:
- Age
- Income

We track slice metrics by age buckets and income quartiles to detect disparities. In production, consider:
- excluding sensitive attributes from decisioning
- using them only for fairness evaluation
- governance sign-off and documentation

## Limitations
- Demo dataset with limited feature set
- No channel, product, or temporal drift data
- No ground-truth default outcomes (only historical approve/reject decisions)

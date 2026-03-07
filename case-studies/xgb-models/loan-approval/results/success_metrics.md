# Success Metrics (Underwriting / Risk)

## Primary objective
**Minimize false approvals** (predict Approved when ground truth is Rejected).

Operationally, this emphasizes:
- **Precision (PPV)** of predicted approvals (higher precision = fewer false approvals)
- **False Positive Rate** among rejections (lower FPR = fewer false approvals)
- **Approval rate** stability (avoid trivial solution of approving nobody)

## Secondary objectives
- **Recall** (how many truly-approvable applicants we still approve)
- **ROC AUC** for ranking quality
- **Calibration (Brier score)** for score reliability (important for score-based policies)

## Policy threshold strategy
We select a threshold to minimize false approvals using a weighted cost function:
- FP cost weight > FN cost weight
- With a minimum approval-rate guardrail (avoid degenerate "approve none")

Policy parameters used in this run:
- FP weight: 7.0
- FN weight: 1.0
- Minimum predicted approval rate: 0.07
- Selected threshold: 0.66

## Fairness (sensitive features)
We treat **Age** and **Income** as sensitive for governance:
- Track approval-rate and error-rate differences across age buckets and income quartiles
- Require review if large disparities emerge

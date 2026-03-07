# Risk Analysis (Governance & Misuse)

## Primary risks
1. **Financial risk from false approvals**
2. **Fair lending / disparate impact risk**
3. **Automation bias** (over-reliance on model)
4. **Data drift** (changing applicant pools)

## Controls
- Conservative thresholding aligned to policy
- Human-in-the-loop requirement
- Explainability and audit logs
- Fairness slice monitoring (Age/Income)
- Rollback and feature flags

## Misuse scenarios
- Fully automated approvals without governance
- Using sensitive attributes without legal/policy review
- Model used outside validated population/product

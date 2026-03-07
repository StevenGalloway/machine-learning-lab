# Error Analysis (False Approvals Focus)

## Why false approvals are high-risk
A false approval can introduce:
- higher default risk
- collections cost and charge-offs
- regulatory and reputational risk if decisioning is inconsistent

## Observed errors (test set)
At policy threshold 0.66:
- False approvals (FP): 12
- Missed approvals (FN): 139

## Mitigations
1. **Conservative thresholds** for high-risk segments
2. **Second-review band**: route borderline scores to manual underwriter review
3. **Reject reasons / explainability** (e.g., SHAP) for auditability
4. **Drift monitoring** for score shifts and data distribution changes
5. **Periodic re-calibration** to maintain score reliability

## Operational considerations
- Track “review burden” created by manual review band
- Ensure applicants are handled consistently across channels

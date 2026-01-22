# Error Analysis (Safety-first)

## Why focus on false negatives?
In this clinical decision support setting, **false negatives are the highest-risk error**:
- A malignant case scored as low-risk could delay follow-up and treatment.

## False negatives observed (test set)
At default threshold 0.50:
- False negatives (FN): **4**

At high-sensitivity threshold 0.193:
- False negatives (FN): **1**
- Tradeoff: more false positives → increased clinician review burden

## Recommended mitigations
1. **Use a conservative threshold** tuned for sensitivity.
2. **Human-in-the-loop escalation**: borderline cases get a second read.
3. **Uncertainty / abstention**: if score is near the threshold, route to “needs review”.
4. **Continuous monitoring** for drift; retrain when performance declines.
5. **Calibration checks** so clinicians can interpret scores responsibly.

## Operational considerations
- Track review volume (FP load) to prevent alert fatigue.
- Provide explanations (top features / SHAP) to support clinician trust.
- Use “model as second opinion,” not as gatekeeper.

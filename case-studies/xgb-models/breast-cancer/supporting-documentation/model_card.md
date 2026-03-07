# Model Card — Breast Cancer Risk Scoring (Assistive CDS)

## Model details
- Model type: Boosted tree classifier (XGBoost-style)
- Input: numeric tumor measurement features (30)
- Output: probability-like **risk score** for malignant outcome
- Decision: optional threshold-based recommendation (“escalate for review”)

## Intended use
**Assistive clinical decision support** for prioritizing follow-up / review. The model is designed to be used:
- By clinicians/radiologists as a *second opinion*
- With documented thresholds aligned to clinical safety targets

## Out of scope / not intended
- Autonomous diagnosis
- Use without clinician oversight
- Use on populations/devices/sites not validated

## Performance (hold-out test set)
Default threshold 0.50:
- Sensitivity: 0.905
- Specificity: 1.000
- ROC AUC: 0.994

High-sensitivity threshold 0.193:
- Sensitivity: 0.976
- Specificity: 1.000

## Ethical considerations
- Avoid over-reliance (automation bias). Provide training and UI cues.
- Ensure equitable performance across demographic groups when real patient data is used.
- Maintain transparency: document limitations, data coverage, and failure modes.

## Caveats and limitations
- Educational dataset; not representative of any real hospital population
- No demographics or site/device metadata included; cannot assess subgroup fairness here
- Calibration may not hold across institutions; requires local validation

## Safety
- Prioritize sensitivity to reduce false negatives
- Human-in-the-loop escalation remains mandatory
- Monitoring and rollback plan required

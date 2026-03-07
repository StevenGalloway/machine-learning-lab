# Monitoring Plan

## What to monitor
### 1) Data integrity
- Missing features / schema drift
- Out-of-range values
- Input distribution shift

### 2) Model performance (requires labels)
- Sensitivity / FN rate (primary safety KPI)
- Specificity / FP load (operational burden)
- ROC AUC (ranking quality)
- Calibration (Brier, reliability curves)

### 3) Drift detection
- Feature distribution drift (PSI / KS tests)
- Prediction drift (mean score shift, tail behavior)
- Concept drift (performance decay over time)

### 4) Safety & workflow metrics
- Escalation rate
- “Disagree with clinician decision” flags (if captured)
- Manual override rate
- Time-to-follow-up metrics (clinical operations)

## Monitoring cadence
- Daily: data integrity, prediction drift
- Weekly: performance (if labels available), alert load
- Monthly/Quarterly: recalibration, retraining evaluation

## Alerts (examples)
- Sensitivity falls below target for N consecutive windows
- FN count exceeds threshold
- Data drift surpasses PSI threshold
- Escalation rate spikes (alert fatigue risk)

## Retraining triggers
- Persistent drift + degraded performance
- New device / protocol rollout
- Periodic refresh (e.g., quarterly) after clinical review

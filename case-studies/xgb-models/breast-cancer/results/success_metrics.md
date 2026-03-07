# Success Metrics (Clinical)

## Primary objective
**Minimize missed malignancies (false negatives).**

In this framing, the *positive class* is **malignant**. Therefore:
- **Sensitivity / Recall** = TP / (TP + FN) is the *most important* metric.
- A higher sensitivity reduces the risk of a malignant case being marked low-risk.

## Secondary objectives
- **Specificity** = TN / (TN + FP) to limit unnecessary escalation.
- **ROC AUC** for overall ranking quality.
- **Calibration** (e.g., Brier score) because clinicians interpret the score as risk.

## Operating threshold philosophy (doctor-in-the-loop)
- Choose a threshold that targets **high sensitivity** (e.g., ≥ 0.97), then optimize for lower false positives.
- Clinical reality: a slightly higher false positive rate may be acceptable if it meaningfully reduces false negatives.

## Reporting
At a minimum, report:
- Confusion matrix (TP, FP, TN, FN)
- Sensitivity, Specificity, PPV, NPV
- ROC AUC
- Calibration metric (Brier)
- Segment/slice performance (if demographics or site data are available)

## Safety KPI examples
- **FN rate** = FN / (TP + FN)
- **Escalation rate** = (TP + FP) / N
- **Review burden** = FP per week/month (operational)

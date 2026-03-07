# EDA Summary (Clinical Perspective)

## Class balance
- Malignant prevalence in test set: **0.368**
- Implication: accuracy alone can be misleading; sensitivity and PPV/NPV matter.

## Feature families
Features appear in mean / standard error / worst variants, which often introduces correlation. In real systems:
- Correlated features can inflate variance for linear models
- Tree models can handle correlation better, but interpretability still requires care

## Data leakage considerations
In a real workflow, ensure features are available **at the time of decision**.
Examples of leakage (real-world):
- Using pathology results to predict pathology
- Using post-biopsy attributes in screening triage

## What we would explore further with real clinical data
- Performance by site / imaging device
- Performance by age buckets
- Calibration by subgroup
- Stability over time (drift due to device/protocol changes)

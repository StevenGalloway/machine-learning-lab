# Data Description

## Dataset provenance
This case study uses the **scikit-learn breast cancer dataset** as a *stand-in* for clinical measurement data. It contains numeric features derived from imaging-based measurements of cell nuclei.

> Important: This dataset is used for educational demonstration. It is **not** representative of all patient populations, devices, or institutions.

## Target definition
We define:
- **Positive (1) = malignant**
- **Negative (0) = benign**

This aligns evaluation metrics with clinical language: sensitivity corresponds to “catching malignant cases”.

## Dataset shape (overall)
- Samples: 569
- Features: 30

## Test split
- Test size: 20%
- Random seed: 42
- Stratified split on target

## Class balance (test set)
- Malignant prevalence (positive rate): 0.368
- Test set count: 114

## Data access & PHI
This dataset contains no PHI. In a real hospital setting:
- PHI would be protected under HIPAA.
- Access would be controlled and audited.
- De-identification and minimization would be required for model training.

# Bias & Fairness Tracking

Bias tracking is a continuous process throughout the ML lifecycle.

``` mermaid
flowchart LR
  D[Data Collection] --> A[Bias Detection]
  A --> M[Mitigation]
  M --> E[Evaluation]
  E --> D
```

## Types of Bias (with examples)

-   **Sampling Bias:** Training data underrepresents rural users in a
    credit model, causing poorer performance for them.
-   **Label Bias:** Historical hiring labels favor a particular
    demographic, encoding past discrimination.
-   **Measurement Bias:** Sensors systematically misread data for
    certain devices or environments.
-   **Historical Bias:** Past decisions (e.g., loan approvals) embed
    structural inequalities.

## Fairness Metrics (how we measure)

-   **Demographic Parity:** Equal positive rates across groups.
-   **Equal Opportunity:** Equal true positive rates across groups.
-   **Equalized Odds:** Equal false positive and false negative rates.

## Mitigation Techniques (how we fix)

-   **Data-level:** rebalancing via oversampling/undersampling; curated
    datasets.
-   **Algorithm-level:** bias-aware loss functions; adversarial
    debiasing.
-   **Post-hoc:** calibration per group; threshold adjustments.
-   **Process-level:** human-in-the-loop reviews for high-stakes
    decisions.

## Continuous Tracking in Production

-   Log predictions, features, and protected attributes (where
    legal/appropriate).
-   Monitor fairness metrics alongside accuracy.
-   Trigger alerts when disparities exceed thresholds.
-   Document decisions in model cards and incident reports.

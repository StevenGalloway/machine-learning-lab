# ML Pipeline Best Practices

These practices are designed to maximize reliability, reproducibility,
and business impact of ML systems.

``` mermaid
flowchart LR
  A[Problem Framing] --> B[Data Design]
  B --> C[Feature Engineering]
  C --> D[Training & Experimentation]
  D --> E[Evaluation]
  E --> F[Deployment]
  F --> G[Monitoring]
  G --> H[Iteration]
```

## Why adopt these practices?

-   **Lower production risk:** catches issues early (leakage, drift, bad
    labels).
-   **Better model quality:** structured evaluation reduces overfitting.
-   **Faster iteration:** clear pipelines make debugging easier.
-   **Regulatory readiness:** traceability and documentation support
    audits.
-   **Team alignment:** shared artifacts reduce miscommunication.

## 1) Problem Framing

-   Define success metrics aligned with business goals.
-   Identify constraints (latency, cost, fairness, compliance).
-   Establish baselines and an upper bound (oracle) for performance.

## 2) Data Design

-   Maintain a living feature dictionary.
-   Validate schema, ranges, freshness, and completeness.
-   Clearly define labels and prediction windows to avoid leakage.

## 3) Experimentation

-   Version control data and code.
-   Use experiment tracking (MLflow/W&B).
-   Fix random seeds and environments for reproducibility.

## 4) Evaluation

-   Use proper train/val/test splits (time-aware when needed).
-   Report multiple metrics (precision, recall, F1, ROC-AUC,
    calibration).
-   Perform slice-based analysis (by region, segment, device).

## 5) Deployment

-   Implement CI/CD for ML.
-   Maintain model cards and changelogs.
-   Use canary releases before full rollout.

## 6) Monitoring

-   Track feature and prediction drift.
-   Define SLIs/SLOs and alerting thresholds.
-   Monitor latency, error rates, and fairness metrics.

## 7) Iteration

-   Conduct blameless post-mortems on incidents.
-   Close feedback loops from production to training.

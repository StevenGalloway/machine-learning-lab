# Model Evaluation Deep Dive

``` mermaid
flowchart LR
  T[Train] --> V[Validate]
  V --> E[Evaluate]
  E --> A[Error Analysis]
  A --> T
```

## Core Metrics

-   **Precision, Recall, F1:** balance of false positives/negatives.
-   **ROC-AUC:** ranking quality across thresholds.
-   **Calibration (ECE):** reliability of predicted probabilities.
-   **Lift & Gain:** business value of targeting.

## Tradeoffs to Reason About

-   **Precision vs Recall:** depends on business cost of FP vs FN.
-   **Latency vs Accuracy:** complex models may be slower.
-   **Cost vs Performance:** serving cost vs incremental gain.

## Error Analysis (how to debug)

-   **Confusion Matrix:** where mistakes occur.
-   **Slice-Based Evaluation:** performance by segment (region, device).
-   **Residual Analysis:** systematic under/over-prediction.
-   **Leakage Checks:** ensure no future data in features.

## Statistical Validation

-   **Confidence Intervals:** uncertainty of metrics.
-   **Significance Testing:** is improvement real?
-   **A/B Testing:** causal impact in production.

## Business Impact

-   Map errors to dollars (cost of FP/FN).
-   Choose decision thresholds to maximize ROI.

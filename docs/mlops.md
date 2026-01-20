# MLOps

``` mermaid
flowchart LR
  A[Data] --> B[Feature Store]
  B --> C[Training]
  C --> D[Model Registry]
  D --> E[Serving]
  E --> F[Monitoring]
```

## Core Components

-   **Feature Store:** ensures consistent features in train and serve.
-   **Experiment Tracking:** MLflow/Weights & Biases.
-   **Model Registry:** versioning, approvals, rollback.
-   **CI/CD for ML:** automated testing and deployment.
-   **Monitoring:** drift detection, latency, errors.

## Pros of this Architecture

-   **Reproducibility:** artifacts are tracked and versioned.
-   **Scalability:** pipelines scale with data and traffic.
-   **Governance:** audit trails for compliance.
-   **Faster cycles:** automated builds reduce manual work.

## Cons / Tradeoffs

-   **Operational complexity:** more moving parts.
-   **Infrastructure cost:** feature stores and monitoring add spend.
-   **Maturity requirement:** needs strong data engineering.
-   **Latency risk:** real-time paths can become bottlenecks.

## Reliability Practices

-   Canary releases and gradual rollouts.
-   Shadow testing before production.
-   Automated rollbacks on SLO violations.

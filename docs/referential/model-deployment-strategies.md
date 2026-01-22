
# DEPLOYMENT STRATEGIES

## Serving Modes

-   **Batch Inference:** Large jobs run on a schedule (e.g., nightly
    scoring)
-   **Streaming Inference:** Event-driven predictions (e.g., click
    scoring)
-   **Real-time Inference:** Low-latency APIs (e.g., recommendations)

## Safe Rollouts

    Canary Release → Gradual Traffic Ramp → Full Rollout
              ↘
             Shadow Traffic (no user impact)

-   **Canary Deployments:** Route a small % of traffic to a new model
-   **A/B Testing:** Compare new vs baseline with randomized traffic
-   **Shadow Testing:** Run new model in parallel without affecting
    users

## Reliability & Observability

-   Define **SLIs/SLOs** (latency, availability, accuracy)
-   Automatic rollbacks on violations
-   Monitoring for:
    -   Prediction drift
    -   Feature drift
    -   Latency spikes
    -   Error rates

🔗 Example: (Insert Example Repo when completed)
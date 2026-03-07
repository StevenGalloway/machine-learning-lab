# Deployment Plan

## Serving patterns
- Batch: score a list of scenarios (first-down counts)
- Real-time: tiny API endpoint that accepts FirstDowns and returns PredictedPoints

## Versioning
- Store coefficients + intercept as the model artifact
- Store dataset version and generator parameters (since data is synthetic)

## Rollout
- Start in “analysis-only” mode (no operational decision impact)
- Add monitoring before embedding into any tooling

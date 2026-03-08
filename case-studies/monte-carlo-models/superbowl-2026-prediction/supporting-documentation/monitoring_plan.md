# Monitoring Plan

## Data monitoring
- cache freshness and file existence checks
- schema validation for required columns
- season coverage checks when loading updated data

## Model-performance monitoring
- track log loss, Brier score, AUC, and accuracy by season
- compare latest-season metrics against historical baseline bands
- monitor the stability of feature distributions such as `elo_diff` and `net_epa`

## Operational monitoring
- runtime duration
- cache refresh success/failure
- output artifact generation success

## Trigger examples
- missing required columns after load
- test Brier score deteriorates beyond agreed threshold
- prediction job fails to create `results/prediction.json`

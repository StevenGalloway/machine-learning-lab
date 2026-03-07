# Deployment Plan

## Target deployment pattern
**Batch scoring** is the simplest and most realistic approach:
- Nightly/weekly run produces probabilities for upcoming games
- Results are written to a table + dashboard

## Runtime interface (suggested)
Inputs:
- `season`, `week`, `team_home`, `team_away`
- optional: `spread_favorite`, `over_under_line`, rest days, neutral site

Outputs:
- `p_home_win` (float)
- `predicted_label` (0/1 at chosen threshold)
- model version + timestamp

## Production hardening ideas
- Move training/eval into a pipeline (Airflow/Prefect)
- Add model registry + artifact storage (MLflow)
- Add CI checks for schema drift + unit tests for feature engineering

# Feature Dictionary

This model intentionally uses **low-latency** features available before kickoff.

- `schedule_season` — used as model input
- `week_num` — used as model input
- `month` — used as model input
- `team_home` — used as model input
- `team_away` — used as model input
- `neutral_site` — used as model input
- `home_rest_days` — used as model input
- `away_rest_days` — used as model input
- `over_under_line` — used as model input
- `spread_favorite` — used as model input

## Notes (Random Forest specific)
- Random forests can capture **non-linear interactions** (e.g., team × season effects).
- Scaling is not required (tree models are scale-invariant), unlike many linear models.

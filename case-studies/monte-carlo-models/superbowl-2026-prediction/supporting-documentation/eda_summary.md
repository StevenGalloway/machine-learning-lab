# EDA Summary

## Data exploration goals
- validate the completeness of play-by-play fields needed for feature engineering
- inspect season coverage and team representation
- compare offensive and defensive efficiency distributions
- review class balance for home-team wins

## Expected observations
- home teams should win slightly more often than away teams, which justifies handling home-field bias carefully
- Elo and EPA-derived metrics should show meaningful separation between stronger and weaker teams
- regular-season data volume is materially larger than postseason volume and is therefore used for training stability

## Recommended local checks
- null-rate by field after schema fallback logic
- distribution plots for `off_epa`, `def_epa_allowed`, `net_epa`, and `elo_diff`
- season-over-season drift checks for EPA-like metrics

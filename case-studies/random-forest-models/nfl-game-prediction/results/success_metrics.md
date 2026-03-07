# Success Metrics

This is a demo use-case, so "success" is defined in **modeling terms**:

## Primary
- ROC AUC ≥ 0.55 on season-aware holdout (better than naive guessing)

## Secondary
- Brier score improves vs baseline (probability quality)
- Model is reproducible: deterministic outputs with pinned random seed
- Clear enterprise artifacts generated (model card, risk, monitoring, etc.)

## Production-ready stretch goals
- Backtesting across multiple holdout seasons
- Rich pre-game features: ELO, injuries, weather, travel, QB starters
- Calibration curve reporting + per-team performance monitoring

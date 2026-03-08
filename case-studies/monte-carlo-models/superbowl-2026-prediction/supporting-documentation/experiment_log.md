# Experiment Log

## Baseline version
- Model: HistGradientBoostingClassifier
- Calibration: isotonic calibration on validation season
- Split: train on older seasons, validate on penultimate season, test on most recent season

## Current experiment configuration
- Rolling window: 8 games
- Elo home-field adjustment: 55 points
- Elo offseason regression: 33%
- Objective: calibrated probability for neutral-site winner prediction

## Next candidate experiments
- compare isotonic vs sigmoid calibration
- add injury-adjusted or betting-market priors if an external feature source is allowed
- compare against logistic regression and random forest baselines
- stress-test different rolling windows such as 4, 6, 10, and 12

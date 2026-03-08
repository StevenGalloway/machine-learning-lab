# Error Analysis

## What errors mean here
Errors in this project are not residuals from regression. They are classification mistakes or poorly calibrated probabilities.

## Primary analysis lenses
- games where predicted favorite lost outright
- probabilities close to 0.50, where uncertainty is naturally highest
- teams with volatile late-season performance not fully captured by rolling means
- mismatches between strong underlying efficiency metrics and noisy scoreboard outcomes

## Metrics to review
- log loss for probability sharpness
- Brier score for calibration quality
- ROC AUC for ranking strength
- accuracy as a simpler directional measure

## Likely failure modes
- injuries and roster availability not represented in the data
- postseason-specific behavior differing from regular-season training patterns
- team-code mismatches or franchise transitions across long historical windows

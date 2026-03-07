# Error Analysis

Because this is a probabilistic classifier, errors should be reviewed by:
- **Confidence bucket** (high-probability mistakes are most actionable)
- **Segment** (team, season, neutral site, etc.)
- **Calibration** (are predicted probabilities aligned with observed win rates?)

## Random Forest considerations
- Miscalibration is common in tree ensembles; we calibrate with **isotonic regression**.
- Feature importance can be biased (high cardinality / correlated features). Prefer permutation/SHAP in production.

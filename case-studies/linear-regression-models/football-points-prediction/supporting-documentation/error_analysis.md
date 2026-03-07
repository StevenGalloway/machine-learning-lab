# Error Analysis

## Definition
Residual = Actual - Predicted

## What to look for
- Bias: mean residual near 0
- Pattern: residuals increasing with predicted points could indicate heteroscedasticity
- Curvature: systematic under/over prediction at extremes suggests nonlinearity

## Mitigations (if needed)
- Add features (turnovers, pace, opponent quality)
- Use regularization (Ridge/Lasso) when adding correlated features
- Consider nonlinear models if residual patterns show curvature


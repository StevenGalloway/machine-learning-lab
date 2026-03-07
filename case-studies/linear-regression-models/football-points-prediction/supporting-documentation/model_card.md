# Model Card — Points Prediction

## Model
- Type: Supervised regression (OLS Linear Regression)
- Input: FirstDowns
- Output: Predicted Points

## Intended use
- Lightweight analytics / scenario modeling
- Explainable “points per first down” relationship

## Performance (test set)
- MAE: 6.02
- RMSE: 7.47
- R²: 0.502

## Limitations
- Synthetic dataset
- Single-feature model; omits major scoring drivers
- Extrapolation outside observed first-down ranges is risky

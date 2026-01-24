# Basketball Points Prediction (Linear Regression)

Predict **next-game points per player** (`PTS_next`) from prior-game box score features.

## What you get
- **Synthetic test dataset** (1000 records): `data/basketball_player_game_logs_synth_1000.csv`
- **Enterprise-style script**: `points_prediction_linear_reg.py`
- **Artifacts**: problem statement, EDA summary, model card, monitoring plan, etc.
- **Results**: `results/metrics.json`, plots, and coefficients table

## Model type
- **Supervised regression (Linear Regression)**
- Output is continuous (points)

## Quick results (test set)
| Model | MAE | RMSE | R2 | SMAPE |
| --- | --- | --- | --- | --- |
| Baseline (next=current) | 7.75 | 9.72 | 0.278 | 0.406 |
| Linear Regression | 6.35 | 8.14 | 0.494 | 0.335 |

## Outputs
- Metrics: `results/metrics.json`
- Plots: `results/parity_plot.png`, `results/residuals.png`, `results/feature_importance.png`
- Coefficients: `results/coefficients.csv`

## Run
```bash
python points_prediction_linear_reg.py
```

---
Generated: 2026-01-24

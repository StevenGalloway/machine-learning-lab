# Points Prediction (Linear Regression) — NFL + College Datasets

> **Enterprise-style demo:** Predict **team points scored** from **first downs** using a **supervised regression** model (OLS Linear Regression).
> This case study includes **two synthetic datasets** to mimic different scoring environments:
> - **NFL-style professional football**
> - **College football (higher variance + higher scoring ceiling)**

## What’s different vs the XGBoost classification case studies
This is **regression**, so you will NOT see:
- ROC curves / AUC
- confusion matrices
- precision/recall, sensitivity/specificity
- threshold selection logic

Instead, we evaluate:
- **MAE / RMSE / MSE** (error magnitude)
- **R²** (variance explained)
- **Residual diagnostics** (bias/pattern checks)
- **Parity plot** (actual vs predicted)

## Datasets
- `data/points_prediction_data_nfl.csv` (N=1000)
- `data/points_prediction_data_cfb.csv` (N=1000)

## Quick results (test set)
| Dataset | Model | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| NFL | Baseline (mean) | 8.52 | 10.59 | -0.000 |
| NFL | Linear Regression | 6.02 | 7.47 | 0.502 |
| CFB | Baseline (mean) | 11.10 | 14.46 | -0.017 |
| CFB | Linear Regression | 8.72 | 11.41 | 0.367 |

## Outputs
### Metrics (JSON)
- `results/metrics_nfl.json`
- `results/metrics_cfb.json`

### Plots (per dataset)
NFL:
- `results/fit_line_nfl.png`
- `results/parity_plot_nfl.png`
- `results/residuals_nfl.png`

CFB:
- `results/fit_line_cfb.png`
- `results/parity_plot_cfb.png`
- `results/residuals_cfb.png`

## Run
```bash
python points_prediction_linear_reg.py --dataset nfl
python points_prediction_linear_reg.py --dataset cfb
```

---
Generated: 2026-01-23

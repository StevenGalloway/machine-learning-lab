# NFL Game Prediction — Random Forest (Enterprise Case Study)

This case study predicts **home-team win probability** for an NFL game using a **Random Forest** model and
auto-generates the documentation artifacts you'd expect in an enterprise ML workflow.

## Quickstart
```bash
python scripts/nfl_game_prediction_random_forest.py
```

## Key outputs (auto-generated)
- `results/metrics.json`
- `results/baseline_results.md`
- `results/success_metrics.md`
- `results/roc_curve.png`
- `results/confusion_matrix.png`
- `results/feature_importance_top20.png`
- `supporting-documentation/` (model card, risk analysis, monitoring plan, etc.)

## Data
Default data file:
- `/mnt/data/ml-lab_repo/case-studies/random-forest-models/nfl-game-prediction/data/spreadspoke_scores_sample.csv` (synthetic sample included so the repo runs end-to-end)

To use real historical data, drop your dataset into `data/` and pass:
```bash
python scripts/nfl_game_prediction_random_forest.py --data data/<your_file.csv>
```

## Why Random Forest here (vs linear / NB / XGBoost)
- Captures **non-linearities and interactions** with minimal feature engineering
- Provides a useful (though imperfect) notion of **feature importance**
- Often strong tabular baseline; can be calibrated for better probability quality

> Differences you may notice vs other model case studies:
> - No scaling required (tree model).
> - We include a calibration step (isotonic) and track Brier score.
> - Feature importance is impurity-based (bias risk noted in model card/risk analysis).

# Model Card — NFL Home Win Predictor (Random Forest)

## Model details
- Model type: **RandomForestClassifier** + **isotonic calibration**
- Task: binary classification (`home_win`)
- Primary metric: ROC AUC
- Training data: historical game logs (see `data_description.md`)

## Intended use
- Offline analytics, scenario planning, feature experimentation
- Demo of enterprise ML documentation + artifact generation

## Not intended for
- Betting advice, financial decision-making, or real-time critical decisions

## Performance (this run)
- Baseline ROC AUC: **0.930**
- Random Forest ROC AUC: **0.930**
- Random Forest Brier: **0.105**

## Important limitations (Random Forest specific)
- **Interpretability:** less transparent than linear models; use SHAP/permutation importance for production explanations.
- **Importance bias:** impurity-based importance can inflate high-cardinality/correlated features.
- **Calibration:** ensembles may be overconfident; we calibrate explicitly and monitor Brier score.

## Ethical / compliance
- Avoid sensitive attributes; if included, use only for fairness monitoring (not decisioning).

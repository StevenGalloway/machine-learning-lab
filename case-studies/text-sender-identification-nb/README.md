# Text Sender Identification (Naive Bayes)

Classify who sent a message using:
- **message text** (`text`)
- **time bucket** (`time_of_day`)

This case study uses **Multinomial Naive Bayes** with **bag-of-words counts**.

## How this differs from Linear Regression / XGBoost projects
- **Classification**, not regression → metrics are **Accuracy / F1**, not MAE/RMSE/R².
- Naive Bayes is **probabilistic**: learns token likelihoods per sender.
- Assumes **conditional independence** of tokens given the sender (“naive”).
- Uses **counts** (CountVectorizer) → natural fit for text.

## Two script versions (why there are two)
This case study includes **two entry points** that use the same features (`text`, `time_of_day`) but serve different purposes:

### 1) `text_sender_identification_nb.py` — Train/Eval (offline pipeline)
Use this when you want a repeatable ML workflow:
- loads the dataset
- performs a train/test split
- trains the model
- evaluates performance (baseline vs model)
- generates artifacts (metrics.json, confusion matrix image, top feature plots)

This is the version you’d use for **model development** and for producing **portfolio artifacts**.

### 2) `text_sender_identification_nb_LIVE.py` — LIVE (interactive predictions)
Use this when you want to demo the model with **user-entered input**:
- prompts the user to type:
  - message text
  - time_of_day
- returns predicted sender + probabilities
- (may still train the model first, depending on how you implemented it)

This is the version you’d use for **hands-on demos** and “what would it predict for this message?”

## Quick results (test set)
| Model | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- |
| Baseline (majority class) | 0.290 | 0.112 | Friend |
| MultinomialNB | 1.000 | 1.000 | alpha=1.0 |

## Files
- Data: `data/sms_sender_dataset_1000.csv`
- Metrics JSON (train/eval): `results/metrics.json`
- Confusion matrix (train/eval): `results/confusion_matrix.png`
- Class feature plots (train/eval): `results/top_features_<Class>.png`

## Run

### Train/Eval pipeline (recommended for artifacts)
```bash
python text_sender_identification_nb.py
```

### LIVE interactive demo
```bash
python text_sender_identification_nb_LIVE.py
```

Generated: 2026-01-25

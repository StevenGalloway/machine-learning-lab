# Experiment Log — Multinomial Naive Bayes

## Model
- MultinomialNB(alpha=1.0)
- CountVectorizer(1–2 grams) for `text`
- CountVectorizer for `time_of_day`

## Split
Stratified random split: test_size=0.2

## Results (test)
- Accuracy: 1.000
- Macro F1: 1.000

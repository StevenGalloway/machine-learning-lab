# Error Analysis

## Common failure modes
- very short messages ("On my way")
- generic phrases shared across senders
- unusual time bucket usage

## Mitigations
- add context features (thread history, day-of-week)
- try TF-IDF or char n-grams
- consider logistic regression / linear SVM / transformer for higher ceiling

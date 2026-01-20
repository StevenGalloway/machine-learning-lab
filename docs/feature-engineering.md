# Feature Engineering & Data Handling

## Feature Selection

Selecting the most relevant features to improve performance and reduce
overfitting.

**Methods** - Filter (correlation, mutual information) - Wrapper (RFE,
forward selection) - Embedded (LASSO, tree-based importance)

Example: https://github.com/example/feature-selection

## Missing Value Imputation

Handling missing data to avoid bias or model failure.

**Methods** - Mean/median imputation - KNN imputation - Model-based
imputation

Example: https://github.com/example/missing-imputation

## Loss Functions

Measure how well a model fits the data.

**Examples** - MSE (regression) - Cross Entropy (classification)

## Cross Entropy

$$ L = -\sum y \log(\hat{y}) $$

## False Positives vs False Negatives

-   **False Positive:** Model predicts positive when true label is
    negative.
-   **False Negative:** Model predicts negative when true label is
    positive.

## Cross Validation

Splitting data into multiple folds to estimate generalization
performance.

Example: https://github.com/example/cross-validation

------------------------------------------------------------------------

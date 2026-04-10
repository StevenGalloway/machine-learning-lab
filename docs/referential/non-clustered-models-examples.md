# Non-Clustered Models

## Linear Regression

Used for predicting continuous outcomes.

**Pros** - Highly interpretable - Fast to train

**Cons** - Assumes linear relationships

Example: [Football Points Prediction](case-studies/linear-regression-models/football-points-prediction/scripts/points_prediction_linear_reg.py)
Example: [Basketball Points Prediction](case-studies/linear-regression-models/basketball-points-prediction/scripts/points_prediction_linear_reg.py)

------------------------------------------------------------------------

## Logistic Regression

Used for binary classification.

$$ P(y=1|x) = \frac{1}{1 + e^{-w^Tx}} $$

**Pros** - Interpretable - Fast to train

**Cons** - Assumes linear decision boundary

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------

## Decision Trees

Splits data based on feature thresholds.

**Pros** - Interpretable - Handles non-linearity

**Cons** - Overfits easily

Example: [Breast Cancer Identification](case-studies\breast-cancer-xgb\scripts\train_eval.py)
Example: [Loan Approval Identification](case-studies\loan-approval-xgb\scripts\train_eval.py)


------------------------------------------------------------------------

## Random Forest

Ensemble of decision trees using bagging.

**Pros** - Strong performance - Reduces overfitting

**Cons** - Less interpretable

*Example:* [Ravens & Steelers Game Prediction (NFL) Random Forest Model](case-studies/nfl-game-prediction/scripts/nfl_game_prediction_random_forest.py)

------------------------------------------------------------------------

## Neural Networks

Multi-layer models that learn complex representations.

**Pros** - Powerful for large datasets - Handles complex patterns

**Cons** - Requires lots of data - Less interpretable

Example: (Insert Example Repo when completed)

------------------------------------------------------------------------


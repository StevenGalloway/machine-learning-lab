# Model Card

## Model name
Super Bowl 2026 Winner Prediction — HistGradientBoostingClassifier

## Intended use
Estimate win probability for a hypothetical or real Super Bowl-style neutral-site NFL matchup.

## Model family
- base learner: HistGradientBoostingClassifier
- post-processing: isotonic calibration
- target: binary home-win indicator during training

## Inputs
Rolling offensive and defensive team metrics, contextual season features, and Elo differential.

## Outputs
- probability that Team A wins
- probability that Team B wins
- evaluation metrics across train, validation, and test splits

## Limitations
- does not explicitly model injuries, weather, coaching changes, or roster transactions
- uses historical season snapshots rather than live depth-chart intelligence
- neutral-site probability is an approximation built from both home/away orientations

## Geometric Brownian Motion clarification
Geometric Brownian Motion is not part of this model card because this project is a sports classification model, not a financial path-simulation model.

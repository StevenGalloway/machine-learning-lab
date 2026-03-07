# Problem Statement

NFL teams (and analysts) often need a **repeatable** way to estimate game outcomes using historical signals.
This case study builds a **probabilistic classifier** that predicts whether the **home team wins** a game.

**Output:** `P(home_win=1)`  
**Decision framing:** support analysis, scenario planning, and back-testing of simple strategies (not betting advice).

## Constraints
- Historical data may have leakage risk (features that encode the outcome).
- Class imbalance is mild but non-trivial (home-field advantage).
- Model should be reproducible and easy to extend with richer features (ELO, injuries, weather, etc.).

# Risk Analysis

## Key risks
1. **Data leakage** — accidentally including post-game information (scores, box-score stats).
2. **Non-stationarity** — teams change year-to-year; performance can drift sharply.
3. **Overconfidence** — probabilities can be poorly calibrated without explicit calibration.
4. **Misuse risk** — betting/financial decisions should not be driven by this demo.

## Mitigations
- Strict feature gating (only pre-game features)
- Season-aware evaluation + monitoring by season/team
- Probability calibration + Brier score tracking
- Clear model card disclaimers + audit logs (dataset hash, params, metrics)

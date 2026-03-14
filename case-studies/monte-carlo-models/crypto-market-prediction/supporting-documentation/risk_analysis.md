# Risk Analysis

## Modeling Risks

- **Regime shift risk:** Historical returns may fail during new market conditions
- **Jump risk:** GBM does not explicitly capture discontinuous jumps
- **Estimation risk:** Drift estimates can be noisy and unstable
- **Sampling risk:** Output distributions depend on the number of simulated paths

## Business Risks

- Stakeholders may overinterpret a probabilistic range as a promise
- Portfolio results may be sensitive to concentrated holdings
- Reviewers may confuse simulation with a trained predictive model unless the README is explicit

## Controls

- Clear README language on limitations
- Distribution-based reporting instead of point-forecast marketing
- Separate monitoring documentation
- Explicit note that this is a case study and not financial advice

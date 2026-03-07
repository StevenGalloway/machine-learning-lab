# Data Description

## Dataset intent
This dataset is used for **demo purposes**, but the case study documentation is written to reflect a realistic underwriting decision flow.

## Schema
Features:
- `Age` (sensitive)
- `Income` (sensitive)
- `LoanAmount`
- `CreditScore`

Target:
- `Approved` (1 = approved, 0 = rejected)

## Size
- Rows: 2000
- Features: 4
- Test split: 30% (stratified)
- Seed: 42

## Distributions (summary)
|             |   count |       mean |        std |   min |     25% |   50% |   75% |    max |
|:------------|--------:|-----------:|-----------:|------:|--------:|------:|------:|-------:|
| Age         |    2000 |    39.5235 |    11.556  |    18 |    31   |    40 |    47 |     70 |
| Income      |    2000 | 63443.4    | 38111.1    | 18000 | 36800   | 54450 | 79200 | 293500 |
| LoanAmount  |    2000 | 10372.4    |  8253.77   |   500 |  4737.5 |  8295 | 13555 |  75000 |
| CreditScore |    2000 |   624.174  |    67.3633 |   424 |   576   |   623 |   670 |    850 |

## Correlations (features + target)
|             |       Age |   Income |   LoanAmount |   CreditScore |   Approved |
|:------------|----------:|---------:|-------------:|--------------:|-----------:|
| Age         | 1         | 0.132594 |    0.113801  |      0.207099 |  0.0775191 |
| Income      | 0.132594  | 1        |    0.785091  |      0.363362 |  0.180987  |
| LoanAmount  | 0.113801  | 0.785091 |    1         |      0.274054 |  0.0244381 |
| CreditScore | 0.207099  | 0.363362 |    0.274054  |      1        |  0.334302  |
| Approved    | 0.0775191 | 0.180987 |    0.0244381 |      0.334302 |  1         |

## Sensitive feature governance
Even when included for modeling in this demo, **Age** and **Income** require special governance in realistic settings:
- Policy/legal review (jurisdiction dependent)
- Fair lending analysis
- Documentation of justification and alternatives

> In real production systems, sensitive attributes may be excluded, used only for fairness evaluation, or handled with constrained policy rules.

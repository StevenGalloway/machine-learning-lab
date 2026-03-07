# EDA Summary

## Class balance
- Approval rate (overall): 0.293
- Approval rate (test set): 0.293

## Feature ranges (high-level)
|             |   min |     25% |   50% |   75% |    max |
|:------------|------:|--------:|------:|------:|-------:|
| Age         |    18 |    31   |    40 |    47 |     70 |
| Income      | 18000 | 36800   | 54450 | 79200 | 293500 |
| LoanAmount  |   500 |  4737.5 |  8295 | 13555 |  75000 |
| CreditScore |   424 |   576   |   623 |   670 |    850 |

## Correlation signals
Top correlations with `Approved` (absolute value):
|             |   abs_corr |
|:------------|-----------:|
| CreditScore |  0.334302  |
| Income      |  0.180987  |
| Age         |  0.0775191 |
| LoanAmount  |  0.0244381 |

## Sensitive attributes (Age, Income)
We will explicitly monitor:
- Approval-rate by age bucket and income quartile
- Precision/recall differences across groups
- Disparities in false approval rates (FPR) across groups

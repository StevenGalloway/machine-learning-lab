# Feature Dictionary

## Features
- **Age** *(Sensitive)*: applicant age. Sensitive for governance; use requires fair lending review.
- **Income** *(Sensitive)*: applicant income. Sensitive; may correlate with protected characteristics.
- **LoanAmount**: requested principal amount.
- **CreditScore**: credit score proxy for creditworthiness.

## Target
- **Approved**: historical decision outcome (1 approved, 0 rejected)

## Notes
- This dataset omits many real underwriting fields (DTI, employment history, delinquencies, collateral, etc.).
- In real systems, feature provenance and decision-time availability must be verified to avoid leakage.

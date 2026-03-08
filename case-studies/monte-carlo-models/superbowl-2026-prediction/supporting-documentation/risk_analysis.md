# Risk Analysis

## Technical risks
- upstream schema changes in `nflreadpy` or source data fields
- overfitting to historical team-strength patterns
- calibration drift in future seasons
- franchise code alias inconsistencies across data vintages

## Business and interpretation risks
- users may over-trust point estimates and ignore uncertainty
- a recruiter-facing portfolio could be misread as a betting recommendation engine
- postseason dynamics can differ from regular-season learned relationships

## Controls
- cached raw data snapshots for reproducibility
- explicit train/validation/test split by season
- structured JSON outputs for downstream auditability
- documented assumptions and limitations in the model card

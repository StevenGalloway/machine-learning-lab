# Risk Analysis (Patient Safety & Governance)

## Key risks
### 1) Patient harm from false negatives
- Mitigation: high-sensitivity threshold, human-in-loop, “needs review” band, monitoring

### 2) Automation bias
Clinicians might over-trust the score.
- Mitigation: training, UI warnings, explanations, mandate clinician decision

### 3) Dataset shift across sites/devices
Different imaging devices/protocols can change feature distributions.
- Mitigation: external validation, per-site calibration, drift detection, phased rollout

### 4) Equity and subgroup performance
Without demographic data, fairness cannot be measured.
- Mitigation: collect appropriate metadata, run subgroup evaluation, governance review

### 5) Regulatory / compliance
Clinical CDS may fall under institutional review, and potentially regulatory scrutiny depending on usage.
- Mitigation: clear intended use, documentation, auditability, clinical validation plan

## Misuse scenarios
- Using the model as an automated diagnosis → prohibited
- Using outside validated population/device → prohibited
- Using score without clinician oversight → prohibited

## Controls
- Model versioning + audit logs
- Access controls
- Monitoring + rollback
- Clinical governance sign-off for threshold changes

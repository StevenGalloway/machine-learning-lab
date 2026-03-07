# Problem Statement — Clinical Decision Support (Doctor-in-the-Loop)

## Context
Breast cancer screening and diagnostic workflows often involve multiple steps (screening → imaging workup → biopsy) with competing clinical goals:
- **Minimize missed malignancies** (false negatives) because they can delay diagnosis and treatment.
- **Avoid unnecessary invasive follow-ups** (false positives) because they add patient stress, cost, and resource burden.

This case study models a simplified version of a decision-support component: given a set of tumor measurements (features), produce a **risk score** that helps clinicians **prioritize follow-up** and **double-check borderline cases**.

## Clinical decision supported
**Decision:** “Should this case be escalated for additional diagnostic workup (e.g., imaging review / biopsy consideration)?”

- The model output is a **risk score** and a recommendation at a chosen threshold.
- **Clinicians remain the decision-makers.** The model is advisory and must not be used as a standalone diagnosis.

## Why ML is appropriate here
- Nonlinear relationships across multiple measurements can be hard to encode as fixed rules.
- A calibrated risk score can complement clinical judgment and triage.

## Assumptions
- Input measurements are available at decision time.
- The model is used in a supervised setting with outcomes known after follow-up.
- Thresholds can be tuned to clinical preference (e.g., prioritize sensitivity).

## Non-goals
- Replacing clinicians or radiologists.
- Claiming clinical validity or regulatory approval.
- Generalizing performance to any specific institution without external validation.

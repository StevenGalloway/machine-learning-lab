# Problem Statement — Loan Approval Decision Support (Human-in-the-Loop)

## Context
Loan approval decisions must balance multiple objectives:
- Maintain portfolio health (avoid high-risk approvals)
- Provide fair and consistent decisions
- Meet business growth goals while adhering to credit policy and regulatory requirements

This case study models a simplified underwriting decision support component. Given applicant attributes and loan request details, produce a score and a recommendation that helps **reduce false approvals** (approving an applicant who policy would reject).

## Decision supported
**Decision:** “Recommend approve vs reject under current policy.”

The model is **assistive**:
- It recommends decisions and provides a score
- Underwriters and credit policy remain responsible for the final outcome

## Non-goals
- Automated approval without governance
- Replacing credit policy or underwriting judgment
- Claiming regulatory compliance or production validity without validation and monitoring

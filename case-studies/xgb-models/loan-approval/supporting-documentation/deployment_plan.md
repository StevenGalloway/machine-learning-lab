# Deployment Plan (Assistive Underwriting)

## Integration pattern
- A scoring service receives application features and returns:
  - score (0–1)
  - recommended decision at policy threshold
  - explanation (top contributing features)

## Workflow (human-in-the-loop)
1. Application submitted
2. Score generated
3. Underwriter reviews recommendation + explanations
4. Final decision recorded (and used for monitoring and retraining)

## Release strategy
- Shadow mode → silent assist → assistive recommendation
- Feature flag to disable scoring if monitoring triggers fire

## Security and privacy
- Treat applicant data as sensitive
- Encrypt in transit/at rest
- Access controls + audit logs

# Deployment Plan (Doctor-in-the-Loop CDS)

## Integration approach
A practical initial deployment is a **sidecar decision support service**:
1. EHR / imaging workflow calls a scoring API with the required measurements
2. API returns:
   - risk score (0–1)
   - thresholded recommendation (e.g., “Escalate for review”)
   - explanation stub (top contributing features)

## Serving pattern
- Real-time scoring via a small API (e.g., FastAPI)
- Latency target: < 200ms per request (excluding network)
- Authentication: mTLS / OAuth2 (hospital standard)
- Audit logging: request metadata (no PHI in logs), model version, score, decision

## Release strategy
- Start with **shadow mode** (score shown only to internal QA, not clinicians)
- Then “silent assist” (score visible but not actionable)
- Then assistive alerting with conservative thresholds

## Rollback
- Versioned model registry
- Feature flag to disable model output
- Automatic rollback on monitoring triggers (e.g., sensitivity drop)

## Security & privacy
- In real deployment, treat inputs as PHI:
  - encrypt in transit and at rest
  - strict access controls
  - minimum necessary retention

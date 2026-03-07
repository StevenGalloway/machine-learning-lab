# Deployment Plan

This portfolio project is designed as a script-first analytics artifact rather than a production trading system.

## Recommended Deployment Pattern

1. Package the script in a lightweight Python environment or container
2. Schedule execution daily or on business days only
3. Refresh cached market data only when older than 24 hours unless a forced refresh is requested
4. Publish `results/metrics.json` and chart artifacts to a dashboard, storage bucket, or internal portal
5. Archive historical result snapshots for trend review

## Enterprise Extension Ideas

- Expose the simulation as an internal API
- Add configurable portfolio weights via config file
- Add parameter validation and business-calendar awareness
- Persist run history to a database for model monitoring and auditability

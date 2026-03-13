# Deployment Plan

This portfolio project is designed as a script-first analytics artifact rather than a live trading system.

## Recommended Deployment Pattern

1. Package the script in a lightweight Python environment or container
2. Schedule execution daily
3. Refresh cached market data only when older than 24 hours unless a forced refresh is requested
4. Publish `results/metrics.json` and chart artifacts to storage, a dashboard, or a portfolio site
5. Archive historical result snapshots for trend review and reproducibility

## Enterprise Extension Ideas

- Add environment-based configuration and secrets handling
- Persist run history to a database for model monitoring and auditability
- Add stress scenarios for crash days and regime breaks
- Compare equal-weight and custom weight allocations across runs

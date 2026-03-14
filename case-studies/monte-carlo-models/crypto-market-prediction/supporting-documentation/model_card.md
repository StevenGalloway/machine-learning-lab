# Model Card

## Model

Multivariate Monte Carlo simulation using **Geometric Brownian Motion (GBM)**.

## Intended Use

- Crypto portfolio scenario analysis
- Demonstration of stochastic modeling and simulation design
- Recruiter-facing data science portfolio project

## Not Intended For

- Live trading without additional controls
- Deterministic price prediction claims
- Intraday or high-frequency forecasting

## Key Assumptions

- Log returns are informative for future drift/volatility estimation
- Asset prices approximately follow a log-normal process over the forecast interval
- Historical covariance structure is a reasonable proxy for near-term future relationships

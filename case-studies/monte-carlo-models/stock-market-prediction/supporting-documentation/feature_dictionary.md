# Feature Dictionary

This project does not use a traditional feature matrix. Instead, it uses **model drivers** estimated from historical market data.

## Model Drivers

- **Ticker symbol:** Selected asset identifier used for price retrieval
- **Adjusted close price:** Historical price series used for return estimation
- **Daily log return:** Continuously compounded daily return derived from adjacent prices
- **Annualized drift (`μ`):** Historical expected return estimate
- **Annualized covariance (`Σ`):** Cross-asset covariance matrix used to preserve correlation in simulation
- **Forecast horizon:** Number of trading days projected into the future
- **Simulation paths:** Number of Monte Carlo draws used to estimate the future distribution
- **Portfolio weights:** Allocation applied to individual simulated asset paths

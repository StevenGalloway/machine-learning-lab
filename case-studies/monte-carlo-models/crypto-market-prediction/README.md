# Crypto Market Prediction with Monte Carlo Simulation

Public-facing quantitative-finance case study that forecasts a weighted crypto portfolio using **Monte Carlo simulation** powered by **Geometric Brownian Motion (GBM)**.

This project is designed for recruiter and hiring-manager review. It demonstrates enterprise-ready scripting patterns: cached inputs, centralized configuration, reproducible runs, and runtime behavior that writes only result artifacts.

## What the Project Does

The script downloads historical crypto prices from `yfinance`, estimates annualized drift and covariance from historical log returns, and simulates future correlated price paths for a configurable 5-coin portfolio. Those simulated paths are then rolled into a weighted portfolio forecast and written to the `results/` folder.

GBM is a standard stochastic process in quantitative finance. It assumes that asset prices evolve continuously over time with a deterministic drift term and a random diffusion term, making it a common baseline for portfolio forecasting and risk analysis.

## Quick Start

### Dependencies

Install the required packages:

```bash
pip install pandas numpy matplotlib yfinance
```

### Recommended Command

Run the model with a forced cache refresh:

```bash
python case-studies/monte-carlo-models/crypto-market-prediction/scripts/crypto_market_prediction_mc.py --refresh-cache
```

This command will:

1. Download historical crypto prices from `yfinance` when the cache is older than 24 hours
2. Cache the cleaned close-price history in `data/cached_prices.csv`
3. Estimate GBM inputs from historical log returns
4. Simulate weighted portfolio paths and terminal values
5. Save results-only artifacts into the `results/` directory

## One-Line Execution Example

```bash
python case-studies/monte-carlo-models/crypto-market-prediction/scripts/crypto_market_prediction_mc.py --tickers BTC-USD ETH-USD SOL-USD BNB-USD XRP-USD --weights 0.40 0.25 0.15 0.10 0.10 --forecast-days 180 --num-paths 5000 --refresh-cache
```

This shows the project is configurable and reproducible instead of being a hard-coded demo script.

## Expected Outputs

Running the script generates the following artifacts:

- `results/metrics.json` — structured run metadata, configuration, and portfolio forecast metrics
- `results/summary.md` — GitHub-friendly markdown summary of the simulation run
- `results/portfolio_paths.png` — sampled simulated portfolio trajectories
- `results/terminal_value_distribution.png` — histogram of terminal portfolio outcomes

## Project Structure

```text
crypto-market-prediction/
├── README.md
├── data/
│   └── cached_prices.csv
├── results/
│   ├── metrics.json
│   ├── summary.md
│   ├── portfolio_paths.png
│   └── terminal_value_distribution.png
├── scripts/
│   └── crypto_market_prediction_mc.py
└── supporting-documentation/
    ├── data_description.md
    ├── deployment_plan.md
    ├── eda_summary.md
    ├── error_analysis.md
    ├── experiment_log.md
    ├── feature_dictionary.md
    ├── model_card.md
    ├── monitoring_plan.md
    ├── problem_statement.md
    ├── risk_analysis.md
    └── stakeholders.md
```

## Enterprise-Ready Design Choices

- **Caching:** Historical market data is cached in `data/` and reused until it is more than 24 hours old.
- **Configurable execution:** Key settings such as tickers, portfolio weights, lookback window, forecast horizon, and simulation count are configurable from the command line.
- **Reproducibility:** The script uses a fixed random seed unless overridden.
- **Results-only runtime behavior:** The script generates only prediction artifacts and does not regenerate repo documentation on each run.
- **Public-facing structure:** Supporting documentation is maintained separately to keep runtime behavior clean and predictable.

## Modeling Notes

### Geometric Brownian Motion (GBM)

GBM models asset prices as a stochastic differential equation of the form:

```text
dS_t = μS_tdt + σS_tdW_t
```

Where:

- `S_t` is the asset price at time `t`
- `μ` is the drift term (expected rate of return)
- `σ` is the volatility term
- `dW_t` is a Wiener process increment representing random market shocks

In this project, GBM is applied in a **multivariate** setting so the covariance matrix preserves cross-asset relationships between the selected coins.

## Important Limitations

This project is a **simulation-based forecasting case study**, not a claim of deterministic crypto prediction. Results are sensitive to:

- the selected historical window
- the assumption that historical drift/volatility remain informative
- the GBM assumption of log-normal price evolution
- the exclusion of macro shocks, exchange outages, and structural breaks
- the chosen portfolio weights

# Stock Market Prediction with Monte Carlo Simulation

Public-facing quantitative finance case study that forecasts a multi-asset portfolio using **Monte Carlo simulation** powered by **Geometric Brownian Motion (GBM)**.

This project is designed for recruiter and hiring-manager review. It shows how to structure a stochastic simulation workflow in an enterprise-ready way: cached inputs, configurable execution, reproducible runs, and results-only runtime outputs.

## What the Project Does

The script downloads historical equity prices from `yfinance`, estimates annualized drift and covariance from historical log returns, and simulates future correlated price paths. Those simulated paths are then rolled into an equal-weight portfolio forecast and written to the `results/` folder.

GBM is a standard stochastic process in quantitative finance. It assumes that asset prices evolve continuously over time with a deterministic drift term and a random diffusion term, making it a common baseline for portfolio forecasting and option-pricing contexts such as Black-Scholes.

## Quick Start

### Dependencies

Install the required packages:

```bash
pip install pandas numpy matplotlib yfinance
```

### Recommended Command

Run the model with a forced cache refresh:

```bash
python case-studies/monte-carlo-models/stock-market-prediction/scripts/stock_market_monte_carlo.py --refresh-cache
```

This command will:

1. Download historical price data from `yfinance` when the cache is older than 24 hours
2. Cache the cleaned adjusted-close price history in `data/cached_prices.csv`
3. Estimate GBM inputs from historical log returns
4. Simulate portfolio paths and terminal values
5. Save results-only artifacts into the `results/` directory

## One-Line Execution Example

```bash
python case-studies/monte-carlo-models/stock-market-prediction/scripts/stock_market_monte_carlo.py --tickers AAPL MSFT NVDA AMZN GOOGL --forecast-days 252 --num-paths 5000 --refresh-cache
```

This shows the project is configurable and reproducible instead of being a hard-coded notebook exercise.

## Expected Outputs

Running the script generates the following artifacts:

- `results/metrics.json` — structured run metadata, configuration, and portfolio forecast metrics
- `results/summary.md` — GitHub-friendly markdown summary of the simulation run
- `results/portfolio_paths.png` — sampled simulated portfolio trajectories
- `results/terminal_value_distribution.png` — histogram of terminal portfolio outcomes

## Project Structure

```text
stock-market-prediction/
├── README.md
├── data/
│   └── cached_prices.csv
├── results/
│   ├── metrics.json
│   ├── summary.md
│   ├── portfolio_paths.png
│   └── terminal_value_distribution.png
├── scripts/
│   └── stock_market_monte_carlo.py
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

## CLI Reference

```
python scripts/stock_market_monte_carlo.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--tickers` | `str+` | `AAPL MSFT NVDA AMZN GOOGL` | One or more equity tickers (e.g. `AAPL MSFT TSLA`) |
| `--lookback-period` | `str` | `3y` | yfinance lookback period (e.g. `1y`, `3y`, `5y`, `max`) |
| `--forecast-days` | `int` | `252` | Number of trading days to simulate (~1 year at 252 days) |
| `--num-paths` | `int` | `5000` | Number of Monte Carlo simulation paths |
| `--seed` | `int` | `123` | Random seed for reproducible simulations |
| `--risk-free-rate` | `float` | `0.0` | Annualized risk-free rate for documentation context |
| `--refresh-cache` | flag | `False` | Force re-download of market data, ignoring the 24h cache |

Portfolio weights are always equal-weighted and computed from the number of tickers. Defaults are loaded from `configs/monte-carlo/stock_market_prediction.yaml`.

## Enterprise-Ready Design Choices

- **Caching:** Historical market data is cached in `data/` and reused until it is more than 24 hours old.
- **Configurable execution:** Key settings such as tickers, lookback window, forecast horizon, and simulation count are configurable from the command line.
- **Reproducibility:** The script uses a fixed random seed unless overridden.
- **Results-only runtime behavior:** The script generates only prediction artifacts and does not regenerate repo documentation on each run.
- **Public-facing structure:** Supporting documentation is maintained separately to keep runtime behavior clean and predictable.

## Modeling Notes

### Geometric Brownian Motion (GBM)

GBM models stock prices as a stochastic differential equation of the form:

```text
dS_t = μS_tdt + σS_tdW_t
```

Where:

- `S_t` is the asset price at time `t`
- `μ` is the drift term (expected rate of return)
- `σ` is the volatility term
- `dW_t` is a Wiener process increment representing random market shocks

In this project, GBM is applied in a **multivariate** setting so the covariance matrix preserves cross-asset relationships between the selected tickers.

## Example Use Cases

- Portfolio scenario analysis
- Recruiter-facing demonstration of stochastic simulation skills
- Educational quant-finance baseline before moving into regime-switching or factor-based models
- Stress-testing directional expectations under uncertainty bands

## Important Limitations

This project is a **simulation-based forecasting case study**, not a claim of deterministic stock prediction. Results are sensitive to:

- the selected historical window
- the assumption that historical drift/volatility remain informative
- the GBM assumption of log-normal price evolution
- the exclusion of macro events, earnings shocks, and regime breaks

## Supporting Documentation

Additional documentation is intentionally stored in `supporting-documentation/` rather than generated at runtime. That keeps the execution path aligned with other portfolio case studies while still showing enterprise documentation maturity.

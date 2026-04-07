from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
CACHE_TTL_HOURS = 24

CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "monte-carlo" / "stock_market_prediction.yaml"


def _load_yaml_config(path: Path) -> dict:
    try:
        import yaml
        with path.open("r") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


@dataclass(slots=True)
class Config:
    """Runtime configuration for the Monte Carlo stock prediction case study."""

    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
    lookback_period: str = "3y"
    forecast_days: int = 252
    num_paths: int = 5000
    seed: int = 123
    cache_ttl_hours: int = CACHE_TTL_HOURS
    refresh_cache: bool = False
    risk_free_rate: float = 0.0


CASE_STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"
CACHE_PATH = DATA_DIR / "cached_prices.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
PATHS_PLOT_PATH = RESULTS_DIR / "portfolio_paths.png"
TERMINAL_DIST_PLOT_PATH = RESULTS_DIR / "terminal_value_distribution.png"


def parse_args() -> Config:
    """Parse CLI arguments into a Config object.

    Example:
        python scripts/stock_market_monte_carlo.py --tickers AAPL MSFT NVDA --forecast-days 126 --num-paths 2000
    """

    parser = argparse.ArgumentParser(description="Enterprise-style Monte Carlo stock simulation using Geometric Brownian Motion.")
    parser.add_argument("--tickers", nargs="+", default=None, help="One or more ticker symbols.")
    parser.add_argument("--lookback-period", default=None, help="yfinance lookback period, e.g. 1y, 3y, 5y, max.")
    parser.add_argument("--forecast-days", type=int, default=None, help="Number of trading days to forecast.")
    parser.add_argument("--num-paths", type=int, default=None, help="Number of Monte Carlo simulation paths.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible simulations.")
    parser.add_argument("--risk-free-rate", type=float, default=None, help="Optional annual risk-free rate for documentation context.")
    parser.add_argument("--refresh-cache", action="store_true", help="Force refresh of cached market data.")
    args = parser.parse_args()

    # Priority: dataclass defaults < YAML config < CLI args
    yaml_cfg = _load_yaml_config(CONFIG_PATH)
    config = Config()
    for key, val in yaml_cfg.items():
        if hasattr(config, key):
            setattr(config, key, val)

    if args.tickers is not None:
        config.tickers = [ticker.upper() for ticker in args.tickers]
    if args.lookback_period is not None:
        config.lookback_period = args.lookback_period
    if args.forecast_days is not None:
        config.forecast_days = args.forecast_days
    if args.num_paths is not None:
        config.num_paths = args.num_paths
    if args.seed is not None:
        config.seed = args.seed
    if args.risk_free_rate is not None:
        config.risk_free_rate = args.risk_free_rate
    if args.refresh_cache:
        config.refresh_cache = True

    return config


def ensure_directories() -> None:
    """Create expected project directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_cache_fresh(path: Path, ttl_hours: int) -> bool:
    """Return True if the cache file exists and is newer than the configured TTL."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    modified_ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified_ts).total_seconds() / 3600
    return age_hours < ttl_hours


def extract_adjusted_close_frame(raw_data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Normalize yfinance output into a clean adjusted-close price frame.

    yfinance changes shape depending on single vs multi-ticker downloads.
    This helper makes the downstream simulation logic stable and readable.
    """
    if raw_data.empty:
        raise ValueError("Downloaded market data is empty.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        for preferred_field in ("Adj Close", "Close"):
            if preferred_field in raw_data.columns.get_level_values(0):
                price_frame = raw_data[preferred_field].copy()
                break
        else:
            raise ValueError("Could not find 'Adj Close' or 'Close' in downloaded data.")
    else:
        if set(tickers).issubset(raw_data.columns):
            price_frame = raw_data.loc[:, list(tickers)].copy()
        elif "Adj Close" in raw_data.columns:
            price_frame = raw_data[["Adj Close"]].copy()
            price_frame.columns = [tickers[0]]
        elif "Close" in raw_data.columns:
            price_frame = raw_data[["Close"]].copy()
            price_frame.columns = [tickers[0]]
        else:
            raise ValueError("Unexpected yfinance column layout; expected ticker columns or Close/Adj Close.")

    price_frame = price_frame.dropna(how="any")
    if price_frame.empty:
        raise ValueError("No complete rows remain after dropping missing market data.")

    # Preserve requested ticker order when all requested names are present.
    available = [ticker for ticker in tickers if ticker in price_frame.columns]
    if available:
        price_frame = price_frame.loc[:, available]

    return price_frame.astype(float)


def download_or_load_prices(config: Config) -> tuple[pd.DataFrame, str]:
    """Load cached prices when available, otherwise download from yfinance and cache them."""
    ensure_directories()

    if not config.refresh_cache and is_cache_fresh(CACHE_PATH, config.cache_ttl_hours):
        cached_prices = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        return cached_prices, "cache"

    raw_data = yf.download(
        tickers=config.tickers,
        period=config.lookback_period,
        progress=False,
        auto_adjust=False,
    )
    price_frame = extract_adjusted_close_frame(raw_data, config.tickers)
    price_frame.to_csv(CACHE_PATH)
    return price_frame, "yfinance"


def compute_log_returns(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from historical price history."""
    log_returns = np.log(price_frame / price_frame.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Log return frame is empty. Check lookback period and input data quality.")
    return log_returns


def simulate_gbm_paths(price_frame: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate future price paths using multivariate Geometric Brownian Motion.

    Returns:
        asset_paths: shape (forecast_days + 1, num_paths, num_assets)
        portfolio_paths: shape (forecast_days + 1, num_paths)
        weights: equal-weight portfolio weights
        log_returns: historical log returns used to estimate drift/covariance
    """
    log_returns = compute_log_returns(price_frame)
    s0 = price_frame.iloc[-1].to_numpy(dtype=float)
    num_assets = len(s0)

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    mu = log_returns.mean().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR
    sigma = log_returns.cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR
    drift = (mu - 0.5 * np.diag(sigma)) * dt

    rng = np.random.default_rng(config.seed)
    chol = np.linalg.cholesky(sigma * dt)
    standard_normals = rng.standard_normal((config.forecast_days, config.num_paths, num_assets))
    correlated_noise = standard_normals.reshape(-1, num_assets) @ chol.T
    correlated_noise = correlated_noise.reshape(config.forecast_days, config.num_paths, num_assets)

    increments = drift[None, None, :] + correlated_noise
    cumulative_log_returns = np.vstack([
        np.zeros((1, config.num_paths, num_assets)),
        np.cumsum(increments, axis=0),
    ])

    asset_paths = s0[None, None, :] * np.exp(cumulative_log_returns)
    weights = np.repeat(1.0 / num_assets, num_assets)
    portfolio_paths = (asset_paths * weights.reshape(1, 1, -1)).sum(axis=2)
    return asset_paths, portfolio_paths, weights, log_returns.to_numpy(dtype=float)


def build_metrics_payload(
    config: Config,
    price_frame: pd.DataFrame,
    portfolio_paths: np.ndarray,
    weights: np.ndarray,
    log_return_array: np.ndarray,
    data_source: str,
) -> dict:
    """Create a structured metrics payload for JSON export."""
    terminal_values = portfolio_paths[-1]
    start_value = float(np.dot(price_frame.iloc[-1].to_numpy(dtype=float), weights))
    percentiles = np.percentile(terminal_values, [5, 50, 95])
    expected_return_pct = ((terminal_values.mean() / start_value) - 1.0) * 100.0
    probability_of_loss = float(np.mean(terminal_values < start_value))

    annualized_portfolio_volatility = float(
        np.sqrt(weights.T @ (price_frame.pct_change().dropna().cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR) @ weights)
    )

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "stochastic_simulation",
            "model_family": "monte_carlo_geometric_brownian_motion",
            "data_source": data_source,
            "case_study_dir": str(CASE_STUDY_DIR.as_posix()),
        },
        "config": asdict(config),
        "inputs": {
            "tickers": config.tickers,
            "lookback_period": config.lookback_period,
            "observation_count": int(len(price_frame)),
            "start_date": str(price_frame.index.min().date()),
            "end_date": str(price_frame.index.max().date()),
            "latest_prices": {ticker: float(price_frame.iloc[-1][ticker]) for ticker in price_frame.columns},
            "portfolio_weights": {ticker: float(weight) for ticker, weight in zip(price_frame.columns, weights)},
        },
        "simulation_summary": {
            "starting_portfolio_value": start_value,
            "terminal_value_mean": float(terminal_values.mean()),
            "terminal_value_std": float(terminal_values.std(ddof=1)),
            "terminal_value_p05": float(percentiles[0]),
            "terminal_value_p50": float(percentiles[1]),
            "terminal_value_p95": float(percentiles[2]),
            "expected_return_pct": float(expected_return_pct),
            "probability_of_loss": probability_of_loss,
            "annualized_portfolio_volatility": annualized_portfolio_volatility,
            "historical_annualized_log_return_by_asset": {
                ticker: float(value)
                for ticker, value in zip(price_frame.columns, log_return_array.mean(axis=0) * TRADING_DAYS_PER_YEAR)
            },
        },
        "artifacts": {
            "metrics_json": str(METRICS_PATH.relative_to(CASE_STUDY_DIR).as_posix()),
            "summary_markdown": str(SUMMARY_PATH.relative_to(CASE_STUDY_DIR).as_posix()),
            "portfolio_paths_png": str(PATHS_PLOT_PATH.relative_to(CASE_STUDY_DIR).as_posix()),
            "terminal_distribution_png": str(TERMINAL_DIST_PLOT_PATH.relative_to(CASE_STUDY_DIR).as_posix()),
        },
    }
    return payload


def write_summary_markdown(metrics_payload: dict) -> None:
    """Write a concise markdown summary that is easy to review in GitHub."""
    summary = metrics_payload["simulation_summary"]
    inputs = metrics_payload["inputs"]
    config = metrics_payload["config"]

    lines = [
        "# Monte Carlo Simulation Summary",
        "",
        "## Run Context",
        f"- Tickers: {', '.join(inputs['tickers'])}",
        f"- Lookback period: {inputs['lookback_period']}",
        f"- Historical window: {inputs['start_date']} to {inputs['end_date']}",
        f"- Forecast horizon: {config['forecast_days']} trading days",
        f"- Simulation paths: {config['num_paths']}",
        f"- Data source: {metrics_payload['metadata']['data_source']}",
        "",
        "## Portfolio Forecast",
        f"- Starting portfolio value: {summary['starting_portfolio_value']:.2f}",
        f"- Expected terminal value: {summary['terminal_value_mean']:.2f}",
        f"- Median terminal value: {summary['terminal_value_p50']:.2f}",
        f"- 5th percentile terminal value: {summary['terminal_value_p05']:.2f}",
        f"- 95th percentile terminal value: {summary['terminal_value_p95']:.2f}",
        f"- Expected return: {summary['expected_return_pct']:.2f}%",
        f"- Probability of loss: {summary['probability_of_loss']:.2%}",
        f"- Annualized portfolio volatility: {summary['annualized_portfolio_volatility']:.4f}",
        "",
        "## Method",
        "This case study uses multivariate **Geometric Brownian Motion (GBM)** to simulate future asset prices. GBM models the continuously compounded return of each asset as a drift term plus a stochastic shock, while the covariance matrix preserves cross-asset correlation.",
        "",
        "## Outputs",
        "- `results/metrics.json`",
        "- `results/summary.md`",
        "- `results/portfolio_paths.png`",
        "- `results/terminal_value_distribution.png`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_portfolio_paths(portfolio_paths: np.ndarray, seed: int) -> None:
    """Save a chart of sampled portfolio paths."""
    rng = np.random.default_rng(seed)
    sample_count = min(200, portfolio_paths.shape[1])
    chosen_indices = rng.choice(portfolio_paths.shape[1], size=sample_count, replace=False)

    plt.figure(figsize=(10, 5.5))
    for idx in chosen_indices:
        plt.plot(portfolio_paths[:, idx], linewidth=0.7, alpha=0.25)
    plt.title("Sample Monte Carlo Portfolio Paths")
    plt.xlabel("Forecast Step (Trading Days)")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(PATHS_PLOT_PATH, dpi=150)
    plt.close()


def plot_terminal_distribution(portfolio_paths: np.ndarray) -> None:
    """Save a histogram of terminal portfolio values."""
    terminal_values = portfolio_paths[-1]
    plt.figure(figsize=(9.5, 5.5))
    plt.hist(terminal_values, bins=40)
    plt.title("Terminal Portfolio Value Distribution")
    plt.xlabel("Terminal Portfolio Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(TERMINAL_DIST_PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    """Run the simulation pipeline and write results-only artifacts.

    Supporting documentation is intentionally managed outside the runtime script so the
    repo behaves like the other case studies in this portfolio.
    """
    config = parse_args()
    price_frame, data_source = download_or_load_prices(config)
    _, portfolio_paths, weights, log_return_array = simulate_gbm_paths(price_frame, config)

    metrics_payload = build_metrics_payload(
        config=config,
        price_frame=price_frame,
        portfolio_paths=portfolio_paths,
        weights=weights,
        log_return_array=log_return_array,
        data_source=data_source,
    )

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    write_summary_markdown(metrics_payload)
    plot_portfolio_paths(portfolio_paths, config.seed)
    plot_terminal_distribution(portfolio_paths)

    summary = metrics_payload["simulation_summary"]
    logger.info("Data source:              %s", data_source)
    logger.info("Starting portfolio value: %s", f"{summary['starting_portfolio_value']:.2f}")
    logger.info("Expected terminal value:  %s", f"{summary['terminal_value_mean']:.2f}")
    logger.info("Expected return (%%):      %s", f"{summary['expected_return_pct']:.2f}")
    logger.info("Probability of loss:      %s", f"{summary['probability_of_loss']:.2%}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    main()

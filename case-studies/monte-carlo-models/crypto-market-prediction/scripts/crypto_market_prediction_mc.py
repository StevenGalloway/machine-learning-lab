from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 365
CACHE_TTL_HOURS = 24


@dataclass(slots=True)
class Config:
    """Runtime configuration for the crypto Monte Carlo case study."""

    tickers: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"])
    weights: list[float] = field(default_factory=lambda: [0.40, 0.25, 0.15, 0.10, 0.10])
    lookback_period: str = "2y"
    interval: str = "1d"
    forecast_days: int = 180
    num_paths: int = 5000
    seed: int = 123
    cache_ttl_hours: int = CACHE_TTL_HOURS
    refresh_cache: bool = False


CASE_STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"
CACHE_PATH = DATA_DIR / "cached_prices.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
PATHS_PLOT_PATH = RESULTS_DIR / "portfolio_paths.png"
TERMINAL_DIST_PLOT_PATH = RESULTS_DIR / "terminal_value_distribution.png"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Enterprise-style Monte Carlo crypto portfolio simulation using Geometric Brownian Motion.")
    parser.add_argument("--tickers", nargs="+", default=None, help="One or more crypto tickers such as BTC-USD ETH-USD.")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Portfolio weights aligned to the ticker order.")
    parser.add_argument("--lookback-period", default=None, help="yfinance lookback period, e.g. 1y, 2y, 5y, max.")
    parser.add_argument("--interval", default=None, help="yfinance interval, typically 1d.")
    parser.add_argument("--forecast-days", type=int, default=None, help="Number of calendar days to forecast.")
    parser.add_argument("--num-paths", type=int, default=None, help="Number of Monte Carlo simulation paths.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible simulations.")
    parser.add_argument("--refresh-cache", action="store_true", help="Force refresh of cached market data.")
    args = parser.parse_args()

    config = Config()
    if args.tickers is not None:
        config.tickers = [ticker.upper() for ticker in args.tickers]
    if args.weights is not None:
        config.weights = args.weights
    if args.lookback_period is not None:
        config.lookback_period = args.lookback_period
    if args.interval is not None:
        config.interval = args.interval
    if args.forecast_days is not None:
        config.forecast_days = args.forecast_days
    if args.num_paths is not None:
        config.num_paths = args.num_paths
    if args.seed is not None:
        config.seed = args.seed
    if args.refresh_cache:
        config.refresh_cache = True

    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if len(config.tickers) < 2:
        raise ValueError("Provide at least two tickers for a portfolio simulation.")
    if len(config.weights) != len(config.tickers):
        raise ValueError("The number of weights must exactly match the number of tickers.")
    total_weight = float(sum(config.weights))
    if not np.isclose(total_weight, 1.0, atol=1e-8):
        raise ValueError(f"Portfolio weights must sum to 1.0. Received {total_weight:.8f}.")
    if config.forecast_days < 2:
        raise ValueError("forecast_days must be at least 2.")
    if config.num_paths < 1:
        raise ValueError("num_paths must be positive.")


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_cache_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    modified_ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified_ts).total_seconds() / 3600
    return age_hours < ttl_hours


def extract_close_frame(raw_data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
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

    available = [ticker for ticker in tickers if ticker in price_frame.columns]
    if available:
        price_frame = price_frame.loc[:, available]
    return price_frame.astype(float)


def download_or_load_prices(config: Config) -> tuple[pd.DataFrame, str]:
    ensure_directories()
    if not config.refresh_cache and is_cache_fresh(CACHE_PATH, config.cache_ttl_hours):
        cached_prices = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        return cached_prices, "cache"

    raw_data = yf.download(
        tickers=config.tickers,
        period=config.lookback_period,
        interval=config.interval,
        progress=False,
        auto_adjust=False,
    )
    price_frame = extract_close_frame(raw_data, config.tickers)
    price_frame.to_csv(CACHE_PATH)
    return price_frame, "yfinance"


def compute_log_returns(price_frame: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(price_frame / price_frame.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Log return frame is empty. Check lookback period and input data quality.")
    return log_returns


def simulate_gbm_paths(price_frame: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_returns = compute_log_returns(price_frame)
    s0 = price_frame.iloc[-1].to_numpy(dtype=float)
    num_assets = len(s0)
    weights = np.array(config.weights, dtype=float)

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
    cumulative_log_returns = np.concatenate(
        [np.zeros((1, config.num_paths, num_assets)), np.cumsum(increments, axis=0)],
        axis=0,
    )

    asset_paths = s0[None, None, :] * np.exp(cumulative_log_returns)
    normalized_asset_paths = asset_paths / s0[None, None, :]
    portfolio_paths = (normalized_asset_paths * weights.reshape(1, 1, -1)).sum(axis=2)
    return asset_paths, portfolio_paths, weights, log_returns.to_numpy(dtype=float)


def build_metrics_payload(config: Config, price_frame: pd.DataFrame, portfolio_paths: np.ndarray, weights: np.ndarray, log_return_array: np.ndarray, data_source: str) -> dict:
    terminal_values = portfolio_paths[-1]
    start_value = float(1.0)
    percentiles = np.percentile(terminal_values, [10, 50, 90])
    expected_return_pct = ((terminal_values.mean() / start_value) - 1.0) * 100.0
    probability_of_loss = float(np.mean(terminal_values < start_value))
    annualized_portfolio_volatility = float(np.sqrt(weights.T @ (price_frame.pct_change().dropna().cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR) @ weights))

    return {
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
            "weights": {ticker: float(weight) for ticker, weight in zip(config.tickers, weights)},
            "lookback_period": config.lookback_period,
            "interval": config.interval,
            "observation_count": int(len(price_frame)),
            "start_date": str(price_frame.index.min().date()),
            "end_date": str(price_frame.index.max().date()),
            "latest_prices": {ticker: float(price_frame.iloc[-1][ticker]) for ticker in price_frame.columns},
        },
        "simulation_summary": {
            "starting_portfolio_value": start_value,
            "terminal_value_mean": float(terminal_values.mean()),
            "terminal_value_std": float(terminal_values.std(ddof=1)),
            "terminal_value_p10": float(percentiles[0]),
            "terminal_value_p50": float(percentiles[1]),
            "terminal_value_p90": float(percentiles[2]),
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


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_portfolio_paths(portfolio_paths: np.ndarray, path: Path, num_paths_to_plot: int = 80) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_paths[:, : min(num_paths_to_plot, portfolio_paths.shape[1])], alpha=0.35, linewidth=1.0)
    plt.title("Monte Carlo Simulation – 5-Coin Crypto Portfolio")
    plt.xlabel("Days Ahead")
    plt.ylabel("Normalized Portfolio Value")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_terminal_distribution(portfolio_paths: np.ndarray, path: Path) -> None:
    terminal_values = portfolio_paths[-1]
    plt.figure(figsize=(10, 6))
    plt.hist(terminal_values, bins=50, alpha=0.85)
    plt.title("Terminal Portfolio Value Distribution")
    plt.xlabel("Normalized Terminal Portfolio Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_summary_markdown(payload: dict) -> str:
    summary = payload["simulation_summary"]
    inputs = payload["inputs"]
    lines = [
        "# Crypto Portfolio Monte Carlo Summary",
        "",
        "## Portfolio Configuration",
        f"- Tickers: {', '.join(inputs['tickers'])}",
        f"- Weights: {inputs['weights']}",
        f"- Lookback period: {inputs['lookback_period']}",
        f"- Forecast days: {payload['config']['forecast_days']}",
        f"- Simulation paths: {payload['config']['num_paths']}",
        "",
        "## Simulation Results",
        f"- Starting portfolio value: {summary['starting_portfolio_value']:.4f}",
        f"- 10th percentile terminal value: {summary['terminal_value_p10']:.4f}",
        f"- Median terminal value: {summary['terminal_value_p50']:.4f}",
        f"- 90th percentile terminal value: {summary['terminal_value_p90']:.4f}",
        f"- Expected return: {summary['expected_return_pct']:.2f}%",
        f"- Probability of loss: {summary['probability_of_loss']:.2%}",
        f"- Annualized portfolio volatility: {summary['annualized_portfolio_volatility']:.2%}",
        "",
        "## Notes",
        "- This is a stochastic simulation baseline, not a deterministic prediction engine.",
        "- Results are sensitive to the selected lookback period, covariance estimate, and portfolio weights.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    config = parse_args()
    ensure_directories()
    price_frame, data_source = download_or_load_prices(config)
    _, portfolio_paths, weights, log_return_array = simulate_gbm_paths(price_frame, config)
    payload = build_metrics_payload(config, price_frame, portfolio_paths, weights, log_return_array, data_source)
    save_json(METRICS_PATH, payload)
    SUMMARY_PATH.write_text(build_summary_markdown(payload), encoding="utf-8")
    plot_portfolio_paths(portfolio_paths, PATHS_PLOT_PATH)
    plot_terminal_distribution(portfolio_paths, TERMINAL_DIST_PLOT_PATH)
    summary = payload["simulation_summary"]
    print("------ 5-Coin Crypto Portfolio Projection ------")
    print(f"10th percentile (bear case): {summary['terminal_value_p10']:.4f}")
    print(f"50th percentile (median):    {summary['terminal_value_p50']:.4f}")
    print(f"90th percentile (bull case): {summary['terminal_value_p90']:.4f}")


if __name__ == "__main__":
    main()

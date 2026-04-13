"""
Tests for case-studies/monte-carlo-models/stock-market-prediction/
         scripts/stock_market_monte_carlo.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "stock_mc",
    "case-studies/monte-carlo-models/stock-market-prediction/"
    "scripts/stock_market_monte_carlo.py",
)

Config = _MOD.Config
_load_yaml_config = _MOD._load_yaml_config
is_cache_fresh = _MOD.is_cache_fresh
extract_adjusted_close_frame = _MOD.extract_adjusted_close_frame
compute_log_returns = _MOD.compute_log_returns
simulate_gbm_paths = _MOD.simulate_gbm_paths
build_metrics_payload = _MOD.build_metrics_payload
TRADING_DAYS_PER_YEAR = _MOD.TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_default_tickers(self):
        cfg = Config()
        assert len(cfg.tickers) >= 2

    def test_default_trading_days_per_year(self):
        # Stock MC uses 252 (not 365 like crypto)
        assert TRADING_DAYS_PER_YEAR == 252

    def test_default_forecast_days_is_one_year(self):
        cfg = Config()
        assert cfg.forecast_days == 252

    def test_default_seed_is_set(self):
        cfg = Config()
        assert isinstance(cfg.seed, int)


# ---------------------------------------------------------------------------
# _load_yaml_config (same logic as crypto, just different path)
# ---------------------------------------------------------------------------

class TestLoadYamlConfig:
    def test_missing_file_returns_empty_dict(self):
        result = _load_yaml_config(pytest.importorskip("pathlib").Path("/missing.yaml"))
        assert result == {}

    def test_loads_real_config_if_present(self):
        from tests.conftest import REPO_ROOT
        cfg_path = REPO_ROOT / "configs" / "monte-carlo" / "stock_market_prediction.yaml"
        if cfg_path.exists():
            result = _load_yaml_config(cfg_path)
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# is_cache_fresh (same function, different module import)
# ---------------------------------------------------------------------------

class TestIsCacheFresh:
    def test_false_for_missing_file(self, tmp_path):
        assert not is_cache_fresh(tmp_path / "missing.csv", 24)

    def test_true_for_freshly_created_file(self, tmp_path):
        f = tmp_path / "prices.csv"
        f.write_text("Date,AAPL\n2024-01-01,100.0\n")
        assert is_cache_fresh(f, 24)

    def test_false_for_stale_file(self, tmp_path):
        f = tmp_path / "old.csv"
        f.write_text("Date,AAPL\n2024-01-01,100.0\n")
        old_ts = time.time() - (25 * 3600)
        import os; os.utime(f, (old_ts, old_ts))
        assert not is_cache_fresh(f, 24)


# ---------------------------------------------------------------------------
# extract_close_frame
# ---------------------------------------------------------------------------

class TestExtractAdjustedCloseFrame:
    def _flat_df(self, tickers=("AAPL", "MSFT")):
        idx = pd.date_range("2022-01-01", periods=30)
        return pd.DataFrame(
            {t: np.random.default_rng(i).uniform(100, 500, 30) for i, t in enumerate(tickers)},
            index=idx,
        )

    def test_extracts_columns_by_name(self):
        raw = self._flat_df(["AAPL", "MSFT"])
        result = extract_adjusted_close_frame(raw, ["AAPL", "MSFT"])
        assert "AAPL" in result.columns

    def test_drops_nans(self):
        raw = self._flat_df()
        raw.iloc[5, 0] = np.nan
        result = extract_adjusted_close_frame(raw, ["AAPL", "MSFT"])
        assert not result.isnull().any().any()

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            extract_adjusted_close_frame(pd.DataFrame(), ["AAPL"])


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def _prices(self, n=60, tickers=("AAPL", "MSFT", "NVDA")):
        rng = np.random.default_rng(99)
        data = np.cumprod(1 + rng.uniform(-0.02, 0.02, (n, len(tickers))), axis=0) * 150
        return pd.DataFrame(data, columns=list(tickers))

    def test_length_is_n_minus_one(self):
        pf = self._prices(60)
        lr = compute_log_returns(pf)
        assert len(lr) == 59

    def test_no_nans(self):
        pf = self._prices(40)
        lr = compute_log_returns(pf)
        assert not lr.isnull().any().any()

    def test_single_row_raises(self):
        pf = pd.DataFrame({"AAPL": [150.0], "MSFT": [300.0]})
        with pytest.raises(ValueError):
            compute_log_returns(pf)


# ---------------------------------------------------------------------------
# simulate_gbm_paths
# ---------------------------------------------------------------------------

class TestSimulateGbmPaths:
    def _build_config(self, n_days=15, n_paths=30, seed=7):
        cfg = Config()
        cfg.tickers = ["AAPL", "MSFT", "NVDA"]
        cfg.forecast_days = n_days
        cfg.num_paths = n_paths
        cfg.seed = seed
        return cfg

    def _price_frame(self, n=80):
        rng = np.random.default_rng(3)
        data = np.cumprod(1 + rng.uniform(-0.02, 0.02, (n, 3)), axis=0) * 200
        return pd.DataFrame(data, columns=["AAPL", "MSFT", "NVDA"])

    def test_portfolio_paths_shape(self):
        cfg = self._build_config(n_days=15, n_paths=30)
        pf = self._price_frame()
        _, pp, _, _ = simulate_gbm_paths(pf, cfg)
        assert pp.shape == (16, 30)  # n_days + 1 rows, n_paths cols

    def test_all_paths_start_at_same_value(self):
        # Stock MC uses absolute prices (not normalised), so all paths share the same
        # starting portfolio value (= dot(last_prices, equal_weights))
        cfg = self._build_config(n_days=10, n_paths=20)
        pf = self._price_frame()
        _, pp, weights, _ = simulate_gbm_paths(pf, cfg)
        s0 = pf.iloc[-1].to_numpy(dtype=float)
        expected_start = float(np.dot(s0, weights))
        np.testing.assert_allclose(pp[0], expected_start, rtol=1e-6)

    def test_reproducible_with_seed(self):
        cfg = self._build_config(seed=99)
        pf = self._price_frame()
        _, pp1, _, _ = simulate_gbm_paths(pf, cfg)
        _, pp2, _, _ = simulate_gbm_paths(pf, cfg)
        np.testing.assert_array_equal(pp1, pp2)

    def test_equal_weights_sum_to_one(self):
        cfg = self._build_config()
        pf = self._price_frame()
        _, _, weights, _ = simulate_gbm_paths(pf, cfg)
        # Config() default has 5 equal-weight tickers; but we overrode tickers to 3
        # weights should reflect the per-asset split
        assert pytest.approx(weights.sum(), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# simulate_gbm_paths: equal weights test (stock uses equal-weight by default)
# ---------------------------------------------------------------------------

class TestSimulateGbmPathsEqualWeights:
    def test_equal_weights_sum_to_one(self):
        cfg = Config()
        cfg.tickers = ["AAPL", "MSFT", "NVDA"]
        cfg.forecast_days = 10
        cfg.num_paths = 20
        cfg.seed = 42
        rng = np.random.default_rng(0)
        n = 60
        data = np.cumprod(1 + rng.uniform(-0.02, 0.02, (n, 3)), axis=0) * 200
        pf = pd.DataFrame(data, columns=["AAPL", "MSFT", "NVDA"])
        _, _, weights, _ = simulate_gbm_paths(pf, cfg)
        assert pytest.approx(weights.sum(), abs=1e-6) == 1.0

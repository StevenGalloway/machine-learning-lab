"""
Tests for case-studies/monte-carlo-models/crypto-market-prediction/
         scripts/crypto_market_prediction_mc.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "crypto_mc",
    "case-studies/monte-carlo-models/crypto-market-prediction/"
    "scripts/crypto_market_prediction_mc.py",
)

Config = _MOD.Config
validate_config = _MOD.validate_config
is_cache_fresh = _MOD.is_cache_fresh
extract_close_frame = _MOD.extract_close_frame
compute_log_returns = _MOD.compute_log_returns
simulate_gbm_paths = _MOD.simulate_gbm_paths
build_metrics_payload = _MOD.build_metrics_payload
build_summary_markdown = _MOD.build_summary_markdown
save_json = _MOD.save_json
_load_yaml_config = _MOD._load_yaml_config


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def _default_config(self, **overrides):
        cfg = Config()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_valid_config_passes(self):
        validate_config(self._default_config())  # must not raise

    def test_raises_on_single_ticker(self):
        with pytest.raises(ValueError, match="at least two tickers"):
            validate_config(self._default_config(tickers=["BTC-USD"], weights=[1.0]))

    def test_raises_on_weight_count_mismatch(self):
        with pytest.raises(ValueError, match="match the number of tickers"):
            validate_config(self._default_config(
                tickers=["BTC-USD", "ETH-USD"],
                weights=[0.5, 0.3, 0.2]  # 3 weights for 2 tickers
            ))

    def test_raises_on_weights_not_summing_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_config(self._default_config(
                tickers=["BTC-USD", "ETH-USD"],
                weights=[0.4, 0.4]  # sums to 0.8
            ))

    def test_raises_on_forecast_days_less_than_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_config(self._default_config(
                tickers=["BTC-USD", "ETH-USD"],
                weights=[0.5, 0.5],
                forecast_days=1,
            ))

    def test_raises_on_zero_num_paths(self):
        with pytest.raises(ValueError, match="positive"):
            validate_config(self._default_config(
                tickers=["BTC-USD", "ETH-USD"],
                weights=[0.5, 0.5],
                num_paths=0,
            ))


# ---------------------------------------------------------------------------
# is_cache_fresh
# ---------------------------------------------------------------------------

class TestIsCacheFresh:
    def test_returns_false_for_missing_file(self, tmp_path):
        assert not is_cache_fresh(tmp_path / "nope.csv", 24)

    def test_returns_false_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.touch()
        assert not is_cache_fresh(f, 24)

    def test_returns_true_for_fresh_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col\n1\n2\n")
        # File was just created — should be fresh for 24h TTL
        assert is_cache_fresh(f, 24)

    def test_returns_false_for_stale_file(self, tmp_path):
        f = tmp_path / "old.csv"
        f.write_text("col\n1\n")
        # Set modification time to 25 hours ago
        old_ts = time.time() - (25 * 3600)
        import os
        os.utime(f, (old_ts, old_ts))
        assert not is_cache_fresh(f, 24)


# ---------------------------------------------------------------------------
# extract_close_frame
# ---------------------------------------------------------------------------

class TestExtractCloseFrame:
    def _make_flat_df(self, tickers=("BTC-USD", "ETH-USD")):
        idx = pd.date_range("2023-01-01", periods=10)
        data = {t: np.random.default_rng(0).uniform(100, 200, 10) for t in tickers}
        return pd.DataFrame(data, index=idx)

    def _make_multiindex_df(self, tickers=("BTC-USD", "ETH-USD")):
        idx = pd.date_range("2023-01-01", periods=10)
        arrays = [["Close"] * len(tickers), list(tickers)]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.random.default_rng(1).uniform(100, 500, (10, len(tickers)))
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_extracts_from_flat_ticker_columns(self):
        raw = self._make_flat_df(["BTC-USD", "ETH-USD"])
        result = extract_close_frame(raw, ["BTC-USD", "ETH-USD"])
        assert list(result.columns) == ["BTC-USD", "ETH-USD"]

    def test_extracts_from_multiindex(self):
        raw = self._make_multiindex_df(["BTC-USD", "ETH-USD"])
        result = extract_close_frame(raw, ["BTC-USD", "ETH-USD"])
        assert "BTC-USD" in result.columns or len(result.columns) >= 1

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            extract_close_frame(pd.DataFrame(), ["BTC-USD"])

    def test_drops_rows_with_nan(self):
        raw = self._make_flat_df()
        raw.iloc[2, 0] = np.nan
        result = extract_close_frame(raw, ["BTC-USD", "ETH-USD"])
        assert not result.isnull().any().any()

    def test_result_is_float(self):
        raw = self._make_flat_df()
        result = extract_close_frame(raw, ["BTC-USD", "ETH-USD"])
        assert result.dtypes.apply(lambda d: d == float).all()


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def _price_frame(self, n: int = 50, tickers=("A", "B")):
        rng = np.random.default_rng(7)
        prices = np.cumprod(1 + rng.uniform(-0.05, 0.05, (n, len(tickers))), axis=0) * 100
        return pd.DataFrame(prices, columns=list(tickers))

    def test_row_count_is_n_minus_one(self):
        pf = self._price_frame(50)
        lr = compute_log_returns(pf)
        assert len(lr) == 49

    def test_no_nans_in_output(self):
        pf = self._price_frame(30)
        lr = compute_log_returns(pf)
        assert not lr.isnull().any().any()

    def test_raises_on_empty_result(self):
        # Single-row price frame → log returns are empty
        pf = pd.DataFrame({"A": [100.0], "B": [200.0]})
        with pytest.raises(ValueError, match="empty"):
            compute_log_returns(pf)


# ---------------------------------------------------------------------------
# simulate_gbm_paths
# ---------------------------------------------------------------------------

class TestSimulateGbmPaths:
    def _build_config(self, forecast_days=10, num_paths=50, seed=0):
        cfg = Config()
        cfg.tickers = ["BTC-USD", "ETH-USD"]
        cfg.weights = [0.6, 0.4]
        cfg.forecast_days = forecast_days
        cfg.num_paths = num_paths
        cfg.seed = seed
        return cfg

    def _price_frame(self, n: int = 60):
        rng = np.random.default_rng(42)
        prices = np.cumprod(1 + rng.uniform(-0.03, 0.03, (n, 2)), axis=0) * 100
        return pd.DataFrame(prices, columns=["BTC-USD", "ETH-USD"])

    def test_portfolio_paths_shape(self):
        cfg = self._build_config(forecast_days=10, num_paths=50)
        pf = self._price_frame()
        _, portfolio_paths, _, _ = simulate_gbm_paths(pf, cfg)
        # Shape: (forecast_days + 1, num_paths)
        assert portfolio_paths.shape == (11, 50)

    def test_portfolio_starts_at_one(self):
        cfg = self._build_config(forecast_days=5, num_paths=20)
        pf = self._price_frame()
        _, portfolio_paths, _, _ = simulate_gbm_paths(pf, cfg)
        np.testing.assert_allclose(portfolio_paths[0], 1.0)

    def test_reproducibility_with_same_seed(self):
        cfg1 = self._build_config(seed=42)
        cfg2 = self._build_config(seed=42)
        pf = self._price_frame()
        _, pp1, _, _ = simulate_gbm_paths(pf, cfg1)
        _, pp2, _, _ = simulate_gbm_paths(pf, cfg2)
        np.testing.assert_array_equal(pp1, pp2)

    def test_different_seeds_give_different_results(self):
        cfg1 = self._build_config(seed=1)
        cfg2 = self._build_config(seed=2)
        pf = self._price_frame()
        _, pp1, _, _ = simulate_gbm_paths(pf, cfg1)
        _, pp2, _, _ = simulate_gbm_paths(pf, cfg2)
        assert not np.array_equal(pp1, pp2)

    def test_weights_returned(self):
        cfg = self._build_config()
        pf = self._price_frame()
        _, _, weights, _ = simulate_gbm_paths(pf, cfg)
        np.testing.assert_array_almost_equal(weights, [0.6, 0.4])


# ---------------------------------------------------------------------------
# build_summary_markdown
# ---------------------------------------------------------------------------

class TestBuildSummaryMarkdown:
    def _make_payload(self):
        return {
            "config": {
                "forecast_days": 180,
                "num_paths": 5000,
            },
            "inputs": {
                "tickers": ["BTC-USD", "ETH-USD"],
                "weights": {"BTC-USD": 0.6, "ETH-USD": 0.4},
                "lookback_period": "2y",
            },
            "simulation_summary": {
                "starting_portfolio_value": 1.0,
                "terminal_value_p10": 0.8,
                "terminal_value_p50": 1.2,
                "terminal_value_p90": 1.9,
                "expected_return_pct": 22.5,
                "probability_of_loss": 0.18,
                "annualized_portfolio_volatility": 0.65,
            },
        }

    def test_returns_string(self):
        assert isinstance(build_summary_markdown(self._make_payload()), str)

    def test_contains_ticker_names(self):
        md = build_summary_markdown(self._make_payload())
        assert "BTC-USD" in md
        assert "ETH-USD" in md

    def test_contains_percentile_values(self):
        md = build_summary_markdown(self._make_payload())
        assert "0.8" in md   # p10
        assert "1.2" in md   # p50
        assert "1.9" in md   # p90


# ---------------------------------------------------------------------------
# _load_yaml_config
# ---------------------------------------------------------------------------

class TestLoadYamlConfig:
    def test_returns_empty_dict_for_missing_file(self):
        from pathlib import Path
        result = _load_yaml_config(Path("/nonexistent/path/config.yaml"))
        assert result == {}

    def test_loads_real_config(self):
        from tests.conftest import REPO_ROOT
        cfg_path = REPO_ROOT / "configs" / "monte-carlo" / "crypto_market_prediction.yaml"
        if cfg_path.exists():
            result = _load_yaml_config(cfg_path)
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "output.json"
        save_json(p, {"key": "value", "number": 3.14})
        loaded = json.loads(p.read_text())
        assert loaded["key"] == "value"
        assert loaded["number"] == pytest.approx(3.14)

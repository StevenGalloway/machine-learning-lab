"""
Tests for case-studies/linear-regression-models/football-points-prediction/
         scripts/points_prediction_linear_reg.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "football_lr",
    "case-studies/linear-regression-models/football-points-prediction/"
    "scripts/points_prediction_linear_reg.py",
)

smape = _MOD.smape
validate_schema = _MOD.validate_schema
compute_metrics = _MOD.compute_metrics
save_metrics_json = _MOD.save_metrics_json
RegressionMetrics = _MOD.RegressionMetrics


# ---------------------------------------------------------------------------
# smape
# ---------------------------------------------------------------------------

class TestSmape:
    def test_perfect_is_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # 2 * |20-10| / (10+20) = 20/30
        assert smape(np.array([10.0]), np.array([20.0])) == pytest.approx(2 * 10.0 / 30.0, abs=1e-6)

    def test_returns_float(self):
        assert isinstance(smape(np.array([1.0, 2.0]), np.array([3.0, 4.0])), float)

    def test_bounded_between_zero_and_two(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(1, 100, 50)
        y_pred = rng.uniform(1, 100, 50)
        result = smape(y_true, y_pred)
        assert 0.0 <= result <= 2.0


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_passes_valid_df(self, synthetic_football_df):
        validate_schema(synthetic_football_df)  # must not raise

    def test_raises_missing_feature_col(self, synthetic_football_df):
        df = synthetic_football_df.drop(columns=["FirstDowns"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_raises_missing_target_col(self, synthetic_football_df):
        df = synthetic_football_df.drop(columns=["Points"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_raises_non_numeric_feature(self):
        # Use numpy object array — avoids pandas StringDtype which np.issubdtype cannot handle
        df = pd.DataFrame({
            "FirstDowns": np.array(["a", "b", "c"], dtype=object),
            "Points": [10.0, 20.0, 30.0],
        })
        with pytest.raises((ValueError, TypeError)):
            validate_schema(df)

    def test_raises_non_numeric_target(self):
        df = pd.DataFrame({
            "FirstDowns": [10.0, 20.0],
            "Points": np.array(["x", "y"], dtype=object),
        })
        with pytest.raises((ValueError, TypeError)):
            validate_schema(df)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.arange(1, 11, dtype=float)
        m = compute_metrics(y, y)
        assert m.r2 == pytest.approx(1.0)
        assert m.mae == pytest.approx(0.0)
        assert m.rmse == pytest.approx(0.0)

    def test_n_test_is_correct(self):
        y = np.ones(15)
        m = compute_metrics(y, y + 1.0)
        assert m.n_test == 15

    def test_rmse_is_sqrt_mse(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([3.0, 5.0, 7.0])
        m = compute_metrics(y_true, y_pred)
        assert m.rmse == pytest.approx(math.sqrt(m.mse), rel=1e-6)

    def test_returns_regression_metrics(self):
        y = np.array([1.0, 2.0])
        assert isinstance(compute_metrics(y, y + 0.5), RegressionMetrics)

    def test_to_dict_has_all_keys(self):
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        d = m.to_dict()
        for key in ("mse", "rmse", "mae", "r2", "smape", "n_test"):
            assert key in d

    def test_to_dict_json_serialisable(self):
        m = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.5]))
        json.dumps(m.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# save_metrics_json
# ---------------------------------------------------------------------------

class TestSaveMetricsJson:
    def test_creates_file_at_path(self, tmp_path):
        p = tmp_path / "metrics_nfl.json"
        save_metrics_json(p, {"dataset": "nfl", "r2": 0.72})
        assert p.exists()

    def test_file_contains_correct_data(self, tmp_path):
        p = tmp_path / "m.json"
        payload = {"mae": 3.14, "n_test": 200}
        save_metrics_json(p, payload)
        assert json.loads(p.read_text()) == payload

    def test_creates_nested_directories(self, tmp_path):
        p = tmp_path / "a" / "b" / "metrics.json"
        save_metrics_json(p, {})
        assert p.exists()

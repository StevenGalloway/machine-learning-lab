"""
Tests for case-studies/linear-regression-models/basketball-points-prediction/
         scripts/points_prediction_linear_reg.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module, REPO_ROOT

# ---------------------------------------------------------------------------
# Load the module under test
# ---------------------------------------------------------------------------
_MOD = load_module(
    "bball_lr",
    "case-studies/linear-regression-models/basketball-points-prediction/"
    "scripts/points_prediction_linear_reg.py",
)

mp_to_minutes = _MOD.mp_to_minutes
smape = _MOD.smape
validate_schema = _MOD.validate_schema
make_features_and_target = _MOD.make_features_and_target
time_ordered_split = _MOD.time_ordered_split
compute_metrics = _MOD.compute_metrics
save_metrics_json = _MOD.save_metrics_json
RegressionMetrics = _MOD.RegressionMetrics


# ---------------------------------------------------------------------------
# mp_to_minutes
# ---------------------------------------------------------------------------

class TestMpToMinutes:
    def test_mm_ss_format(self):
        # "25.30" → 25 + 30/60 = 25.5
        assert mp_to_minutes("25.30") == pytest.approx(25.5)

    def test_mm_ss_zeros(self):
        assert mp_to_minutes("10.00") == pytest.approx(10.0)

    def test_integer_string(self):
        assert mp_to_minutes("30") == pytest.approx(30.0)

    def test_float_no_seconds(self):
        # "20.0" → 20 + 0/60 = 20.0
        assert mp_to_minutes("20.0") == pytest.approx(20.0)

    def test_invalid_returns_nan(self):
        result = mp_to_minutes("abc.xyz")
        assert math.isnan(result)

    def test_empty_string_returns_nan(self):
        result = mp_to_minutes("")
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# smape
# ---------------------------------------------------------------------------

class TestSmape:
    def test_perfect_prediction_is_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # smape([10], [20]) = 2 * |20-10| / (10+20) = 20/30 ≈ 0.6667
        result = smape(np.array([10.0]), np.array([20.0]))
        assert result == pytest.approx(2 * 10.0 / 30.0, abs=1e-6)

    def test_symmetry(self):
        y_true = np.array([5.0, 15.0, 25.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        # smape is symmetric by construction when swapped
        r1 = smape(y_true, y_pred)
        r2 = smape(y_pred, y_true)
        assert r1 == pytest.approx(r2, abs=1e-6)

    def test_returns_float(self):
        assert isinstance(smape(np.array([1.0]), np.array([2.0])), float)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_passes_with_all_required_columns(self, synthetic_basketball_df):
        # Should not raise
        validate_schema(synthetic_basketball_df)

    def test_raises_on_missing_column(self, synthetic_basketball_df):
        df = synthetic_basketball_df.drop(columns=["PTS"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_raises_on_multiple_missing(self, synthetic_basketball_df):
        df = synthetic_basketball_df.drop(columns=["PTS", "FGA"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)


# ---------------------------------------------------------------------------
# make_features_and_target
# ---------------------------------------------------------------------------

class TestMakeFeaturesAndTarget:
    def test_creates_target_column(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        assert "PTS_next" in df.columns

    def test_drops_last_row_per_player(self, synthetic_basketball_df):
        # Each player's last game has no "next" → those rows are dropped
        original_n = len(synthetic_basketball_df)
        df = make_features_and_target(synthetic_basketball_df)
        assert len(df) < original_n

    def test_min_float_column_created(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        assert "MIN_float" in df.columns

    def test_min_float_non_negative(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        assert (df["MIN_float"] >= 0).all()

    def test_target_is_float(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        assert df["PTS_next"].dtype == float


# ---------------------------------------------------------------------------
# time_ordered_split
# ---------------------------------------------------------------------------

class TestTimeOrderedSplit:
    def test_split_ratio(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        train, test = time_ordered_split(df)
        total = len(train) + len(test)
        assert total == len(df)
        # test set should be ~20% (default TEST_SIZE)
        assert 0.10 <= len(test) / total <= 0.35

    def test_train_before_test_by_date(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        train, test = time_ordered_split(df)
        max_train_date = pd.to_datetime(train["Date"]).max()
        min_test_date = pd.to_datetime(test["Date"]).min()
        assert max_train_date <= min_test_date

    def test_no_overlap(self, synthetic_basketball_df):
        df = make_features_and_target(synthetic_basketball_df)
        train, test = time_ordered_split(df)
        shared = set(train.index) & set(test.index)
        assert len(shared) == 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        m = compute_metrics(y, y)
        assert m.r2 == pytest.approx(1.0)
        assert m.mae == pytest.approx(0.0)
        assert m.rmse == pytest.approx(0.0)
        assert m.mse == pytest.approx(0.0)

    def test_returns_regression_metrics(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y + 1)
        assert isinstance(m, RegressionMetrics)

    def test_n_test_matches_length(self):
        y = np.arange(10, dtype=float)
        m = compute_metrics(y, y + 0.5)
        assert m.n_test == 10

    def test_rmse_equals_sqrt_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        m = compute_metrics(y_true, y_pred)
        assert m.rmse == pytest.approx(math.sqrt(m.mse))


# ---------------------------------------------------------------------------
# RegressionMetrics.to_dict
# ---------------------------------------------------------------------------

class TestRegressionMetricsToDict:
    def test_all_keys_present(self):
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        d = m.to_dict()
        for key in ("mae", "mse", "rmse", "r2", "smape", "n_test"):
            assert key in d

    def test_values_are_serialisable(self):
        m = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 3.5]))
        d = m.to_dict()
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# save_metrics_json
# ---------------------------------------------------------------------------

class TestSaveMetricsJson:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "metrics.json"
        save_metrics_json(path, {"test": 42})
        assert path.exists()

    def test_content_is_valid_json(self, tmp_path):
        path = tmp_path / "metrics.json"
        payload = {"mae": 1.23, "n": 100}
        save_metrics_json(path, payload)
        loaded = json.loads(path.read_text())
        assert loaded == payload

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "metrics.json"
        save_metrics_json(path, {"k": "v"})
        assert path.exists()

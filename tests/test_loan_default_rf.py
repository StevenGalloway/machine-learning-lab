"""
Tests for case-studies/random-forest-models/loan-default-prediction/
         scripts/loan_default_random_forest.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "loan_default_rf",
    "case-studies/random-forest-models/loan-default-prediction/"
    "scripts/loan_default_random_forest.py",
)

threshold_metrics = _MOD.threshold_metrics
choose_cost_aware_threshold = _MOD.choose_cost_aware_threshold
get_rf_feature_importances = _MOD.get_rf_feature_importances
_load_config = _MOD._load_config


# ---------------------------------------------------------------------------
# threshold_metrics
# ---------------------------------------------------------------------------

class TestThresholdMetrics:
    def _make_perfect(self, n: int = 100):
        """y_true == pred at threshold 0.5."""
        y_true = np.array([0] * (n // 2) + [1] * (n // 2))
        prob = np.array([0.1] * (n // 2) + [0.9] * (n // 2))
        return y_true, prob

    def test_perfect_predictions_zero_fnr_fpr(self):
        y_true, prob = self._make_perfect()
        m = threshold_metrics(y_true, prob, 0.5)
        assert m["fnr"] == pytest.approx(0.0)
        assert m["fpr"] == pytest.approx(0.0)

    def test_perfect_predictions_accuracy_one(self):
        y_true, prob = self._make_perfect()
        m = threshold_metrics(y_true, prob, 0.5)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_all_predicted_positive_fpr_one(self):
        y_true = np.array([0, 0, 0, 1, 1])
        prob = np.ones(5)
        m = threshold_metrics(y_true, prob, 0.5)
        # All negatives flagged → FPR = 1.0
        assert m["fpr"] == pytest.approx(1.0)
        # No false negatives → FNR = 0.0
        assert m["fnr"] == pytest.approx(0.0)

    def test_confusion_matrix_counts_sum_to_n(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        prob = rng.uniform(0, 1, 200)
        m = threshold_metrics(y_true, prob, 0.5)
        total = m["tn"] + m["fp"] + m["fn"] + m["tp"]
        assert total == 200

    def test_threshold_key_is_passed_through(self):
        y_true = np.array([0, 1])
        prob = np.array([0.3, 0.7])
        m = threshold_metrics(y_true, prob, 0.42)
        assert m["threshold"] == pytest.approx(0.42)

    def test_fnr_fpr_bounded_zero_to_one(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 150)
        prob = rng.uniform(0, 1, 150)
        for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            m = threshold_metrics(y_true, prob, thr)
            assert 0.0 <= m["fnr"] <= 1.0
            assert 0.0 <= m["fpr"] <= 1.0


# ---------------------------------------------------------------------------
# choose_cost_aware_threshold
# ---------------------------------------------------------------------------

class TestChooseCostAwareThreshold:
    def _make_separable(self, n: int = 200):
        y_true = np.array([0] * (n // 2) + [1] * (n // 2))
        prob = np.concatenate([
            np.random.default_rng(0).uniform(0.05, 0.40, n // 2),
            np.random.default_rng(1).uniform(0.60, 0.95, n // 2),
        ])
        return y_true, prob

    def test_returns_dict_with_threshold_key(self):
        y, p = self._make_separable()
        best = choose_cost_aware_threshold(y, p, max_fpr=0.35)
        assert "threshold" in best

    def test_respects_max_fpr_constraint(self):
        y, p = self._make_separable()
        best = choose_cost_aware_threshold(y, p, max_fpr=0.35)
        assert best["fpr"] <= 0.35 + 1e-6  # small tolerance for grid search

    def test_threshold_in_valid_range(self):
        y, p = self._make_separable()
        best = choose_cost_aware_threshold(y, p, max_fpr=0.5)
        assert 0.0 <= best["threshold"] <= 1.0

    def test_fallback_when_constraint_impossible(self):
        # Very strict max_fpr=0.0 → no threshold can satisfy it → falls back to min FNR
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(0, 1, 100)
        best = choose_cost_aware_threshold(y, p, max_fpr=0.0)
        assert "threshold" in best


# ---------------------------------------------------------------------------
# get_rf_feature_importances
# ---------------------------------------------------------------------------

class TestGetRfFeatureImportances:
    def test_direct_rf_with_feature_importances(self):
        mock_clf = MagicMock()
        mock_clf.feature_importances_ = np.array([0.3, 0.4, 0.3])
        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {"clf": mock_clf}

        result = get_rf_feature_importances(mock_pipeline)
        np.testing.assert_array_equal(result, np.array([0.3, 0.4, 0.3]))

    def test_calibrated_wrapper_averages_importances(self):
        imp1 = np.array([0.2, 0.5, 0.3])
        imp2 = np.array([0.4, 0.3, 0.3])

        mock_est1 = MagicMock()
        mock_est1.feature_importances_ = imp1
        mock_cc1 = MagicMock()
        mock_cc1.estimator = mock_est1

        mock_est2 = MagicMock()
        mock_est2.feature_importances_ = imp2
        mock_cc2 = MagicMock()
        mock_cc2.estimator = mock_est2

        mock_clf = MagicMock(spec=[])   # no feature_importances_ attribute
        mock_clf.calibrated_classifiers_ = [mock_cc1, mock_cc2]
        # Ensure hasattr(clf, 'feature_importances_') is False
        del mock_clf.feature_importances_

        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {"clf": mock_clf}

        result = get_rf_feature_importances(mock_pipeline)
        expected = (imp1 + imp2) / 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_raises_when_no_importances_found(self):
        mock_clf = MagicMock(spec=[])
        mock_clf.calibrated_classifiers_ = []
        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {"clf": mock_clf}
        with pytest.raises(AttributeError):
            get_rf_feature_importances(mock_pipeline)


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self):
        cfg = _load_config()
        assert isinstance(cfg, dict)

    def test_has_required_keys(self):
        cfg = _load_config()
        for key in ("random_state", "test_size", "max_fpr", "baseline", "random_forest"):
            assert key in cfg

    def test_random_forest_sub_keys(self):
        cfg = _load_config()
        rf = cfg["random_forest"]
        for k in ("n_estimators", "class_weight", "calibration_method"):
            assert k in rf

    def test_baseline_sub_keys(self):
        cfg = _load_config()
        bl = cfg["baseline"]
        assert "max_iter" in bl
        assert "solver" in bl

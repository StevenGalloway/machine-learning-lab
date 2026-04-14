"""
Tests for case-studies/xgb-models/loan-approval/scripts/loan_approval_model.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "loan_approval_xgb",
    "case-studies/xgb-models/loan-approval/scripts/loan_approval_model.py",
)

validate_schema = _MOD.validate_schema
compute_metrics = _MOD.compute_metrics
choose_threshold_by_precision = _MOD.choose_threshold_by_precision
save_metrics_json = _MOD.save_metrics_json
Metrics = _MOD.Metrics
FEATURE_COLS = _MOD.FEATURE_COLS
TARGET_COL = _MOD.TARGET_COL


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_loan_df():
    rng = np.random.default_rng(10)
    n = 50
    return pd.DataFrame({
        "Age": rng.integers(20, 65, n).astype(float),
        "Income": rng.uniform(20000, 150000, n),
        "LoanAmount": rng.uniform(1000, 50000, n),
        "CreditScore": rng.integers(500, 850, n).astype(float),
        "Approved": rng.integers(0, 2, n),
    })


class TestValidateSchema:
    def test_passes_with_valid_df(self, valid_loan_df):
        validate_schema(valid_loan_df)  # must not raise

    def test_raises_on_missing_feature(self, valid_loan_df):
        df = valid_loan_df.drop(columns=["Age"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_raises_on_missing_target(self, valid_loan_df):
        df = valid_loan_df.drop(columns=["Approved"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_raises_on_non_numeric_feature(self, valid_loan_df):
        df = valid_loan_df.copy()
        # Use numpy object array — avoids pandas StringDtype which np.issubdtype cannot handle
        df["Age"] = np.array(["twenty"] * len(df), dtype=object)
        with pytest.raises((ValueError, TypeError)):
            validate_schema(df)

    def test_raises_on_non_binary_target(self, valid_loan_df):
        df = valid_loan_df.copy()
        df["Approved"] = np.arange(len(df))  # not binary
        with pytest.raises(ValueError, match="must be binary"):
            validate_schema(df)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_arrays():
    """Simple arrays with known confusion matrix."""
    # 4 TP, 2 FP, 1 FN, 3 TN → total 10
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    # Arbitrary probs for AUC (must be ordinal)
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.3, 0.2, 0.4, 0.1])
    return y_true, y_pred, y_prob


class TestComputeMetrics:
    def test_returns_metrics_dataclass(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        assert isinstance(m, Metrics)

    def test_confusion_matrix_correct(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        # tp=4, fp=2, fn=1, tn=3
        assert m.tp == 4
        assert m.fp == 2
        assert m.fn == 1
        assert m.tn == 3

    def test_accuracy_correct(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        # (4 + 3) / 10 = 0.7
        assert m.accuracy == pytest.approx(0.7)

    def test_specificity_correct(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        # tn / (tn + fp) = 3 / (3 + 2) = 0.6
        assert m.specificity == pytest.approx(0.6)

    def test_support_test_is_n(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.support_test == 10

    def test_positive_rate_test(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.positive_rate_test == pytest.approx(0.5)

    def test_roc_auc_bounded(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m.roc_auc <= 1.0

    def test_brier_bounded(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        m = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m.brier <= 1.0


# ---------------------------------------------------------------------------
# Metrics.to_dict
# ---------------------------------------------------------------------------

class TestMetricsToDict:
    def test_all_keys_present(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        d = compute_metrics(y_true, y_pred, y_prob).to_dict()
        for k in ("tn", "fp", "fn", "tp", "accuracy", "roc_auc",
                  "precision_ppv", "recall_sensitivity", "specificity",
                  "f1", "brier", "support_test", "positive_rate_test"):
            assert k in d

    def test_json_serialisable(self, binary_arrays):
        y_true, y_pred, y_prob = binary_arrays
        json.dumps(compute_metrics(y_true, y_pred, y_prob).to_dict())


# ---------------------------------------------------------------------------
# choose_threshold_by_precision
# ---------------------------------------------------------------------------

class TestChooseThresholdByPrecision:
    def _separable_arrays(self, n: int = 300):
        rng = np.random.default_rng(42)
        y_true = np.array([0] * (n // 2) + [1] * (n // 2))
        prob = np.concatenate([
            rng.uniform(0.05, 0.40, n // 2),
            rng.uniform(0.55, 0.95, n // 2),
        ])
        return y_true, prob

    def test_returns_float(self):
        y, p = self._separable_arrays()
        thr = choose_threshold_by_precision(y, p, min_precision=0.90)
        assert isinstance(thr, float)

    def test_threshold_in_valid_range(self):
        y, p = self._separable_arrays()
        thr = choose_threshold_by_precision(y, p, min_precision=0.90)
        assert 0.0 <= thr <= 1.0

    def test_achieves_precision_floor(self):
        from sklearn.metrics import precision_score
        y, p = self._separable_arrays()
        min_prec = 0.80
        thr = choose_threshold_by_precision(y, p, min_precision=min_prec)
        pred = (p >= thr).astype(int)
        prec = precision_score(y, pred, zero_division=0)
        assert prec >= min_prec - 0.05  # small tolerance

    def test_fallback_returns_valid_threshold(self):
        # min_precision=0.999 is impossible → falls back to F1 maximiser
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(0, 1, 100)
        thr = choose_threshold_by_precision(y, p, min_precision=0.999)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0


# ---------------------------------------------------------------------------
# save_metrics_json
# ---------------------------------------------------------------------------

class TestSaveMetricsJson:
    def test_creates_file_with_content(self, tmp_path):
        p = tmp_path / "metrics.json"
        save_metrics_json(p, {"model": "xgb", "auc": 0.95})
        assert p.exists()
        assert json.loads(p.read_text())["auc"] == pytest.approx(0.95)

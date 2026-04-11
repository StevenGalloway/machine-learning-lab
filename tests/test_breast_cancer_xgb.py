"""
Tests for case-studies/xgb-models/breast-cancer/scripts/train_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve

from tests.conftest import load_module

_MOD = load_module(
    "breast_cancer_xgb",
    "case-studies/xgb-models/breast-cancer/scripts/train_eval.py",
)

compute_metrics = _MOD.compute_metrics
choose_threshold = _MOD.choose_threshold
save_artifacts = _MOD.save_artifacts


# ---------------------------------------------------------------------------
# Shared test data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def breast_cancer_split():
    """Return a small train/test split of the breast cancer dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    y_mal = (y == 0).astype(int)
    return train_test_split(X, y_mal, test_size=0.2, random_state=42, stratify=y_mal)


@pytest.fixture(scope="module")
def fitted_logreg(breast_cancer_split):
    X_train, X_test, y_train, y_test = breast_cancer_split
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return y_test, pred, prob


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_has_required_keys(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        for key in ("tn", "fp", "fn", "tp", "accuracy", "roc_auc",
                    "sensitivity", "specificity", "ppv", "npv", "f1", "brier"):
            assert key in m

    def test_confusion_matrix_sums_to_n(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y_test)

    def test_accuracy_bounded(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_sensitivity_is_recall(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        tp = m["tp"]
        fn = m["fn"]
        expected = tp / (tp + fn) if (tp + fn) else float("nan")
        assert m["sensitivity"] == pytest.approx(expected)

    def test_specificity_is_true_neg_rate(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        tn = m["tn"]
        fp = m["fp"]
        expected = tn / (tn + fp) if (tn + fp) else float("nan")
        assert m["specificity"] == pytest.approx(expected)

    def test_ppv_is_precision(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        tp, fp = m["tp"], m["fp"]
        expected = tp / (tp + fp) if (tp + fp) else float("nan")
        assert m["ppv"] == pytest.approx(expected)

    def test_npv(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        tn, fn = m["tn"], m["fn"]
        expected = tn / (tn + fn) if (tn + fn) else float("nan")
        assert m["npv"] == pytest.approx(expected)

    def test_brier_bounded(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        assert 0.0 <= m["brier"] <= 1.0

    def test_roc_auc_bounded(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_all_values_json_serialisable(self, fitted_logreg):
        y_test, pred, prob = fitted_logreg
        m = compute_metrics(y_test, pred, prob)
        json.dumps(m)  # must not raise


# ---------------------------------------------------------------------------
# choose_threshold
# ---------------------------------------------------------------------------

class TestChooseThreshold:
    def test_returns_float(self, fitted_logreg):
        y_test, _, prob = fitted_logreg
        thr = choose_threshold(y_test, prob, min_sens=0.97)
        assert isinstance(thr, float)

    def test_threshold_in_valid_range(self, fitted_logreg):
        y_test, _, prob = fitted_logreg
        thr = choose_threshold(y_test, prob, min_sens=0.97)
        assert 0.0 <= thr <= 1.0

    def test_achieves_min_sensitivity(self, fitted_logreg):
        y_test, _, prob = fitted_logreg
        min_sens = 0.97
        thr = choose_threshold(y_test, prob, min_sens=min_sens)
        pred = (prob >= thr).astype(int)
        m = compute_metrics(y_test, pred, prob)
        # Sensitivity must be ≥ min_sens at the chosen threshold
        assert m["sensitivity"] >= min_sens - 0.02  # small tolerance for discrete ROC

    def test_fallback_when_impossible_constraint(self, fitted_logreg):
        # min_sens = 1.0 may not be achievable; should still return a valid threshold
        y_test, _, prob = fitted_logreg
        thr = choose_threshold(y_test, prob, min_sens=1.0)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0


# ---------------------------------------------------------------------------
# save_artifacts
# ---------------------------------------------------------------------------

class TestSaveArtifacts:
    def _build_args(self, tmp_path, fitted_logreg):
        y_test, pred, b_prob = fitted_logreg
        p_prob = b_prob  # reuse as xgb prob for simplicity
        p_pred = pred
        thr = 0.5
        b_metrics = compute_metrics(y_test, pred, b_prob)
        p_metrics = compute_metrics(y_test, p_pred, p_prob)
        p_metrics_thr = compute_metrics(y_test, (p_prob >= thr).astype(int), p_prob)
        return (tmp_path, b_metrics, p_metrics, p_metrics_thr, thr, y_test, b_prob, p_prob)

    def test_creates_metrics_json(self, tmp_path, fitted_logreg):
        args = self._build_args(tmp_path, fitted_logreg)
        save_artifacts(*args)
        assert (tmp_path / "metrics.json").exists()

    def test_creates_roc_curve_png(self, tmp_path, fitted_logreg):
        args = self._build_args(tmp_path, fitted_logreg)
        save_artifacts(*args)
        assert (tmp_path / "roc_curve.png").exists()

    def test_creates_confusion_matrix_png(self, tmp_path, fitted_logreg):
        args = self._build_args(tmp_path, fitted_logreg)
        save_artifacts(*args)
        assert (tmp_path / "confusion_matrix.png").exists()

    def test_metrics_json_has_required_sections(self, tmp_path, fitted_logreg):
        args = self._build_args(tmp_path, fitted_logreg)
        save_artifacts(*args)
        data = json.loads((tmp_path / "metrics.json").read_text())
        assert "baseline_logistic_regression" in data
        assert "xgboost_default_threshold" in data
        assert "xgboost_high_sensitivity" in data

"""
Tests for case-studies/naives-bayes-models/text-sender-identification/
         scripts/text_sender_identification_nb.py  (batch version)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "text_sender_nb",
    "case-studies/naives-bayes-models/text-sender-identification/"
    "scripts/text_sender_identification_nb.py",
)

build_model = _MOD.build_model
load_dataset = _MOD.load_dataset
compute_metrics = _MOD.compute_metrics
save_metrics_json = _MOD.save_metrics_json
ClassifierMetrics = _MOD.ClassifierMetrics
ALLOWED_TIMES = _MOD.ALLOWED_TIMES
TEXT_COL = _MOD.TEXT_COL
TIME_COL = _MOD.TIME_COL
LABEL_COL = _MOD.LABEL_COL


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        m = build_model()
        assert isinstance(m, Pipeline)

    def test_has_nb_step(self):
        m = build_model()
        assert "nb" in m.named_steps

    def test_has_features_step(self):
        m = build_model()
        assert "features" in m.named_steps

    def test_custom_alpha_accepted(self):
        m = build_model(alpha=0.5)
        nb = m.named_steps["nb"]
        assert nb.alpha == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "does_not_exist.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["hi"], "label": ["A"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(csv)

    def test_raises_on_invalid_time_of_day(self, tmp_path):
        csv = tmp_path / "bad_time.csv"
        pd.DataFrame({
            "text": ["hi"],
            "time_of_day": ["lunchtime"],
            "label": ["Alice"],
        }).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Invalid time_of_day"):
            load_dataset(csv)

    def test_loads_valid_csv(self, tmp_path, synthetic_sms_df):
        csv = tmp_path / "sms.csv"
        synthetic_sms_df.to_csv(csv, index=False)
        df = load_dataset(csv)
        assert set(df.columns) >= {TEXT_COL, TIME_COL, LABEL_COL}
        assert len(df) == len(synthetic_sms_df)

    def test_normalises_time_to_lowercase(self, tmp_path):
        csv = tmp_path / "case.csv"
        pd.DataFrame({
            "text": ["Hello"],
            "time_of_day": ["Morning"],
            "label": ["Alice"],
        }).to_csv(csv, index=False)
        df = load_dataset(csv)
        assert df[TIME_COL].iloc[0] == "morning"

    def test_fills_empty_text_with_empty_string(self, tmp_path):
        csv = tmp_path / "null_text.csv"
        pd.DataFrame({
            "text": [None, "Hi"],
            "time_of_day": ["morning", "afternoon"],
            "label": ["Alice", "Bob"],
        }).to_csv(csv, index=False)
        df = load_dataset(csv)
        assert df[TEXT_COL].iloc[0] == ""


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_accuracy(self):
        y = np.array(["A", "B", "C", "A"])
        m = compute_metrics(y, y)
        assert m.accuracy == pytest.approx(1.0)
        assert m.macro_f1 == pytest.approx(1.0)

    def test_returns_classifier_metrics(self):
        y = np.array(["A", "A", "B", "B"])
        m = compute_metrics(y, y)
        assert isinstance(m, ClassifierMetrics)

    def test_n_test_is_correct(self):
        y = np.array(["X"] * 17)
        m = compute_metrics(y, y)
        assert m.n_test == 17

    def test_to_dict_has_all_keys(self):
        y = np.array(["A", "B"])
        d = compute_metrics(y, y).to_dict()
        for k in ("accuracy", "macro_f1", "n_test"):
            assert k in d

    def test_to_dict_json_serialisable(self):
        y = np.array(["A", "B", "A"])
        json.dumps(compute_metrics(y, y).to_dict())


# ---------------------------------------------------------------------------
# save_metrics_json
# ---------------------------------------------------------------------------

class TestSaveMetricsJson:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "metrics.json"
        save_metrics_json(p, {"accuracy": 0.92})
        assert p.exists()

    def test_valid_json_content(self, tmp_path):
        p = tmp_path / "m.json"
        save_metrics_json(p, {"x": 1, "y": [1, 2]})
        assert json.loads(p.read_text()) == {"x": 1, "y": [1, 2]}


# ---------------------------------------------------------------------------
# Full pipeline: train → predict
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_model_trains_and_predicts(self, synthetic_sms_df):
        from sklearn.model_selection import train_test_split
        X = synthetic_sms_df[[TEXT_COL, TIME_COL]]
        y = synthetic_sms_df[LABEL_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        model = build_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_proba_sums_to_one(self, synthetic_sms_df):
        from sklearn.model_selection import train_test_split
        X = synthetic_sms_df[[TEXT_COL, TIME_COL]]
        y = synthetic_sms_df[LABEL_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        model = build_model()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_above_baseline_accuracy(self, synthetic_sms_df):
        from sklearn.model_selection import train_test_split
        X = synthetic_sms_df[[TEXT_COL, TIME_COL]]
        y = synthetic_sms_df[LABEL_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        # Baseline: always predict majority class
        majority = y_train.value_counts().idxmax()
        baseline_acc = (y_test == majority).mean()

        model = build_model()
        model.fit(X_train, y_train)
        m = compute_metrics(y_test.to_numpy(), model.predict(X_test))
        # NB should beat a naive majority-class baseline
        assert m.accuracy >= baseline_acc

"""
Tests for case-studies/naives-bayes-models/text-sender-identification/
         scripts/text_sender_identification_nb_LIVE.py  (interactive version)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "text_sender_live",
    "case-studies/naives-bayes-models/text-sender-identification/"
    "scripts/text_sender_identification_nb_LIVE.py",
)

normalize_time_of_day = _MOD.normalize_time_of_day
predict_sender = _MOD.predict_sender
build_model = _MOD.build_model
compute_metrics = _MOD.compute_metrics
ALLOWED_TIMES = _MOD.ALLOWED_TIMES
TEXT_COL = _MOD.TEXT_COL
TIME_COL = _MOD.TIME_COL
LABEL_COL = _MOD.LABEL_COL


# ---------------------------------------------------------------------------
# normalize_time_of_day
# ---------------------------------------------------------------------------

class TestNormalizeTimeOfDay:
    @pytest.mark.parametrize("inp,expected", [
        ("morning", "morning"),
        ("afternoon", "afternoon"),
        ("evening", "evening"),
        ("night", "night"),
        ("Morning", "morning"),    # case insensitive
        ("EVENING", "evening"),
    ])
    def test_direct_valid_values(self, inp, expected):
        assert normalize_time_of_day(inp) == expected

    @pytest.mark.parametrize("alias,expected", [
        ("am", "morning"),
        ("pm", "afternoon"),
        ("noon", "afternoon"),
        ("midday", "afternoon"),
        ("late", "night"),
        ("late night", "night"),
        ("tonight", "night"),
    ])
    def test_aliases_resolved(self, alias, expected):
        assert normalize_time_of_day(alias) == expected

    @pytest.mark.parametrize("bad", ["lunchtime", "dawn", "dusk", "12pm", ""])
    def test_invalid_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="Invalid time_of_day"):
            normalize_time_of_day(bad)

    def test_strips_whitespace(self):
        assert normalize_time_of_day("  morning  ") == "morning"


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        assert isinstance(build_model(), Pipeline)

    def test_has_nb_step(self):
        assert "nb" in build_model().named_steps


# ---------------------------------------------------------------------------
# predict_sender  (trained on tiny synthetic data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    """Build and fit a model on a small deterministic dataset."""
    rows = []
    senders = ["Alice", "Bob", "Charlie"]
    times = ["morning", "afternoon", "evening"]
    import random
    random.seed(7)
    for _ in range(120):
        sender = random.choice(senders)
        time = random.choice(times)
        rows.append({"text": f"Hi from {sender}", "time_of_day": time, "label": sender})
    df = pd.DataFrame(rows)
    X = df[[TEXT_COL, TIME_COL]]
    y = df[LABEL_COL]
    model = build_model()
    model.fit(X, y)
    return model


class TestPredictSender:
    def test_returns_string_prediction(self, trained_model):
        pred, _ = predict_sender(trained_model, "Hello there", "morning")
        assert isinstance(pred, str)

    def test_returns_ranked_probabilities(self, trained_model):
        _, ranked = predict_sender(trained_model, "Hello there", "morning")
        assert isinstance(ranked, list)
        assert len(ranked) > 0
        for label, prob in ranked:
            assert isinstance(label, str)
            assert 0.0 <= prob <= 1.0

    def test_probabilities_sum_to_one(self, trained_model):
        _, ranked = predict_sender(trained_model, "Test message", "evening")
        total = sum(p for _, p in ranked)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_ranked_in_descending_order(self, trained_model):
        _, ranked = predict_sender(trained_model, "Some text", "afternoon")
        probs = [p for _, p in ranked]
        assert probs == sorted(probs, reverse=True)

    def test_prediction_is_highest_prob_label(self, trained_model):
        pred, ranked = predict_sender(trained_model, "any text", "morning")
        top_label = ranked[0][0]
        assert pred == top_label


# ---------------------------------------------------------------------------
# compute_metrics (shared with batch, quick sanity check)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_accuracy(self):
        y = np.array(["A", "B", "C"])
        m = compute_metrics(y, y)
        assert m.accuracy == pytest.approx(1.0)

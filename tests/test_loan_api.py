"""
Tests for case-studies/random-forest-models/loan-default-prediction/api/api.py

The API loads the model at import time, so we mock joblib.load before
the module is executed to avoid requiring a trained artifact on disk.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

API_PATH = (
    REPO_ROOT
    / "case-studies"
    / "random-forest-models"
    / "loan-default-prediction"
    / "api"
    / "api.py"
)

# ---------------------------------------------------------------------------
# Build a minimal mock artifact (pipeline + threshold) used across all tests
# ---------------------------------------------------------------------------

def _make_mock_artifact(prob: float = 0.72, threshold: float = 0.45):
    """Return a dict that mimics the joblib artifact {pipeline, threshold}."""
    mock_pipeline = MagicMock()
    # predict_proba returns [[1-prob, prob]]
    mock_pipeline.predict_proba.return_value = np.array([[1 - prob, prob]])
    return {"pipeline": mock_pipeline, "threshold": threshold}


# ---------------------------------------------------------------------------
# Module-level fixture: import api.py with mocked joblib
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_app():
    """
    Import api.py with joblib.load mocked so no real model file is needed.
    Returns the FastAPI app object.
    """
    pytest.importorskip("fastapi", reason="fastapi not installed")
    pytest.importorskip("httpx", reason="httpx not installed (needed by TestClient)")

    mock_artifact = _make_mock_artifact(prob=0.72, threshold=0.45)

    # Patch joblib.load at the joblib module level before exec_module runs
    with patch("joblib.load", return_value=mock_artifact):
        spec = importlib.util.spec_from_file_location("loan_api_mod", str(API_PATH))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["loan_api_mod"] = mod
        spec.loader.exec_module(mod)

    return mod.app


@pytest.fixture(scope="module")
def client(api_app):
    from fastapi.testclient import TestClient
    return TestClient(api_app)


# ---------------------------------------------------------------------------
# Minimal valid request payload
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "loan_amount": 15000.0,
    "term_months": 36,
    "interest_rate": 9.5,
    "annual_income": 55000.0,
    "credit_score": 720,
    "dti": 0.28,
    "utilization": 0.35,
    "employment_years": 4.0,
    "delinquencies": 0,
    "prior_defaults": 0,
    "payment_to_income": 0.08,
    "purpose": "debt_consolidation",
    "state": "CA",
    "age": 34,
    "sex": "M",
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_contains_model_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data

    def test_contains_threshold(self, client):
        data = client.get("/health").json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)


# ---------------------------------------------------------------------------
# POST /v1/score
# ---------------------------------------------------------------------------

class TestScoreEndpoint:
    def test_returns_200(self, client):
        resp = client.post("/v1/score", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_default_probability(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        assert "default_probability" in data

    def test_default_probability_is_float(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        assert isinstance(data["default_probability"], float)

    def test_default_probability_bounded(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["default_probability"] <= 1.0

    def test_flag_for_review_is_bool(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        assert isinstance(data["flag_for_review"], bool)

    def test_flag_for_review_reflects_threshold(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        prob = data["default_probability"]
        thr = data["threshold_used"]
        expected_flag = prob >= thr
        assert data["flag_for_review"] == expected_flag

    def test_contains_threshold_used(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        assert "threshold_used" in data

    def test_contains_model_version(self, client):
        data = client.post("/v1/score", json=VALID_PAYLOAD).json()
        assert "model_version" in data

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "credit_score"}
        resp = client.post("/v1/score", json=bad_payload)
        assert resp.status_code == 422

    def test_optional_field_can_be_null(self, client):
        payload = {**VALID_PAYLOAD, "annual_income": None, "utilization": None}
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200

    def test_high_risk_borrower_flagged(self):
        """Re-import with a high-risk mock (prob = 0.85 > threshold = 0.45)."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        mock_artifact = _make_mock_artifact(prob=0.85, threshold=0.45)
        with patch("joblib.load", return_value=mock_artifact):
            spec = importlib.util.spec_from_file_location("loan_api_hi", str(API_PATH))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["loan_api_hi"] = mod
            spec.loader.exec_module(mod)

        tc = TestClient(mod.app)
        data = tc.post("/v1/score", json=VALID_PAYLOAD).json()
        assert data["flag_for_review"] is True

    def test_low_risk_borrower_not_flagged(self):
        """Re-import with a low-risk mock (prob = 0.10 < threshold = 0.45)."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        mock_artifact = _make_mock_artifact(prob=0.10, threshold=0.45)
        with patch("joblib.load", return_value=mock_artifact):
            spec = importlib.util.spec_from_file_location("loan_api_lo", str(API_PATH))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["loan_api_lo"] = mod
            spec.loader.exec_module(mod)

        tc = TestClient(mod.app)
        data = tc.post("/v1/score", json=VALID_PAYLOAD).json()
        assert data["flag_for_review"] is False

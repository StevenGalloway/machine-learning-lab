
# Run:
#   uvicorn api:app --reload      (from the api/ directory)
#   uvicorn case_studies.random_forest_models.loan_default_prediction.api.api:app --reload

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent.parent / "models" / "loan_default_rf_pipeline.joblib"

app = FastAPI(title="Loan Default Scoring API", version="1.0")

# Load model once at startup — fail fast if artifact is missing
try:
    _artifact = joblib.load(MODEL_PATH)
    _pipeline = _artifact["pipeline"]
    _threshold = float(_artifact["threshold"])
except FileNotFoundError:
    raise RuntimeError(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run scripts/loan_default_random_forest.py first to train and serialize the model."
    )


class ScoreRequest(BaseModel):
    loan_amount: float
    term_months: int
    interest_rate: float
    annual_income: float | None = None
    credit_score: int
    dti: float
    utilization: float | None = None
    employment_years: float
    delinquencies: int
    prior_defaults: int
    payment_to_income: float
    purpose: str
    state: str
    age: int
    sex: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "rf_v1", "threshold": _threshold}


@app.post("/v1/score")
def score(req: ScoreRequest):
    X = pd.DataFrame([req.model_dump()])
    prob = float(_pipeline.predict_proba(X)[:, 1][0])
    return {
        "default_probability": round(prob, 4),
        "flag_for_review": prob >= _threshold,
        "threshold_used": _threshold,
        "model_version": "rf_v1",
    }

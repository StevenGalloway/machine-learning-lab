
# Run:
  # uvicorn api:app --reload

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/loan_default_rf_pipeline.joblib")
app = FastAPI(title="Loan Default Scoring API", version="1.0")

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

@app.post("/v1/score")
def score(req: ScoreRequest):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(X)[:, 1][0])
    return {"default_probability": prob, "model_version": "rf_v1"}

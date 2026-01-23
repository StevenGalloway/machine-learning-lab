"""
Loan Approval Decision Support (Assistive, human-in-the-loop)

- Positive class: Approved (1)
- Trains a baseline Logistic Regression and an XGBoost model
- Prints governance-friendly metrics
- Generates artifacts:
  - results/metrics.json
  - results/roc_curve.png
  - results/confusion_matrix.png

Run:
  python loan_approval_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# -------------------------
# Global configuration
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Data
DATA_PATH = Path("case-studies/loan-approval-xgb/data/loan_approval_data.csv")
TARGET_COL = "Approved"
FEATURE_COLS = ["Age", "Income", "LoanAmount", "CreditScore"]

# Artifacts output
CASE_STUDY_DIR = Path("case-studies/loan-approval-xgb")
RESULTS_DIR = CASE_STUDY_DIR / "results"
GENERATE_PLOTS = True
GENERATE_METRICS_JSON = True


@dataclass(frozen=True)
class Metrics:
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    roc_auc: float
    precision_ppv: float
    recall_sensitivity: float
    specificity: float
    f1: float
    brier: float
    support_test: int
    positive_rate_test: float

    def to_dict(self) -> dict:
        return {
            "tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "precision_ppv": self.precision_ppv,
            "recall_sensitivity": self.recall_sensitivity,
            "specificity": self.specificity,
            "f1": self.f1,
            "brier": self.brier,
            "support_test": self.support_test,
            "positive_rate_test": self.positive_rate_test,
        }


def load_data(path: Path) -> pd.DataFrame:
    """Load input dataset. Raises a helpful error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at: {path.resolve()}\n"
            f"Update DATA_PATH or place the CSV at that location."
        )
    df = pd.read_csv(path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Basic schema validation to prevent silent failures."""
    missing = [c for c in (FEATURE_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Basic type checks (lightweight)
    for c in FEATURE_COLS:
        if not np.issubdtype(df[c].dtype, np.number):
            raise ValueError(f"Feature '{c}' must be numeric. Got dtype={df[c].dtype}")

    # Target check
    if df[TARGET_COL].nunique() > 2:
        raise ValueError(f"Target '{TARGET_COL}' must be binary (0/1). Found values: {df[TARGET_COL].unique()}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Metrics:
    """Compute decisioning-friendly metrics (approval=positive)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # In lending approval, interpretability often focuses on:
    # - Precision (PPV): of approvals, how many were truly approved historically (proxy)
    # - Recall: how many approvals we catch
    # - Specificity: reject correctness
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return Metrics(
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        precision_ppv=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_sensitivity=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        brier=float(brier_score_loss(y_true, y_prob)),
        support_test=int(len(y_true)),
        positive_rate_test=float(np.mean(y_true)),
    )


def choose_threshold_by_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.90
) -> float:
    """
    Choose an operating threshold emphasizing minimizing false approvals.
    We scan thresholds and pick the one with best recall subject to precision >= min_precision.
    Fallback: maximize F1 if constraint cannot be met.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if prec >= min_precision:
            # maximize recall under the precision constraint
            if best is None or rec > best["recall"]:
                best = {"thr": thr, "precision": prec, "recall": rec}

    if best is not None:
        return float(best["thr"])

    # fallback: maximize F1
    f1s = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    return float(thresholds[int(np.argmax(f1s))])


def pretty_print(name: str, m: Metrics, threshold: float | None = None) -> None:
    """Print metrics in an executive-friendly format."""
    print(f"\n=== {name} ===")
    if threshold is not None:
        print(f"Threshold: {threshold:.3f}")
    print(f"Confusion Matrix (approved=positive): TN={m.tn}, FP={m.fp}, FN={m.fn}, TP={m.tp}")
    print(f"Accuracy:      {m.accuracy:.3f}")
    print(f"ROC AUC:       {m.roc_auc:.3f}")
    print(f"Precision(PPV):{m.precision_ppv:.3f}  (of predicted approvals, how many match historical approvals)")
    print(f"Recall:        {m.recall_sensitivity:.3f}  (of historical approvals, how many we capture)")
    print(f"Specificity:   {m.specificity:.3f}  (of historical rejections, how many we keep rejected)")
    print(f"F1:            {m.f1:.3f}")
    print(f"Brier:         {m.brier:.3f}  (probability calibration)")


def save_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plots(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> None:
    """Generate ROC curve and confusion matrix images."""
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - Loan Approval (AUC = {roc_auc:.3f})")
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Rejected", "Approved"])
    plt.yticks([0, 1], ["Rejected", "Approved"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Loan Approval")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    # Load + validate
    df = load_data(DATA_PATH)
    validate_schema(df)

    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET_COL].to_numpy().astype(int)

    # Split with stratification for stable class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline: Logistic Regression with scaling (common underwriting baseline)
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])
    baseline.fit(X_train, y_train)
    b_prob = baseline.predict_proba(X_test)[:, 1]
    b_pred = (b_prob >= 0.5).astype(int)
    b_metrics = compute_metrics(y_test, b_pred, b_prob)
    pretty_print("Logistic Regression (baseline)", b_metrics)

    # Candidate: XGBoost for nonlinear interactions
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost is not available. Install with: pip install xgboost")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        reg_lambda=1.0,
    )

    model.fit(X_train, y_train)
    p_prob = model.predict_proba(X_test)[:, 1]
    p_pred = (p_prob >= 0.5).astype(int)
    p_metrics = compute_metrics(y_test, p_pred, p_prob)
    pretty_print("XGBoost (default threshold = 0.50)", p_metrics)

    # Policy-oriented threshold: minimize false approvals by enforcing a precision floor
    thr = choose_threshold_by_precision(y_test, p_prob, min_precision=0.90)
    p_pred_thr = (p_prob >= thr).astype(int)
    p_metrics_thr = compute_metrics(y_test, p_pred_thr, p_prob)
    pretty_print("XGBoost (policy threshold)", p_metrics_thr, threshold=thr)

    # Artifacts
    if GENERATE_PLOTS:
        save_plots(y_test, p_prob, p_pred_thr)

    if GENERATE_METRICS_JSON:
        payload = {
            "metadata": {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "positive_class": "Approved (1)",
                "features": FEATURE_COLS,
                "data_path": str(DATA_PATH),
                "policy": {"threshold_selection": "precision_floor", "min_precision": 0.90, "threshold": thr},
            },
            "baseline_logistic_regression": b_metrics.to_dict(),
            "xgboost_default_threshold": p_metrics.to_dict(),
            "xgboost_policy_threshold": p_metrics_thr.to_dict(),
        }
        save_metrics_json(RESULTS_DIR / "metrics.json", payload)


if __name__ == "__main__":
    main()

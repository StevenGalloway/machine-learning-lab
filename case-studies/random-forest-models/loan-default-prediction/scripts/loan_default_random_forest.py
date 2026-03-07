from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

RANDOM_STATE = 42
TEST_SIZE = 0.2

CASE_STUDY_DIR = Path("case-studies/random-forest-models/loan-default-prediction")
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"
DATA_PATH = DATA_DIR / "personal_loan_default_synthetic_1500.csv"
TARGET_COL = "default"  # 1=default, 0=no default
SENSITIVE_COLS = ["sex"]

def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision_default": float(prec),
        "recall_default": float(rec),
        "f1_default": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fnr": float(fnr),
        "fpr": float(fpr),
    }

def choose_cost_aware_threshold(y_true: np.ndarray, prob: np.ndarray, max_fpr: float = 0.35) -> dict:
    thresholds = np.linspace(0.05, 0.95, 181)
    metrics = [threshold_metrics(y_true, prob, t) for t in thresholds]
    ok = [m for m in metrics if m["fpr"] <= max_fpr]
    best = min(ok, key=lambda m: m["fnr"]) if ok else min(metrics, key=lambda m: m["fnr"])
    return best

def get_rf_feature_importances(model_pipeline) -> np.ndarray:
    
    clf = model_pipeline.named_steps["clf"]

    # Case 1: direct RF
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_

    # Case 2: calibrated wrapper (uses fitted calibrators / estimators_)
    if hasattr(clf, "calibrated_classifiers_"):
        # Each calibrated classifier wraps an estimator; pull and average importances
        importances = []
        for cc in clf.calibrated_classifiers_:
            est = cc.estimator  # fitted base model
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        if importances:
            return np.mean(np.vstack(importances), axis=0)

    raise AttributeError("Could not find feature_importances_ in model or its calibrated base estimators.")

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    baseline = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1200, solver="lbfgs"))
    ])
    baseline.fit(X_train, y_train)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    base_auc = float(roc_auc_score(y_test, base_prob))

    rf_raw = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    rf = Pipeline([
        ("preprocess", preprocess),
        ("clf", CalibratedClassifierCV(rf_raw, method="isotonic", cv=3)
        )
    ])
    
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_prob))

    # plots
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fpr_b, tpr_b, _ = roc_curve(y_test, base_prob)
    fpr_r, tpr_r, _ = roc_curve(y_test, rf_prob)
    plt.figure(figsize=(7, 5.2))
    plt.plot(fpr_b, tpr_b, label=f"LogReg (AUC={base_auc:.3f})")
    plt.plot(fpr_r, tpr_r, label=f"RandomForest (AUC={rf_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Personal Loan Default Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    best = choose_cost_aware_threshold(y_test.to_numpy(), rf_prob, max_fpr=0.35)
    pred_best = (rf_prob >= best["threshold"]).astype(int)
    cm = confusion_matrix(y_test, pred_best, labels=[0, 1])
    plt.figure(figsize=(5.8, 4.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — RF @ thr={best['threshold']:.2f}")
    plt.xticks([0, 1], ["No default (0)", "Default (1)"], rotation=25, ha="right")
    plt.yticks([0, 1], ["No default (0)", "Default (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    ohe = rf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    feature_names = np.array(num_cols + list(ohe.get_feature_names_out(cat_cols)))
    importances = get_rf_feature_importances(rf)
    idx = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(9, 5.6))
    plt.barh(range(len(idx))[::-1], importances[idx][::-1])
    plt.yticks(range(len(idx))[::-1], feature_names[idx][::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances — Random Forest")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_top20.png", dpi=150)
    plt.close()

    # fairness: AUC by sex
    auc_by_sex = {}
    for g in sorted(X_test["sex"].dropna().unique().tolist()):
        mask = (X_test["sex"] == g).to_numpy()
        if mask.sum() >= 20 and len(np.unique(y_test.to_numpy()[mask])) > 1:
            auc_by_sex[g] = float(roc_auc_score(y_test.to_numpy()[mask], rf_prob[mask]))
        else:
            auc_by_sex[g] = None

    payload = {
        "metadata": {
            "task_type": "classification",
            "use_case": "personal_loan_default_prediction",
            "label_definition": {"1": "default", "0": "no_default"},
            "primary_metric": "roc_auc",
            "cost_asymmetry": "false approvals are most costly (predict 0 when actual 1)",
            "sensitive_features": SENSITIVE_COLS,
            "deployment_target": "real_time_api",
            "dataset_path": str(DATA_PATH.as_posix()),
        },
        "baseline_logistic_regression": {"roc_auc": base_auc},
        "random_forest": {
            "roc_auc": rf_auc,
            "threshold_cost_aware": best,
            "auc_by_sex": auc_by_sex,
        },
        "artifacts": {
            "roc_curve_png": "results/roc_curve.png",
            "confusion_matrix_png": "results/confusion_matrix.png",
            "feature_importance_png": "results/feature_importance_top20.png",
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Baseline LogReg ROC AUC: {base_auc:.3f}")
    print(f"RandomForest ROC AUC:    {rf_auc:.3f}")
    print(f"Cost-aware threshold:    {best['threshold']:.2f} (FNR={best['fnr']:.3f}, FPR={best['fpr']:.3f})")

if __name__ == "__main__":
    main()

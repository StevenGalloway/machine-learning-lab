from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Global
RANDOM_STATE = 42
TEST_SIZE = 0.20

CASE_STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"
DOCS_DIR = CASE_STUDY_DIR / "supporting-documentation"

DEFAULT_DATA = DATA_DIR / "spreadspoke_scores_sample.csv"


@dataclass(frozen=True)
class PredictionQuery:
    season: int
    week: int
    home_team: str
    away_team: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_mkdirs() -> None:
    """Create required directories."""
    for p in [DATA_DIR, RESULTS_DIR, DOCS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _week_to_num(w: Any) -> int:
    mapping = {
        "wildcard": 19,
        "division": 20,
        "conference": 21,
        "superbowl": 22,
        "super bowl": 22,
    }
    s = str(w).strip().lower()
    if s.isdigit():
        return int(s)
    return mapping.get(s, 18)


def load_and_prepare(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required = {
        "schedule_season", "schedule_week", "schedule_date",
        "team_home", "team_away",
        "score_home", "score_away",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    df = df.copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df = df.dropna(subset=["schedule_date", "score_home", "score_away"])

    # Target: 1 if home team wins
    df["home_win"] = (df["score_home"] > df["score_away"]).astype(int)

    # Feature engineering
    df["week_num"] = df["schedule_week"].apply(_week_to_num).astype(int)
    df["month"] = df["schedule_date"].dt.month.astype(int)

    # Optional features (if available)
    if "neutral_site" in df.columns:
        df["neutral_site"] = df["neutral_site"].fillna(0).astype(int)
    if "home_rest_days" in df.columns:
        df["home_rest_days"] = pd.to_numeric(df["home_rest_days"], errors="coerce")
    if "away_rest_days" in df.columns:
        df["away_rest_days"] = pd.to_numeric(df["away_rest_days"], errors="coerce")
    if "over_under_line" in df.columns:
        df["over_under_line"] = pd.to_numeric(df["over_under_line"], errors="coerce")
    if "spread_favorite" in df.columns:
        df["spread_favorite"] = pd.to_numeric(df["spread_favorite"], errors="coerce")

    # Keep "modern" era by default
    df = df[df["schedule_season"] >= 2002].copy()

    return df


def build_pipelines(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline, List[str]]:
    feature_cols = [
        "schedule_season", "week_num", "month",
        "team_home", "team_away",
    ]
    optional_numeric = ["neutral_site", "home_rest_days", "away_rest_days", "over_under_line", "spread_favorite"]
    for c in optional_numeric:
        if c in df.columns:
            feature_cols.append(c)

    categorical = ["team_home", "team_away"]
    numeric = [c for c in feature_cols if c not in categorical]

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric),
        ],
        remainder="drop",
    )

    baseline = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=None, random_state=RANDOM_STATE)),
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=3)

    model = Pipeline(steps=[
        ("pre", pre),
        ("clf", rf_calibrated),
    ])

    return baseline, model, feature_cols


def season_aware_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent season when possible; otherwise random stratified split."""
    seasons = sorted(df["schedule_season"].unique())
    if len(seasons) >= 2:
        test_season = seasons[-1]
        train_df = df[df["schedule_season"] < test_season].copy()
        test_df = df[df["schedule_season"] == test_season].copy()
        return train_df, test_df

    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["home_win"]
    )


# -----------------------------
# Plotting (regenerated each run)
# -----------------------------
def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, outpath: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return float(auc)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_feature_importance(model: Pipeline, outpath: Path, top_n: int = 20) -> None:
    pre: ColumnTransformer = model.named_steps["pre"]
    calibrated: CalibratedClassifierCV = model.named_steps["clf"]
    base_rf: RandomForestClassifier = calibrated.calibrated_classifiers_[0].estimator

    cat_ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_features = list(cat_ohe.get_feature_names_out(["team_home", "team_away"]))
    num_features = pre.transformers_[1][2]  # columns for "num" transformer
    feature_names = cat_features + list(num_features)

    importances = base_rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_vals))[::-1], top_vals[::-1])
    plt.yticks(range(len(top_names))[::-1], top_names[::-1], fontsize=8)
    plt.xlabel("Impurity-based importance")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------
# JSON output (regenerated each run)
# -----------------------------
def write_metrics_json(payload: Dict[str, Any]) -> None:
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def predict_single_game(model: Pipeline, query: PredictionQuery, template_row: pd.Series) -> Dict[str, Any]:
    row = template_row.copy()
    row["schedule_season"] = query.season
    row["week_num"] = query.week
    row["team_home"] = query.home_team
    row["team_away"] = query.away_team
    if "month" in row.index:
        row["month"] = int(row.get("month", 9))  # default to September if unknown
    X = pd.DataFrame([row])
    p = float(model.predict_proba(X)[:, 1][0])
    return {"p_home_win": p, "predicted_label": int(p >= 0.5)}


def main() -> None:
    _safe_mkdirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to CSV dataset")
    parser.add_argument("--predict-home", type=str, default="Baltimore Ravens")
    parser.add_argument("--predict-away", type=str, default="Pittsburgh Steelers")
    parser.add_argument("--predict-season", type=int, default=2025)
    parser.add_argument("--predict-week", type=int, default=3)
    parser.add_argument("--regen-static", action="store_true", help="Regenerate static docs/markdown artifacts")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            f"Place a CSV under {DATA_DIR.as_posix()} or pass --data <path>."
        )

    df = load_and_prepare(data_path)
    train_df, test_df = season_aware_split(df)

    baseline, model, feature_cols = build_pipelines(df)

    X_train = train_df[feature_cols]
    y_train = train_df["home_win"].values
    X_test = test_df[feature_cols]
    y_test = test_df["home_win"].values

    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)

    # Evaluate baseline (kept in JSON for context, but no baseline markdown is regenerated)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    base_pred = (base_prob >= 0.5).astype(int)

    # Evaluate RF
    rf_prob = model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    def _pack(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1])),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }

    baseline_metrics = _pack(y_test, base_prob, base_pred)

    calibrated: CalibratedClassifierCV = model.named_steps["clf"]
    rf_est: RandomForestClassifier = calibrated.calibrated_classifiers_[0].estimator
    rf_hp = {
        "n_estimators": int(getattr(rf_est, "n_estimators", 0)),
        "max_depth": getattr(rf_est, "max_depth", None),
        "min_samples_leaf": int(getattr(rf_est, "min_samples_leaf", 0)),
        "min_samples_split": int(getattr(rf_est, "min_samples_split", 0)),
        "class_weight": getattr(rf_est, "class_weight", None),
        "calibration": {"method": calibrated.method, "cv": calibrated.cv},
    }

    rf_metrics = _pack(y_test, rf_prob, rf_pred)
    rf_metrics["hyperparameters"] = rf_hp

    # Regenerate plots every run
    roc_path = RESULTS_DIR / "roc_curve.png"
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fi_path = RESULTS_DIR / "feature_importance_top20.png"

    rf_auc = plot_roc(y_test, rf_prob, roc_path)
    plot_confusion(y_test, rf_pred, cm_path)
    plot_feature_importance(model, fi_path, top_n=20)

    # Prediction example
    query = PredictionQuery(
        season=int(args.predict_season),
        week=int(args.predict_week),
        home_team=str(args.predict_home),
        away_team=str(args.predict_away),
    )
    template_row = X_test.iloc[0].copy()
    pred_payload = predict_single_game(model, query, template_row)

    payload: Dict[str, Any] = {
        "metadata": {
            "task_type": "classification",
            "use_case": "nfl_home_win_prediction",
            "primary_metric": "roc_auc",
            "data_path": str(Path(args.data)),
            "dataset_sha256": _sha256_file(data_path),
            "split_strategy": "season_aware_holdout" if len(df["schedule_season"].unique()) >= 2 else "random_stratified",
            "test_rows": int(len(test_df)),
            "random_state": RANDOM_STATE,
        },
        "baseline": baseline_metrics,
        "random_forest": rf_metrics,
        "artifacts": {
            "roc_curve_png": str(roc_path.relative_to(CASE_STUDY_DIR)),
            "confusion_matrix_png": str(cm_path.relative_to(CASE_STUDY_DIR)),
            "feature_importance_png": str(fi_path.relative_to(CASE_STUDY_DIR)),
            "metrics_json": "results/metrics.json",
        },
        "prediction_example": {
            "query": {
                "season": query.season,
                "week": query.week,
                "home_team": query.home_team,
                "away_team": query.away_team,
            },
            **pred_payload,
        },
    }

    # JSON is regenerated every run
    write_metrics_json(payload)

    print(f"Holdout AUC (RF): {rf_auc:.3f}")
    print(f"Example prediction: P(home_win)={pred_payload['p_home_win']:.3f} for {query.home_team} vs {query.away_team}")
    print("Regenerated each run: results/metrics.json + PNGs in results/")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Global configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

CASE_STUDY_DIR = Path("case-studies/basketball-points-prediction-linear-reg")
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"

DATA_PATH = DATA_DIR / "points_prediction_data.csv"

PLAYER_COL = "Player"
DATE_COL = "Date"
MINUTES_COL = "MP"
POINTS_COL = "PTS"
TARGET_COL = "PTS_next"

FEATURE_COLS = ["MIN_float","FGA","3PA","FTA","ORB","DRB","AST","STL","BLK","TOV","PF"]

GENERATE_PLOTS = True
GENERATE_METRICS_JSON = True


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float
    smape: float
    n_test: int

    def to_dict(self) -> dict:
        return {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "r2": self.r2,
            "smape": self.smape,
            "n_test": self.n_test,
        }


def mp_to_minutes(x: object) -> float:
    s = str(x)
    if "." in s:
        mm, ss = s.split(".", 1)
        try:
            return float(mm) + float(ss) / 60.0
        except Exception:
            return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path.resolve()}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    required = [PLAYER_COL, DATE_COL, MINUTES_COL, POINTS_COL, "FGA", "3PA", "FTA", "ORB", "DRB", "AST", "STL", "BLK", "TOV", "PF"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["MIN_float"] = out[MINUTES_COL].apply(mp_to_minutes)

    out = out.sort_values([PLAYER_COL, DATE_COL])
    out[TARGET_COL] = out.groupby(PLAYER_COL)[POINTS_COL].shift(-1)
    out = out.dropna(subset=[TARGET_COL, "MIN_float"]).copy()

    for c in FEATURE_COLS:
        out[c] = out[c].astype(float)
    out[POINTS_COL] = out[POINTS_COL].astype(float)
    out[TARGET_COL] = out[TARGET_COL].astype(float)

    return out

# split by time to show trends from most recent games (more realistic)
def time_ordered_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.sort_values(DATE_COL).reset_index(drop=True)
    cut = int(len(df2) * (1 - TEST_SIZE))
    return df2.iloc[:cut].copy(), df2.iloc[cut:].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return RegressionMetrics(
        mae=float(mae),
        mse=float(mse),
        rmse=float(rmse),
        r2=float(r2),
        smape=float(smape(y_true, y_pred)),
        n_test=int(len(y_true)),
    )


def pretty_print(name: str, m: RegressionMetrics) -> None:
    print(f"\n=== {name} ===")
    print(f"MAE:   {m.mae:.3f}")
    print(f"RMSE:  {m.rmse:.3f}")
    print(f"MSE:   {m.mse:.3f}")
    print(f"R^2:   {m.r2:.3f}")
    print(f"SMAPE: {m.smape:.3f}")
    print(f"N:     {m.n_test}")


def save_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plots(y_true: np.ndarray, y_pred: np.ndarray, coef_std: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parity
    plt.figure()
    plt.scatter(y_true, y_pred)
    minv = min(float(np.min(y_true)), float(np.min(y_pred)))
    maxv = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual Next-Game Points")
    plt.ylabel("Predicted Next-Game Points")
    plt.title("Parity Plot — Basketball Next-Game Points (Linear Regression)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parity_plot.png", dpi=150)
    plt.close()

    # Residuals
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Next-Game Points")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted — Basketball Next-Game Points")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "residuals.png", dpi=150)
    plt.close()

    # Standardized coefficient importance
    order = np.argsort(np.abs(coef_std))[::-1]
    plt.figure(figsize=(9, 4.8))
    plt.bar(range(len(FEATURE_COLS)), coef_std[order])
    plt.xticks(range(len(FEATURE_COLS)), np.array(FEATURE_COLS)[order], rotation=45, ha="right")
    plt.ylabel("Coefficient (standardized)")
    plt.title("Feature importance (Linear Regression, standardized)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    raw = load_data(DATA_PATH)
    validate_schema(raw)
    df = make_features_and_target(raw)

    train_df, test_df = time_ordered_split(df)

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    # Baseline: next = current points
    baseline_pred = test_df[POINTS_COL].to_numpy()
    baseline_metrics = compute_metrics(y_test, baseline_pred)
    pretty_print("Baseline (next=current)", baseline_metrics)

    # Linear regression
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_metrics = compute_metrics(y_test, y_pred)
    pretty_print("Linear Regression", model_metrics)

    # Standardized coefficients for interpretability across different scales
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    lin_std = LinearRegression().fit(Xs_train, y_train)

    # Save artifacts
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Coefficients table
    coef_df = pd.DataFrame({
        "coef_raw": pd.Series(model.coef_, index=FEATURE_COLS),
        "coef_standardized": pd.Series(lin_std.coef_, index=FEATURE_COLS),
    })
    coef_df.loc["intercept", "coef_raw"] = float(model.intercept_)
    coef_df.to_csv(RESULTS_DIR / "coefficients.csv", index=True)

    if GENERATE_PLOTS:
        save_plots(y_test, y_pred, lin_std.coef_)

    if GENERATE_METRICS_JSON:
        payload = {
            "metadata": {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "task_type": "regression",
                "domain": "basketball",
                "dataset_path": str(DATA_PATH.as_posix()),
                "target": TARGET_COL,
                "features": FEATURE_COLS,
                "split_strategy": "time_ordered_by_date",
            },
            "baseline": baseline_metrics.to_dict(),
            "linear_regression": model_metrics.to_dict(),
            "coefficients": {
                "intercept": float(model.intercept_),
                "raw": dict(zip(FEATURE_COLS, model.coef_.tolist())),
                "standardized": dict(zip(FEATURE_COLS, lin_std.coef_.tolist())),
            },
        }
        save_metrics_json(RESULTS_DIR / "metrics.json", payload)


if __name__ == "__main__":
    main()

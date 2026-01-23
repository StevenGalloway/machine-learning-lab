from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


# -------------------------
# Global configuration
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

CASE_STUDY_DIR = Path("case-studies/football-points-prediction-linear-reg")
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"

FEATURE_COL = "FirstDowns"
TARGET_COL = "Points"

GENERATE_PLOTS = True
GENERATE_METRICS_JSON = True


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    r2: float
    smape: float
    n_test: int

    def to_dict(self) -> dict:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "smape": self.smape,
            "n_test": self.n_test,
        }


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def load_data(dataset: str) -> pd.DataFrame:
    dataset = dataset.lower().strip()
    if dataset not in {"nfl", "cfb"}:
        dataset = "nfl" # set to NFL by default.
    path = DATA_DIR / f"points_prediction_data_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at: {path.resolve()}\n"
            "Place the dataset CSV in the data/ folder."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in [FEATURE_COL, TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    for c in [FEATURE_COL, TARGET_COL]:
        if not np.issubdtype(df[c].dtype, np.number):
            raise ValueError(f"Column '{c}' must be numeric. Got dtype={df[c].dtype}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    s = smape(y_true, y_pred)
    return RegressionMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2),
        smape=float(s),
        n_test=int(len(y_true)),
    )


def pretty_print(name: str, m: RegressionMetrics) -> None:
    print(f"\n=== {name} ===")
    print(f"MAE:   {m.mae:.3f}  (avg absolute error in points)")
    print(f"RMSE:  {m.rmse:.3f} (penalizes large misses)")
    print(f"MSE:   {m.mse:.3f}  ((points)^2)")
    print(f"R^2:   {m.r2:.3f}   (variance explained)")
    print(f"SMAPE: {m.smape:.3f} (stable relative error)")
    print(f"N:     {m.n_test}")


def save_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plots(dataset: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, model: LinearRegression) -> None:
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_all = np.concatenate([x_train, x_test]).reshape(-1, 1)
    xline = np.linspace(x_all.min(), x_all.max(), 200).reshape(-1, 1)
    yline = model.predict(xline)

    plt.figure()
    plt.scatter(x_train, y_train, label="Train")
    plt.scatter(x_test, y_test, marker="s", label="Test")
    plt.plot(xline.flatten(), yline, linewidth=2, label="Best-fit line")
    plt.xlabel("First Downs")
    plt.ylabel("Points Scored")
    plt.title(f"Points vs First Downs (Linear Regression) — {dataset.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"fit_line_{dataset}.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(y_test, y_pred)
    minv = min(float(y_test.min()), float(y_pred.min()))
    maxv = max(float(y_test.max()), float(y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(f"Parity Plot (Actual vs Predicted) — {dataset.upper()}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"parity_plot_{dataset}.png", dpi=150)
    plt.close()

    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Points")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residuals vs Predicted — {dataset.upper()}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"residuals_{dataset}.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nfl", "cfb"], default="nfl", help="Which dataset to use")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    df = load_data(dataset)
    validate_schema(df)

    X = df[[FEATURE_COL]].to_numpy()
    y = df[TARGET_COL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    b_pred = baseline.predict(X_test)
    b_metrics = compute_metrics(y_test, b_pred)
    pretty_print(f"Baseline (mean) — {dataset.upper()}", b_metrics)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    m_metrics = compute_metrics(y_test, y_pred)
    pretty_print(f"Linear Regression — {dataset.upper()}", m_metrics)

    if GENERATE_PLOTS:
        save_plots(dataset, X_train.flatten(), y_train, X_test.flatten(), y_test, y_pred, model)

    if GENERATE_METRICS_JSON:
        payload = {
            "metadata": {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "task_type": "regression",
                "dataset": dataset,
                "features": [FEATURE_COL],
                "target": TARGET_COL,
            },
            "baseline": b_metrics.to_dict(),
            "linear_regression": m_metrics.to_dict(),
            "coefficients": {
                "intercept": float(model.intercept_),
                "coef_firstdowns": float(model.coef_[0]),
            },
        }
        save_metrics_json(RESULTS_DIR / f"metrics_{dataset}.json", payload)


if __name__ == "__main__":
    main()

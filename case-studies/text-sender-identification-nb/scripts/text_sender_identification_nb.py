from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


RANDOM_STATE = 42
TEST_SIZE = 0.2

CASE_STUDY_DIR = Path("case-studies/text-sender-identification-nb")
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"

DATA_PATH = DATA_DIR / "sms_sender_dataset_1000.csv"

ALLOWED_TIMES = {"morning", "afternoon", "evening", "night"}
TEXT_COL = "text"
TIME_COL = "time_of_day"
LABEL_COL = "label"


@dataclass(frozen=True)
class ClassifierMetrics:
    accuracy: float
    macro_f1: float
    n_test: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "n_test": self.n_test,
        }


def build_model(alpha: float = 1.0) -> Pipeline:
    featurizer = ColumnTransformer(
        transformers=[
            ("text_vec", CountVectorizer(ngram_range=(1, 2), min_df=2), TEXT_COL),
            ("time_vec", CountVectorizer(), TIME_COL),
        ],
        remainder="drop",
    )
    return Pipeline([("features", featurizer), ("nb", MultinomialNB(alpha=alpha))])


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path.resolve()}'")
    df = pd.read_csv(path)

    required = {TEXT_COL, TIME_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    df[TIME_COL] = df[TIME_COL].astype(str).str.lower().str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    bad_times = sorted(set(df[TIME_COL]) - ALLOWED_TIMES)
    if bad_times:
        raise ValueError(f"Invalid time_of_day values: {bad_times}. Allowed: {sorted(ALLOWED_TIMES)}")

    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassifierMetrics:
    return ClassifierMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        n_test=int(len(y_true)),
    )


def pretty_print(name: str, m: ClassifierMetrics) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy: {m.accuracy:.3f}")
    print(f"Macro F1:  {m.macro_f1:.3f}")
    print(f"N:         {m.n_test}")


def save_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_confusion_matrix_plot(cm: np.ndarray, labels: list[str]) -> None:
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix — Sender Identification (Naive Bayes)")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_dataset(DATA_PATH)

    X = df[[TEXT_COL, TIME_COL]]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline: majority class
    majority = y_train.value_counts().idxmax()
    baseline_pred = np.array([majority] * len(y_test))
    baseline_metrics = compute_metrics(y_test.to_numpy(), baseline_pred)
    pretty_print("Baseline (majority class)", baseline_metrics)

    # Train + predict
    model = build_model(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_metrics = compute_metrics(y_test.to_numpy(), y_pred)
    pretty_print("MultinomialNB", model_metrics)

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    save_confusion_matrix_plot(cm, labels)

    report = classification_report(y_test, y_pred, output_dict=True)
    payload = {
        "metadata": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "task_type": "classification",
            "model_family": "Multinomial Naive Bayes",
            "dataset_path": str(DATA_PATH.as_posix()),
            "features": [TEXT_COL, TIME_COL],
            "target": LABEL_COL,
            "labels": labels,
            "split_strategy": "stratified_random",
        },
        "baseline": {
            **baseline_metrics.to_dict(),
            "strategy": "majority_class",
            "majority_class": str(majority),
        },
        "naive_bayes": {
            **model_metrics.to_dict(),
            "alpha": 1.0,
            "per_class": {k: {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1-score"],
                "support": v["support"],
            } for k, v in report.items() if k in labels},
        },
    }
    save_metrics_json(RESULTS_DIR / "metrics.json", payload)


if __name__ == "__main__":
    main()

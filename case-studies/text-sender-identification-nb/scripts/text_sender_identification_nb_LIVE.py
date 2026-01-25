"""
Text Sender Identification (Naive Bayes) — Interactive demo

What it does:
1) Loads the dataset CSV (for training/eval)
2) Trains a Multinomial Naive Bayes model (CountVectorizer on text + time_of_day)
3) Prints quick eval metrics + confusion matrix (optional)
4) Starts an interactive prompt so a user can enter:
   - message text
   - time_of_day
   and receive predicted sender + probabilities

Run:
  python case-studies/text-sender-identification-nb/text_sender_identification_nb_interactive.py
"""

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


# -------------------------
# Global config
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

CASE_STUDY_DIR = Path("case-studies/text-sender-identification-nb")
DATA_DIR = CASE_STUDY_DIR / "data"
RESULTS_DIR = CASE_STUDY_DIR / "results"

DATA_PATH = DATA_DIR / "sms_sender_dataset_1000.csv"

TEXT_COL = "text"
TIME_COL = "time_of_day"
LABEL_COL = "label"
ALLOWED_TIMES = {"morning", "afternoon", "evening", "night"}

PRINT_EVAL = True
SAVE_METRICS_JSON = False  # keep interactive script lightweight by default


@dataclass(frozen=True)
class ClassifierMetrics:
    accuracy: float
    macro_f1: float
    n_test: int

    def to_dict(self) -> dict:
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1, "n_test": self.n_test}


def build_model(alpha: float = 1.0) -> Pipeline:
    """
    Multinomial Naive Bayes expects non-negative count-like features.
    We use CountVectorizer to create:
      - text unigram+bigram counts
      - a 'token' count for the time_of_day category
    """
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
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)

    required = {TEXT_COL, TIME_COL, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # normalize
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    df[TIME_COL] = df[TIME_COL].astype(str).str.lower().str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    bad_times = sorted(set(df[TIME_COL]) - ALLOWED_TIMES)
    if bad_times:
        raise ValueError(f"Invalid time_of_day values in dataset: {bad_times} (allowed={sorted(ALLOWED_TIMES)})")

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


def normalize_time_of_day(raw: str) -> str:
    s = (raw or "").strip().lower()

    # friendly aliases
    alias = {
        "am": "morning",
        "pm": "afternoon",
        "noon": "afternoon",
        "midday": "afternoon",
        "late": "night",
        "late night": "night",
        "tonight": "night",
    }
    s = alias.get(s, s)

    if s not in ALLOWED_TIMES:
        raise ValueError(f"Invalid time_of_day '{raw}'. Use one of: {sorted(ALLOWED_TIMES)}")
    return s


def predict_sender(model: Pipeline, text: str, time_of_day: str) -> tuple[str, list[tuple[str, float]]]:
    """
    Returns:
      predicted_label, [(label, prob), ...] sorted by prob desc
    """
    x = pd.DataFrame([{TEXT_COL: text, TIME_COL: time_of_day}])

    pred = model.predict(x)[0]

    # Probabilities (MultinomialNB supports predict_proba)
    probs = model.predict_proba(x)[0]
    labels = model.named_steps["nb"].classes_
    ranked = sorted(zip(labels, probs), key=lambda t: t[1], reverse=True)

    return str(pred), [(str(lbl), float(p)) for lbl, p in ranked]


def interactive_loop(model: Pipeline) -> None:
    print("\n--- Interactive Mode ---")
    print("Type 'exit' at any prompt to quit.")
    print(f"Valid time_of_day values: {sorted(ALLOWED_TIMES)}")

    while True:
        raw_text = input("\nEnter message text: ").strip()
        if raw_text.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        if not raw_text:
            print("Please enter some text.")
            continue

        raw_time = input("Enter time_of_day (morning/afternoon/evening/night): ").strip()
        if raw_time.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        try:
            tod = normalize_time_of_day(raw_time)
        except ValueError as e:
            print(f"⚠️  {e}")
            continue

        pred, ranked = predict_sender(model, raw_text, tod)

        print(f"\nPredicted sender: {pred}")
        print("Top probabilities:")
        for lbl, p in ranked[:4]:
            print(f"  - {lbl}: {p:.3f}")


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

    # Train model
    model = build_model(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    model_metrics = compute_metrics(y_test.to_numpy(), y_pred)

    if PRINT_EVAL:
        pretty_print("Baseline (majority class)", baseline_metrics)
        pretty_print("MultinomialNB", model_metrics)

        labels = sorted(y.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("\nConfusion Matrix (rows=actual, cols=predicted):")
        print("labels:", labels)
        print(cm)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

    if SAVE_METRICS_JSON:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "task_type": "classification",
                "model_family": "Multinomial Naive Bayes",
                "dataset_path": str(DATA_PATH.as_posix()),
            },
            "baseline": {**baseline_metrics.to_dict(), "strategy": "majority_class", "majority_class": str(majority)},
            "naive_bayes": model_metrics.to_dict(),
        }
        (RESULTS_DIR / "metrics_interactive.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Start interactive demo
    interactive_loop(model)


if __name__ == "__main__":
    main()
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# global
RANDOM_STATE = 42
TEST_SIZE = 0.2


# Evaluate
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": float(sensitivity), # Minimize malignancies
        "specificity": float(specificity), # Control escalations
        "ppv": float(ppv), # Interpretability (clinics)
        "npv": float(npv), # Calibration (reliability)
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

# Risk Tolerance, Safety Constraints
# Minimize false negatives, choose threshold with lowest false positives
def choose_threshold(y_true, y_prob, min_sens=0.97):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = None
    for fp_rate, tp_rate, t in zip(fpr, tpr, thr):
        if tp_rate >= min_sens:
            if best is None or fp_rate < best[0]:
                best = (fp_rate, tp_rate, t)
    if best is None:
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(thr[idx])
    return float(best[2])

# Create human-readable results
def pretty_print(name, m, threshold=None):
    print(f"\n=== {name} ===")
    if threshold is not None:
        print(f"Threshold: {threshold:.3f}")
    print(f"Confusion Matrix (malignant=positive): TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
    print(f"Accuracy:     {m['accuracy']:.3f}")
    print(f"ROC AUC:      {m['roc_auc']:.3f}")
    print(f"Sensitivity:  {m['sensitivity']:.3f}  (catch malignant)")
    print(f"Specificity:  {m['specificity']:.3f}  (avoid unnecessary escalation)")
    print(f"PPV:          {m['ppv']:.3f}")
    print(f"NPV:          {m['npv']:.3f}")
    print(f"F1:           {m['f1']:.3f}")
    print(f"Brier:        {m['brier']:.3f}  (calibration)")

def main():
    X, y = load_breast_cancer(return_X_y=True)
    
    # 1 - malignant, 0 - benign
    y_malignant = (y == 0).astype(int)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_malignant, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_malignant
    )

    # Baseline
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])
    
    baseline.fit(X_train, y_train)
    b_prob = baseline.predict_proba(X_test)[:, 1]
    b_pred = (b_prob >= 0.5).astype(int)
    b_metrics = compute_metrics(y_test, b_pred, b_prob)
    pretty_print("Logistic Regression (baseline)", b_metrics)

    # Boosted model (XGBoost if installed, else fallback)
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            objective="binary:logistic", # logistic loss
            n_estimators=300, # number of trees; not too big (noise reduction)
            learning_rate=0.05, # step size for trees
            max_depth=3, # max number of splits per tree
            subsample=0.9, # sample size training visibility (what each tree sees)
            colsample_bytree=0.9, # feature visibility
            eval_metric="logloss", 
            random_state=RANDOM_STATE, # Random variability for starting dataset
            n_jobs=4, # run 4 parallel jobs
            reg_lambda=1.0 
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    model.fit(X_train, y_train)
    p_prob = model.predict_proba(X_test)[:, 1]
    p_pred = (p_prob >= 0.5).astype(int)
    p_metrics = compute_metrics(y_test, p_pred, p_prob)
    pretty_print("Boosted Trees", p_metrics)

    # Clinically oriented threshold selection
    thr = choose_threshold(y_test, p_prob, min_sens=0.97)
    p_pred_thr = (p_prob >= thr).astype(int)
    p_metrics_thr = compute_metrics(y_test, p_pred_thr, p_prob)
    pretty_print("Boosted Trees (high-sensitivity operating point)", p_metrics_thr, threshold=thr)

if __name__ == "__main__":
    main()
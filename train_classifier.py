"""
train_classifier.py
--------------------
Trains a multi-class root-cause classifier for failed payment transactions.

Features used (deliberately EXCLUDES true_root_cause and _customer_quality,
which are ground truth / latent variables only used for evaluation):
  - decline_code (categorical, one-hot)
  - payment_method (categorical, one-hot)
  - amount, hour_of_day, is_recurring, prior_failures_7d
  - past_success_rate, past_attempts_30d, account_age_days, avg_txn_amount

Model: GradientBoostingClassifier (sklearn) wrapped in a ColumnTransformer
pipeline. Chosen over a black-box deep model because:
  - tabular data with < 10k rows
  - need per-class precision/recall + feature importances for the writeup
  - fast to train/iterate, no GPU dependency

Outputs:
  - trained pipeline pickled to ../outputs/root_cause_model.pkl
  - classification report (per-class precision/recall/F1) printed + saved
  - confusion matrix figure saved to ../outputs/confusion_matrix.png
  - predictions on the held-out test set saved to ../data/test_predictions.csv
    (used downstream by the recovery-action engine)
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "../data/failed_transactions.csv"
MODEL_OUT = "../outputs/root_cause_model.pkl"
REPORT_OUT = "../outputs/classification_report.json"
CM_FIG_OUT = "../outputs/confusion_matrix.png"
TEST_PRED_OUT = "../data/test_predictions.csv"

CATEGORICAL = ["decline_code", "payment_method"]
NUMERIC = [
    "amount", "hour_of_day", "is_recurring", "prior_failures_7d",
    "past_success_rate", "past_attempts_30d", "account_age_days", "avg_txn_amount",
]
TARGET = "true_root_cause"


def build_pipeline():
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.08,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    classes = pipe.named_steps["clf"].classes_

    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted root cause")
    ax.set_ylabel("True root cause")
    ax.set_title("Root Cause Classifier — Confusion Matrix (held-out test set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(CM_FIG_OUT, dpi=150)
    print(f"Saved confusion matrix to {CM_FIG_OUT}")

    # Save test set with predictions + probabilities for downstream use
    test_out = df.loc[idx_test].copy()
    test_out["pred_root_cause"] = y_pred
    for i, c in enumerate(classes):
        test_out[f"proba_{c}"] = y_proba[:, i]
    test_out.to_csv(TEST_PRED_OUT, index=False)
    print(f"Saved test predictions to {TEST_PRED_OUT}")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved model to {MODEL_OUT}")

    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]
    print(f"\nAccuracy: {acc:.3f}  |  Macro F1: {macro_f1:.3f}")


if __name__ == "__main__":
    main()

"""Common evaluation script: accuracy, precision, recall, F1, confusion matrix.

Tumor (label 1) is the positive class, so recall = tumor recall (sensitivity).
Works for any model that outputs a sigmoid probability.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)  # NOT argmax: single sigmoid output
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=["no_tumor", "tumor"], output_dict=True, zero_division=0
        ),
    }


def evaluate_generator(model, gen, y_true, name, out_dir="results", threshold=0.5):
    """Evaluate on a NON-shuffled generator; y_true = labels in the same order (from the CSV)."""
    gen.reset()
    y_prob = model.predict(gen, verbose=0)
    return _finish(y_true, y_prob, name, out_dir, threshold)


def evaluate_arrays(model, X, y, name, out_dir="results", threshold=0.5):
    return _finish(y, model.predict(X, verbose=0), name, out_dir, threshold)


def _finish(y_true, y_prob, name, out_dir, threshold):
    m = compute_metrics(y_true, y_prob, threshold)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(m, indent=2))
    cm = m["confusion_matrix"]
    print(
        f"[{name}] acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
        f"tumor_recall={m['recall']:.4f} f1={m['f1']:.4f}  CM(rows=true, cols=pred)={cm}"
    )
    return m

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_classes: int | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        ),
    }
    if y_prob is not None:
        try:
            labels = list(range(n_classes or y_prob.shape[1]))
            metrics["auc_macro"] = float(
                roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro")
                * 100.0
            )
        except ValueError:
            metrics["auc_macro"] = float("nan")
    return metrics


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return f1_score(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    ) * 100.0


def feature_center(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("Cannot compute a feature center for an empty array")
    return np.mean(x, axis=0)

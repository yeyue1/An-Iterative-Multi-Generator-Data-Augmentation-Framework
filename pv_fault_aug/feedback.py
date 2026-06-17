from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass
class FeedbackSignals:
    error_rate: np.ndarray
    confidence: np.ndarray
    scarcity: np.ndarray
    class_counts: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def compute_feedback(
    classifier: "torch.nn.Module",
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_aug: np.ndarray,
    y_aug: np.ndarray,
    n_classes: int,
    device: "torch.device",
) -> FeedbackSignals:
    from .models.classifier import predict_proba

    y_prob = predict_proba(classifier, x_val, device)
    y_pred = y_prob.argmax(axis=1)

    error = np.zeros(n_classes, dtype=np.float64)
    confidence = np.zeros(n_classes, dtype=np.float64)
    log_c = np.log(max(n_classes, 2))

    for class_id in range(n_classes):
        mask = y_val == class_id
        if not np.any(mask):
            error[class_id] = 1.0
            confidence[class_id] = 0.0
            continue
        error[class_id] = 1.0 - np.mean(y_pred[mask] == class_id)
        probs = np.clip(y_prob[mask], 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        confidence[class_id] = 1.0 - float(np.mean(entropy) / log_c)

    counts = np.bincount(y_aug, minlength=n_classes).astype(np.float64)
    max_count = max(float(counts.max()), 1.0)
    scarcity = (max_count - counts) / max_count

    return FeedbackSignals(
        error_rate=error,
        confidence=confidence,
        scarcity=scarcity,
        class_counts=counts,
        y_pred=y_pred,
        y_prob=y_prob,
    )

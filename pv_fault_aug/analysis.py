from __future__ import annotations

"""Error, robustness, and physical-consistency analyses.

These helpers implement the additional evaluations reported after the main
comparison: class confusion, operating-regime performance, measurement and
domain perturbations, group-held-out shifts, and physical I-V parameters.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .config import DataConfig, ExperimentConfig
from .data import DatasetSplit
from .metrics import classification_metrics


# --------------------------------------------------------------------------- #
# Confusion and class-pair analysis
# --------------------------------------------------------------------------- #
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        counts[t, p] += 1
    return counts


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-normalised confusion matrix in percent."""
    counts = confusion_counts(y_true, y_pred, n_classes)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums * 100.0


def normalize_confusion_counts(counts: np.ndarray) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums * 100.0


def class_pair_table_from_counts(
    counts: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> list[dict[str, float | int | str]]:
    """Off-diagonal confusions from pooled confusion counts.

    Each row reports the pooled count of true-class records predicted as the
    competing class, the pooled support of the true class, and the percentage
    of that support, mirroring Table 9 of the manuscript.
    """
    counts = np.asarray(counts, dtype=np.float64)
    support = counts.sum(axis=1)
    pairs = []
    for true_id in range(counts.shape[0]):
        for pred_id in range(counts.shape[1]):
            if true_id == pred_id or counts[true_id, pred_id] <= 0:
                continue
            pairs.append(
                {
                    "true": class_names[true_id],
                    "predicted": class_names[pred_id],
                    "count": int(counts[true_id, pred_id]),
                    "support": int(support[true_id]),
                    "rate": float(
                        counts[true_id, pred_id] / max(support[true_id], 1) * 100.0
                    ),
                }
            )
    pairs.sort(key=lambda item: item["rate"], reverse=True)
    return pairs[:top_k]


def class_pair_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_label: dict[int, str],
    top_k: int = 10,
) -> list[dict[str, float | int | str]]:
    """Convenience wrapper around :func:`class_pair_table_from_counts`."""
    counts = confusion_counts(y_true, y_pred, len(index_to_label))
    names = [index_to_label[idx] for idx in range(len(index_to_label))]
    return class_pair_table_from_counts(counts, names, top_k)


# --------------------------------------------------------------------------- #
# Operating-regime analysis
# --------------------------------------------------------------------------- #
def regime_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    values: np.ndarray,
    bins: list[tuple[float, float]],
    n_classes: int,
) -> list[dict[str, float | int | str]]:
    """F1-macro and sample count inside each metadata bin (no retraining)."""
    values = np.asarray(values, dtype=np.float64)
    rows = []
    for low, high in bins:
        mask = (values >= low) & (values < high)
        if not np.any(mask):
            continue
        metrics = classification_metrics(y_true[mask], y_pred[mask], None, n_classes)
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "samples": int(mask.sum()),
                "f1_macro": metrics["f1_macro"],
            }
        )
    return rows


IRRADIANCE_BINS = [(600.0, 700.0), (700.0, 800.0), (800.0, 900.0), (900.0, 1000.0)]
TEMPERATURE_BINS = [(15.0, 25.0), (25.0, 35.0), (35.0, 45.0)]


# --------------------------------------------------------------------------- #
# Measurement and domain perturbations
# --------------------------------------------------------------------------- #
def apply_perturbation(
    x: np.ndarray,
    kind: str,
    level: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    if kind == "none":
        return x.copy()
    if kind == "gaussian":
        power = float(np.mean(x**2))
        snr = max(level, 1e-6)
        sigma = np.sqrt(power / (10.0 ** (snr / 10.0)))
        return np.clip(x + rng.normal(0.0, sigma, x.shape).astype(np.float32), 0.0, 1.0)
    if kind == "missing":
        out = x.copy()
        n_points = x.shape[1]
        n_missing = max(1, int(round(level * n_points)))
        for row in range(out.shape[0]):
            idx = np.sort(rng.choice(n_points, size=n_missing, replace=False))
            valid = np.setdiff1d(np.arange(n_points), idx)
            if len(valid) >= 2:
                out[row, idx] = np.interp(idx, valid, out[row, valid])
        return out
    if kind == "gain":
        return np.clip(x * (1.0 + level), 0.0, 1.0)
    if kind == "offset":
        return np.clip(x + level, 0.0, 1.0)
    raise ValueError(f"Unknown perturbation: {kind}")


PERTURBATION_SETTINGS = (
    [("none", 0.0)]
    + [("gaussian", snr) for snr in (30.0, 20.0, 10.0)]
    + [("missing", frac) for frac in (0.05, 0.10, 0.20)]
    + [("gain", level) for level in (0.02, 0.05)]
    + [("offset", level) for level in (0.01, 0.03)]
)


def evaluate_perturbations(
    predict_fn,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    settings=PERTURBATION_SETTINGS,
    seed: int = 0,
) -> list[dict[str, float | str]]:
    rows = []
    for kind, level in settings:
        x_perturbed = apply_perturbation(x_test, kind, level, seed=seed)
        prob = predict_fn(x_perturbed)
        pred = prob.argmax(axis=1)
        metrics = classification_metrics(y_test, pred, prob, n_classes)
        rows.append(
            {
                "kind": kind,
                "level": float(level),
                "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Group-held-out evaluation (unseen modules / operating conditions)
# --------------------------------------------------------------------------- #
def make_group_heldout_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: DataConfig,
    seed: int,
    test_fraction: float = 0.25,
) -> DatasetSplit:
    """Hold out whole groups for testing, then split the rest into train/val."""
    from sklearn.model_selection import train_test_split

    from .data import DatasetSplit, _subset_metadata

    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required for a group-held-out split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_test = max(1, int(round(len(unique_groups) * test_fraction)))
    test_groups = set(shuffled[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    trainval_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    rel_val = config.val_ratio / (config.val_ratio + config.test_ratio)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        train_size=1.0 - rel_val,
        random_state=seed,
        stratify=y[trainval_idx],
    )

    x_train_raw = x[train_idx]
    x_min = x_train_raw.min(axis=0, keepdims=True)
    x_range = x_train_raw.max(axis=0, keepdims=True) - x_min
    x_range = np.where(x_range > 1e-12, x_range, 1.0)

    def scale(arr: np.ndarray) -> np.ndarray:
        return np.clip((arr - x_min) / x_range, 0.0, 1.0).astype(np.float32)

    label_to_index = {name: idx for idx, name in enumerate(config.class_names)}
    index_to_label = {idx: name for name, idx in label_to_index.items()}
    return DatasetSplit(
        x_train=scale(x[train_idx]),
        y_train=y[train_idx].astype(np.int64),
        x_val=scale(x[val_idx]),
        y_val=y[val_idx].astype(np.int64),
        x_test=scale(x[test_idx]),
        y_test=y[test_idx].astype(np.int64),
        label_to_index=label_to_index,
        index_to_label=index_to_label,
    )


# --------------------------------------------------------------------------- #
# Physical I-V parameters
# --------------------------------------------------------------------------- #
def physical_parameters(
    curves: np.ndarray,
    voltage_max: float | None = None,
    current_max: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute Voc, Isc, Pmax, and fill factor for normalised I-V curves.

    ``curves`` are expected on a normalised [0, 1] grid with current samples
    ordered from short circuit (index 0) to open circuit (last index).  When
    ``voltage_max`` / ``current_max`` are given, the values are mapped back to
    physical units.
    """
    curves = np.asarray(curves, dtype=np.float64)
    n_points = curves.shape[1]
    voltage = np.linspace(0.0, 1.0, n_points)
    current = curves
    i_sc = current[:, 0].copy()
    v_oc = np.zeros(len(curves), dtype=np.float64)
    for row in range(len(curves)):
        below = np.where(current[row] <= 0.01 * max(i_sc[row], 1e-9))[0]
        if len(below) == 0:
            v_oc[row] = 1.0
        else:
            idx = int(below[0])
            if idx == 0:
                v_oc[row] = 0.0
            else:
                v0, v1 = voltage[idx - 1], voltage[idx]
                i0, i1 = current[row, idx - 1], current[row, idx]
                frac = 0.0 if i0 == i1 else (i0 - 0.0) / (i0 - i1)
                v_oc[row] = v0 + frac * (v1 - v0)
    power = current * voltage[None, :]
    p_max = power.max(axis=1)
    denominator = np.maximum(v_oc * i_sc, 1e-6)
    fill_factor = np.clip(p_max / denominator, 0.0, 1.0)
    if voltage_max is not None and current_max is not None:
        v_oc = v_oc * voltage_max
        i_sc = i_sc * current_max
        p_max = p_max * voltage_max * current_max
    return {
        "voc": v_oc,
        "isc": i_sc,
        "pmax": p_max,
        "fill_factor": fill_factor,
    }


def summarize_physical(
    real_curves: np.ndarray,
    generated_curves: np.ndarray,
    voltage_max: float | None = None,
    current_max: float | None = None,
) -> dict[str, dict[str, float]]:
    real = physical_parameters(real_curves, voltage_max, current_max)
    generated = physical_parameters(generated_curves, voltage_max, current_max)
    summary: dict[str, dict[str, float]] = {}
    for key in real:
        real_mean = float(np.nanmean(real[key]))
        gen_mean = float(np.nanmean(generated[key]))
        if not np.isfinite(real_mean) or np.isfinite(real_mean) and abs(real_mean) < 1e-9:
            rel_error = float("nan")
        else:
            rel_error = abs(gen_mean - real_mean) / abs(real_mean) * 100.0
        summary[key] = {
            "real_mean": real_mean,
            "generated_mean": gen_mean,
            "relative_error_pct": rel_error,
        }
    return summary


def predict_with_model(config: ExperimentConfig, model, x: np.ndarray, device: torch.device) -> np.ndarray:
    from .models.classifier import predict_proba

    return predict_proba(model, x, device)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score

from .config import ClassifierConfig, FrameworkConfig
from .models.classifier import predict_proba, train_classifier


@dataclass
class CandidateBatch:
    x: np.ndarray
    y: np.ndarray
    source: np.ndarray


@dataclass
class ScreeningResult:
    accepted_x: np.ndarray
    accepted_y: np.ndarray
    accepted_source: np.ndarray
    mmd_by_class: dict[int, float]
    wd_by_class: dict[int, float]
    f1_gain: float
    accepted_counts_by_source: dict[str, int]


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("inf")
    if gamma is None:
        merged = np.vstack([x, y])
        var = float(np.var(merged))
        gamma = 1.0 / max(2.0 * var * merged.shape[1], 1e-8)
    xx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)).mean()
    yy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)).mean()
    xy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)).mean()
    return float(xx + yy - 2.0 * xy)


def feature_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("inf")
    n = min(len(x), len(y))
    x_sorted = np.sort(x, axis=0)
    y_sorted = np.sort(y, axis=0)
    x_idx = np.linspace(0, len(x_sorted) - 1, n).round().astype(int)
    y_idx = np.linspace(0, len(y_sorted) - 1, n).round().astype(int)
    return float(np.mean(np.abs(x_sorted[x_idx] - y_sorted[y_idx])))


def merge_candidates(results: list, feature_dim: int) -> CandidateBatch:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for result in results:
        if len(result.samples) == 0:
            continue
        xs.append(result.samples)
        ys.append(result.labels)
        sources.append(np.asarray([result.name] * len(result.labels), dtype=object))
    if not xs:
        return CandidateBatch(
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
        )
    return CandidateBatch(np.vstack(xs).astype(np.float32), np.concatenate(ys), np.concatenate(sources))


def distributional_gate(
    x_real: np.ndarray,
    y_real: np.ndarray,
    candidates: CandidateBatch,
    n_classes: int,
    config: FrameworkConfig,
) -> tuple[CandidateBatch, dict[int, float], dict[int, float]]:
    keep = np.zeros(len(candidates.y), dtype=bool)
    mmd_by_class: dict[int, float] = {}
    wd_by_class: dict[int, float] = {}
    for class_id in range(n_classes):
        real_mask = y_real == class_id
        cand_mask = candidates.y == class_id
        if not np.any(cand_mask):
            continue
        mmd = rbf_mmd(x_real[real_mask], candidates.x[cand_mask])
        wd = feature_wasserstein(x_real[real_mask], candidates.x[cand_mask])
        mmd_by_class[class_id] = mmd
        wd_by_class[class_id] = wd
        if mmd <= config.mmd_threshold and wd <= config.wasserstein_threshold:
            keep[cand_mask] = True
    return (
        CandidateBatch(candidates.x[keep], candidates.y[keep], candidates.source[keep]),
        mmd_by_class,
        wd_by_class,
    )


def _evaluate_candidate_gain(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    current_classifier: torch.nn.Module,
    candidates: CandidateBatch,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    device: torch.device,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    if len(candidates.y) == 0:
        current_prob = predict_proba(current_classifier, x_val, device)
        current_pred = current_prob.argmax(axis=1)
        current_class_f1 = f1_score(
            y_val,
            current_pred,
            labels=list(range(n_classes)),
            average=None,
            zero_division=0,
        )
        current_macro = f1_score(y_val, current_pred, average="macro", zero_division=0)
        return current_class_f1, current_macro, current_class_f1.copy(), current_macro

    current_prob = predict_proba(current_classifier, x_val, device)
    current_pred = current_prob.argmax(axis=1)
    current_f1 = f1_score(y_val, current_pred, average="macro", zero_division=0)
    current_class_f1 = f1_score(
        y_val,
        current_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )

    x_tmp = np.vstack([x_train, candidates.x])
    y_tmp = np.concatenate([y_train, candidates.y])
    tmp_model = train_classifier(
        x_tmp,
        y_tmp,
        x_val,
        y_val,
        input_dim,
        n_classes,
        classifier_config,
        device,
        epochs=classifier_config.feedback_epochs,
    )
    tmp_prob = predict_proba(tmp_model, x_val, device)
    tmp_pred = tmp_prob.argmax(axis=1)
    tmp_f1 = f1_score(y_val, tmp_pred, average="macro", zero_division=0)
    tmp_class_f1 = f1_score(
        y_val,
        tmp_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )
    return current_class_f1, float(current_f1), tmp_class_f1, float(tmp_f1)


def performance_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    current_classifier: torch.nn.Module,
    candidates: CandidateBatch,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    framework_config: FrameworkConfig,
    device: torch.device,
) -> tuple[bool, float]:
    current_class_f1, current_macro, tmp_class_f1, tmp_macro = _evaluate_candidate_gain(
        x_train,
        y_train,
        x_val,
        y_val,
        current_classifier,
        candidates,
        input_dim,
        n_classes,
        classifier_config,
        device,
    )
    class_gain = float(np.mean(tmp_class_f1 - current_class_f1))
    macro_gain = float(tmp_macro - current_macro)
    utility = (
        framework_config.gate_utility_weight * class_gain
        + (1.0 - framework_config.gate_utility_weight) * macro_gain
    )
    return (
        utility >= framework_config.min_f1_gain
        and macro_gain >= -framework_config.macro_f1_degradation_tolerance,
        macro_gain,
    )


def class_bound_performance_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    current_classifier: torch.nn.Module,
    candidates: CandidateBatch,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    framework_config: FrameworkConfig,
    device: torch.device,
) -> tuple[CandidateBatch, float]:
    if len(candidates.y) == 0:
        return candidates, 0.0

    keep = np.zeros(len(candidates.y), dtype=bool)
    accepted_macro_gains: list[float] = []

    base_prob = predict_proba(current_classifier, x_val, device)
    base_pred = base_prob.argmax(axis=1)
    base_class_f1 = f1_score(
        y_val,
        base_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )
    base_macro = float(f1_score(y_val, base_pred, average="macro", zero_division=0))

    for class_id in range(n_classes):
        class_mask = candidates.y == class_id
        if not np.any(class_mask):
            continue

        class_candidates = CandidateBatch(
            candidates.x[class_mask],
            candidates.y[class_mask],
            candidates.source[class_mask],
        )
        class_gains: list[float] = []
        macro_gains: list[float] = []

        for eval_idx in range(framework_config.gate_eval_runs):
            torch.manual_seed(10_000 + class_id * 101 + eval_idx)
            np.random.seed(10_000 + class_id * 101 + eval_idx)
            _, _, tmp_class_f1, tmp_macro = _evaluate_candidate_gain(
                x_train,
                y_train,
                x_val,
                y_val,
                current_classifier,
                class_candidates,
                input_dim,
                n_classes,
                classifier_config,
                device,
            )
            class_gains.append(float(tmp_class_f1[class_id] - base_class_f1[class_id]))
            macro_gains.append(float(tmp_macro - base_macro))

        class_gain = float(np.mean(class_gains))
        macro_gain = float(np.mean(macro_gains))
        utility = (
            framework_config.gate_utility_weight * class_gain
            + (1.0 - framework_config.gate_utility_weight) * macro_gain
        )
        adaptive_threshold = framework_config.min_f1_gain * max(
            0.25, 1.0 - float(base_class_f1[class_id])
        )
        if (
            utility >= adaptive_threshold
            and macro_gain >= -framework_config.macro_f1_degradation_tolerance
        ):
            keep[class_mask] = True
            accepted_macro_gains.append(macro_gain)

    accepted = CandidateBatch(candidates.x[keep], candidates.y[keep], candidates.source[keep])
    return accepted, float(np.mean(accepted_macro_gains)) if accepted_macro_gains else 0.0


def dual_gate_screen(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    current_classifier: torch.nn.Module,
    candidates: CandidateBatch,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    framework_config: FrameworkConfig,
    device: torch.device,
) -> ScreeningResult:
    if not framework_config.use_dual_gate:
        _, f1_gain = performance_gate(
            x_train,
            y_train,
            x_val,
            y_val,
            current_classifier,
            candidates,
            input_dim,
            n_classes,
            classifier_config,
            framework_config,
            device,
        )
        counts = {
            str(source): int(np.sum(candidates.source == source))
            for source in np.unique(candidates.source)
        }
        return ScreeningResult(
            accepted_x=candidates.x,
            accepted_y=candidates.y,
            accepted_source=candidates.source,
            mmd_by_class={},
            wd_by_class={},
            f1_gain=f1_gain,
            accepted_counts_by_source=counts,
        )

    filtered, mmd_by_class, wd_by_class = distributional_gate(
        x_train, y_train, candidates, n_classes, framework_config
    )
    accepted, f1_gain = class_bound_performance_gate(
        x_train,
        y_train,
        x_val,
        y_val,
        current_classifier,
        filtered,
        input_dim,
        n_classes,
        classifier_config,
        framework_config,
        device,
    )

    counts = {
        str(source): int(np.sum(accepted.source == source))
        for source in np.unique(accepted.source)
    }
    return ScreeningResult(
        accepted_x=accepted.x,
        accepted_y=accepted.y,
        accepted_source=accepted.source,
        mmd_by_class=mmd_by_class,
        wd_by_class=wd_by_class,
        f1_gain=f1_gain,
        accepted_counts_by_source=counts,
    )

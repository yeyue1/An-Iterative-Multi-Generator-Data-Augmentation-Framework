from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score, pairwise_distances

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
    # Macro F1 of the R_eval matched baseline models trained on the current
    # augmented set.  They are reused for the paired round-level gain.
    base_macro_f1: list[float] = field(default_factory=list)


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Biased RBF-kernel MMD with a median-heuristic scale."""
    if len(x) == 0 or len(y) == 0:
        return float("inf")
    if gamma is None:
        merged = np.vstack([x, y])
        var = float(np.var(merged))
        gamma = 1.0 / max(2.0 * var * merged.shape[1], 1e-8)
    # pairwise_distances avoids the n x m x d temporary created by
    # broadcasting, which is prohibitive for the NS class.
    xx = np.exp(-gamma * pairwise_distances(x, x, metric="sqeuclidean")).mean()
    yy = np.exp(-gamma * pairwise_distances(y, y, metric="sqeuclidean")).mean()
    xy = np.exp(-gamma * pairwise_distances(x, y, metric="sqeuclidean")).mean()
    return float(xx + yy - 2.0 * xy)


def feature_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """Mean one-dimensional Wasserstein distance across feature dimensions."""
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
    return CandidateBatch(
        np.vstack(xs).astype(np.float32),
        np.concatenate(ys),
        np.concatenate(sources),
    )


def distributional_gate(
    x_real: np.ndarray,
    y_real: np.ndarray,
    candidates: CandidateBatch,
    n_classes: int,
    config: FrameworkConfig,
) -> tuple[CandidateBatch, dict[int, float], dict[int, float]]:
    """Gate 1: class-wise MMD and Wasserstein tests, Eqs. (mmd) and (wd)."""
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


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _train_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Train one evaluation model and return per-class and macro validation F1."""
    _set_seed(seed)
    model = train_classifier(
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim,
        n_classes,
        classifier_config,
        device,
        epochs=classifier_config.feedback_epochs,
        architecture=classifier_config.architecture,
    )
    prob = predict_proba(model, x_val, device)
    pred = prob.argmax(axis=1)
    per_class = f1_score(
        y_val,
        pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )
    macro = float(f1_score(y_val, pred, average="macro", zero_division=0))
    return per_class, macro


def class_bound_performance_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    candidates: CandidateBatch,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    framework_config: FrameworkConfig,
    device: torch.device,
    seed_base: int,
) -> tuple[CandidateBatch, float, list[float]]:
    """Gate 2: class-bound marginal gain, Eq. (gate2utility).

    Both the baseline and the temporary classifier are retrained for each of
    the R_eval matched evaluation runs with the same seed, so the reported
    class gain is a paired mean rather than a difference between a single
    baseline model and several temporary models.
    """
    if len(candidates.y) == 0:
        return candidates, 0.0, []

    n_eval = framework_config.gate_eval_runs
    base_per_class: list[np.ndarray] = []
    base_macro: list[float] = []
    for eval_idx in range(n_eval):
        per_class, macro = _train_and_score(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim,
            n_classes,
            classifier_config,
            device,
            seed_base + eval_idx,
        )
        base_per_class.append(per_class)
        base_macro.append(macro)

    keep = np.zeros(len(candidates.y), dtype=bool)
    accepted_macro_gains: list[float] = []

    for class_id in range(n_classes):
        class_mask = candidates.y == class_id
        if not np.any(class_mask):
            continue

        class_candidates = CandidateBatch(
            candidates.x[class_mask],
            candidates.y[class_mask],
            candidates.source[class_mask],
        )
        x_tmp = np.vstack([x_train, class_candidates.x])
        y_tmp = np.concatenate([y_train, class_candidates.y])

        class_gains: list[float] = []
        macro_gains: list[float] = []
        for eval_idx in range(n_eval):
            tmp_per_class, tmp_macro = _train_and_score(
                x_tmp,
                y_tmp,
                x_val,
                y_val,
                input_dim,
                n_classes,
                classifier_config,
                device,
                seed_base + eval_idx,
            )
            class_gains.append(
                float(tmp_per_class[class_id] - base_per_class[eval_idx][class_id])
            )
            macro_gains.append(float(tmp_macro - base_macro[eval_idx]))

        class_gain = float(np.mean(class_gains))
        macro_gain = float(np.mean(macro_gains))
        utility = (
            framework_config.gate_utility_weight * class_gain
            + (1.0 - framework_config.gate_utility_weight) * macro_gain
        )
        base_class_f1 = float(np.mean([row[class_id] for row in base_per_class]))
        adaptive_threshold = framework_config.min_f1_gain * max(
            0.25, 1.0 - base_class_f1
        )
        if (
            utility >= adaptive_threshold
            and macro_gain >= -framework_config.macro_f1_degradation_tolerance
        ):
            keep[class_mask] = True
            accepted_macro_gains.append(macro_gain)

    accepted = CandidateBatch(
        candidates.x[keep], candidates.y[keep], candidates.source[keep]
    )
    mean_accepted_macro_gain = (
        float(np.mean(accepted_macro_gains)) if accepted_macro_gains else 0.0
    )
    return accepted, mean_accepted_macro_gain, base_macro


def evaluate_round_f1_gain(
    x_base: np.ndarray,
    y_base: np.ndarray,
    x_updated: np.ndarray,
    y_updated: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    framework_config: FrameworkConfig,
    device: torch.device,
    seed_base: int,
    base_macro_f1: list[float] | None = None,
    fallback_f1: float = 0.0,
) -> tuple[float, float]:
    """Round-level gain, Eq. (deltaf1round), averaged over R_eval runs.

    ``base_macro_f1`` contains the R_eval baseline scores already produced by
    Gate 2 for the same seeds.  When it is unavailable (e.g. the no-dual-gate
    ablation), the baseline models are retrained here.
    """
    if len(y_updated) == len(y_base):
        mean_base = float(np.mean(base_macro_f1)) if base_macro_f1 else fallback_f1
        return 0.0, mean_base

    n_eval = framework_config.gate_eval_runs
    if base_macro_f1 is None or len(base_macro_f1) != n_eval:
        base_macro_f1 = [
            _train_and_score(
                x_base,
                y_base,
                x_val,
                y_val,
                input_dim,
                n_classes,
                classifier_config,
                device,
                seed_base + eval_idx,
            )[1]
            for eval_idx in range(n_eval)
        ]

    gains: list[float] = []
    updated_scores: list[float] = []
    for eval_idx in range(n_eval):
        _, updated_macro = _train_and_score(
            x_updated,
            y_updated,
            x_val,
            y_val,
            input_dim,
            n_classes,
            classifier_config,
            device,
            seed_base + eval_idx,
        )
        gains.append(float(updated_macro - base_macro_f1[eval_idx]))
        updated_scores.append(float(updated_macro))
    return float(np.mean(gains)), float(np.mean(updated_scores))


def dual_gate_screen(
    x_aug: np.ndarray,
    y_aug: np.ndarray,
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    candidates: CandidateBatch,
    input_dim: int,
    n_classes: int,
    classifier_config: ClassifierConfig,
    framework_config: FrameworkConfig,
    device: torch.device,
    seed_base: int,
) -> ScreeningResult:
    """Apply Gate 1 and Gate 2 in sequence.

    Gate 1 compares the candidates with the real samples P_{r,k}; Gate 2
    trains the temporary classifiers on D_aug (``x_aug``) plus the class-wise
    candidates, as in Eq. (gate2utility). When ``use_dual_gate`` is false (the
    corresponding ablation), every candidate is admitted and only the
    round-level gain of the convergence module remains active, matching the
    ablation definition in the paper.
    """
    if not framework_config.use_dual_gate:
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
            f1_gain=0.0,
            accepted_counts_by_source=counts,
            base_macro_f1=[],
        )

    filtered, mmd_by_class, wd_by_class = distributional_gate(
        x_real, y_real, candidates, n_classes, framework_config
    )
    accepted, f1_gain, base_macro_f1 = class_bound_performance_gate(
        x_aug,
        y_aug,
        x_val,
        y_val,
        filtered,
        input_dim,
        n_classes,
        classifier_config,
        framework_config,
        device,
        seed_base,
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
        base_macro_f1=base_macro_f1,
    )
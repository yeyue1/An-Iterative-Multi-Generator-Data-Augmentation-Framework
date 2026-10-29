from __future__ import annotations

"""Baseline augmentation strategies used in the comparison table.

Every method receives the same :class:`DatasetSplit` (identical real train /
validation / test partitions) and returns an :class:`ExperimentOutcome` with
the held-out test metrics, per-class F1, wall-clock time, and predictions.
Synthetic samples are only ever added to the training partition.
"""

from dataclasses import dataclass, field, replace
import time

import numpy as np
import torch

from .allocation import compute_class_quotas
from .config import ExperimentConfig
from .data import DatasetSplit
from .metrics import classification_metrics, per_class_f1
from .models.classifier import predict_proba, train_classifier
from .models.generators import build_generators


@dataclass
class ExperimentOutcome:
    method: str
    scenario: str
    final_metrics: dict[str, float]
    per_class_f1: dict[str, float]
    accepted_samples: int = 0
    iterations: int = 0
    elapsed_sec: float = 0.0
    y_true: np.ndarray | None = field(default=None, repr=False)
    y_pred: np.ndarray | None = field(default=None, repr=False)
    y_prob: np.ndarray | None = field(default=None, repr=False)


def _n_classes(split: DatasetSplit) -> int:
    return len(split.label_to_index)


def _seed_everything(seed: int) -> None:
    """Match the seed handling of the closed-loop runs.

    All methods share the same split and are run with the same five seeds;
    every stochastic component of a baseline is reset from that seed so that
    differences in Table 5 reflect the augmentation strategy rather than
    initialization noise.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def _train_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    split: DatasetSplit,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    n_classes = _n_classes(split)
    model = train_classifier(
        x_train,
        y_train,
        split.x_val,
        split.y_val,
        split.x_train.shape[1],
        n_classes,
        config.classifier,
        device,
        architecture=config.classifier.architecture,
    )
    prob = predict_proba(model, split.x_test, device)
    pred = prob.argmax(axis=1)
    metrics = classification_metrics(split.y_test, pred, prob, n_classes)
    class_f1 = per_class_f1(split.y_test, pred, n_classes)
    per_class = {
        split.index_to_label[idx]: float(class_f1[idx]) for idx in range(n_classes)
    }
    return metrics, per_class, split.y_test.copy(), pred, prob


def _run_epochs(config: ExperimentConfig) -> int:
    return min(config.generators.epochs, config.generators.quick_epochs)


def _fit_generators(
    split: DatasetSplit,
    config: ExperimentConfig,
    device: torch.device,
    x_train: np.ndarray,
    y_train: np.ndarray,
):
    input_dim = split.x_train.shape[1]
    generators = build_generators(
        input_dim, _n_classes(split), config.generators, device
    )
    for generator in generators:
        generator.fit(x_train, y_train, epochs=_run_epochs(config))
    return generators


def _deficit_quotas(
    y: np.ndarray,
    n_classes: int,
    budget: int,
) -> dict[int, int]:
    """Allocate a fixed total budget in proportion to class deficits."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    deficits = counts.max() - counts
    if deficits.sum() <= 0:
        deficits = np.ones(n_classes, dtype=np.float64)
    raw = budget * deficits / deficits.sum()
    quotas = np.floor(raw).astype(int)
    remainder = int(budget - quotas.sum())
    if remainder > 0:
        order = np.argsort(-(raw - quotas))
        quotas[order[:remainder]] += 1
    return {k: int(v) for k, v in enumerate(quotas) if v > 0}


def smote_oversample(
    x: np.ndarray,
    y: np.ndarray,
    target_counts: dict[int, int],
    k: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal SMOTE implementation (no external dependency)."""
    rng = np.random.default_rng(seed)
    xs = [x]
    ys = [y]
    for class_id, target in target_counts.items():
        idx = np.where(y == class_id)[0]
        n_current = len(idx)
        if n_current == 0 or target <= n_current:
            continue
        need = target - n_current
        base = x[idx]
        kk = min(k, n_current - 1)
        if kk < 1:
            synth = base[rng.integers(0, n_current, size=need)]
        else:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=kk + 1).fit(base)
            _, neighbours = nn.kneighbors(base)
            pick = rng.integers(0, n_current, size=need)
            neighbour = neighbours[pick, rng.integers(1, kk + 1, size=need)]
            gap = rng.random((need, x.shape[1]))
            synth = base[pick] + gap * (base[neighbour] - base[pick])
        xs.append(np.clip(synth, 0.0, 1.0))
        ys.append(np.full(need, class_id, dtype=np.int64))
    x_out = np.vstack(xs)
    y_out = np.concatenate(ys)
    order = rng.permutation(len(y_out))
    return x_out[order].astype(np.float32), y_out[order].astype(np.int64)


def _equal_generator_split(quota: int, names: list[str]) -> dict[str, int]:
    if not names or quota <= 0:
        return {name: 0 for name in names}
    per = quota // len(names)
    out = {name: per for name in names}
    out[names[0]] += quota - per * len(names)
    return out


def _sample_from_generators(
    generators,
    class_quotas: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    names = [g.name for g in generators]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for class_id, quota in class_quotas.items():
        if quota <= 0:
            continue
        split = _equal_generator_split(quota, names)
        for generator in generators:
            n = split.get(generator.name, 0)
            if n <= 0:
                continue
            samples = generator.sample(class_id, n)
            xs.append(samples)
            ys.append(np.full(n, class_id, dtype=np.int64))
    if not xs:
        dim = generators[0].feature_dim if generators else 0
        return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.int64), 0
    x_new = np.vstack(xs).astype(np.float32)
    y_new = np.concatenate(ys)
    return x_new, y_new, len(y_new)


def run_no_augmentation(
    config: ExperimentConfig, split: DatasetSplit, seed: int, device: torch.device
) -> ExperimentOutcome:
    _seed_everything(seed)
    start = time.perf_counter()
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(
        split.x_train, split.y_train, split, config, device
    )
    return ExperimentOutcome(
        method="no_augmentation",
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_smote(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    budget: int | None = None,
) -> ExperimentOutcome:
    _seed_everything(seed)
    start = time.perf_counter()
    n_classes = _n_classes(split)
    budget = budget or max(200, len(split.x_train))
    quotas = _deficit_quotas(split.y_train, n_classes, budget)
    counts = np.bincount(split.y_train, minlength=n_classes)
    targets = {k: int(counts[k]) + int(v) for k, v in quotas.items()}
    x_aug, y_aug = smote_oversample(split.x_train, split.y_train, targets, seed=seed)
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(
        x_aug, y_aug, split, config, device
    )
    return ExperimentOutcome(
        method="smote",
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        accepted_samples=int(len(y_aug) - len(split.y_train)),
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_oneshot(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    generator_names: tuple[str, ...] | None = None,
    budget: int | None = None,
    method_name: str | None = None,
) -> ExperimentOutcome:
    """One-shot generation with a fixed budget and no classifier feedback."""
    _seed_everything(seed)
    start = time.perf_counter()
    n_classes = _n_classes(split)
    budget = budget or max(200, len(split.x_train))
    names = generator_names or config.generators.names
    local = replace(config, generators=replace(config.generators, names=tuple(names)))
    generators = _fit_generators(split, local, device, split.x_train, split.y_train)
    quotas = _deficit_quotas(split.y_train, n_classes, budget)
    x_new, y_new, accepted = _sample_from_generators(generators, quotas)
    x_aug = np.vstack([split.x_train, x_new]) if accepted else split.x_train
    y_aug = np.concatenate([split.y_train, y_new]) if accepted else split.y_train
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(
        x_aug, y_aug, split, config, device
    )
    label = method_name or f"oneshot_{'_'.join(names)}"
    return ExperimentOutcome(
        method=label,
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        accepted_samples=accepted,
        iterations=1,
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_static_ensemble(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    budget: int | None = None,
) -> ExperimentOutcome:
    """Equal-weight multi-generator ensemble without feedback or gating."""
    _seed_everything(seed)
    start = time.perf_counter()
    n_classes = _n_classes(split)
    budget = budget or max(200, len(split.x_train))
    generators = _fit_generators(split, config, device, split.x_train, split.y_train)
    quotas = _deficit_quotas(split.y_train, n_classes, budget)
    x_new, y_new, accepted = _sample_from_generators(generators, quotas)
    x_aug = np.vstack([split.x_train, x_new]) if accepted else split.x_train
    y_aug = np.concatenate([split.y_train, y_new]) if accepted else split.y_train
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(
        x_aug, y_aug, split, config, device
    )
    return ExperimentOutcome(
        method="static_ensemble",
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        accepted_samples=accepted,
        iterations=1,
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_gan_smote(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    budget: int | None = None,
    generator_name: str = "wgan_gp",
) -> ExperimentOutcome:
    """GAN generation for rare classes followed by SMOTE balancing."""
    _seed_everything(seed)
    start = time.perf_counter()
    n_classes = _n_classes(split)
    budget = budget or max(200, len(split.x_train))
    local = replace(config, generators=replace(config.generators, names=(generator_name,)))
    generators = _fit_generators(split, local, device, split.x_train, split.y_train)
    quotas = _deficit_quotas(split.y_train, n_classes, budget)
    x_new, y_new, accepted = _sample_from_generators(generators, quotas)
    x_stage = np.vstack([split.x_train, x_new]) if accepted else split.x_train
    y_stage = np.concatenate([split.y_train, y_new]) if accepted else split.y_train
    counts = np.bincount(y_stage, minlength=n_classes)
    targets = {k: int(counts.max()) for k in range(n_classes)}
    x_aug, y_aug = smote_oversample(x_stage, y_stage, targets, seed=seed)
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(
        x_aug, y_aug, split, config, device
    )
    return ExperimentOutcome(
        method="gan_smote",
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        accepted_samples=int(len(y_aug) - len(split.y_train)),
        iterations=1,
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def _entropy_difficulty(model, x_val, y_val, n_classes, device) -> np.ndarray:
    prob = predict_proba(model, x_val, device)
    scores = np.zeros(n_classes, dtype=np.float64)
    for class_id in range(n_classes):
        mask = y_val == class_id
        if not np.any(mask):
            scores[class_id] = 1.0
            continue
        p = prob[mask]
        entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean()
        confidence = 1.0 - entropy / np.log(n_classes)
        scores[class_id] = 1.0 - confidence
    return scores


def _class_losses(model, x_val, y_val, n_classes, device) -> np.ndarray:
    prob = predict_proba(model, x_val, device)
    losses = np.zeros(n_classes, dtype=np.float64)
    for class_id in range(n_classes):
        mask = y_val == class_id
        if not np.any(mask):
            losses[class_id] = 1.0
            continue
        losses[class_id] = float(-np.log(np.clip(prob[mask, class_id], 1e-12, 1.0)).mean())
    return losses


def _iterative_quota_baseline(
    config: ExperimentConfig,
    split: DatasetSplit,
    device: torch.device,
    mode: str,
    iterations: int,
    budget_per_round: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Shared driver for uncertainty-guided and curriculum baselines."""
    n_classes = _n_classes(split)
    input_dim = split.x_train.shape[1]
    generators = _fit_generators(split, config, device, split.x_train, split.y_train)
    x_aug = split.x_train.copy()
    y_aug = split.y_train.copy()
    total_accepted = 0
    used_iterations = 0
    framework = config.framework
    for iteration in range(iterations):
        model = train_classifier(
            x_aug,
            y_aug,
            split.x_val,
            split.y_val,
            input_dim,
            n_classes,
            config.classifier,
            device,
            epochs=config.classifier.feedback_epochs,
            architecture=config.classifier.architecture,
        )
        if mode == "uncertainty":
            weights = _entropy_difficulty(model, split.x_val, split.y_val, n_classes, device)
        else:
            losses = _class_losses(model, split.x_val, split.y_val, n_classes, device)
            exponent = 1.0 + iteration
            weights = np.power(np.clip(losses, 1e-6, None), exponent)
        weights = weights / max(weights.sum(), 1e-12)
        quotas = compute_class_quotas(weights, np.bincount(y_aug, minlength=n_classes), framework)
        scaled_total = int(sum(quotas.values()))
        if scaled_total > 0:
            scale = budget_per_round / scaled_total
            quotas = {k: max(1, int(round(v * scale))) if v > 0 else 0 for k, v in quotas.items()}
        x_new, y_new, accepted = _sample_from_generators(generators, quotas)
        if accepted == 0:
            break
        x_aug = np.vstack([x_aug, x_new])
        y_aug = np.concatenate([y_aug, y_new])
        total_accepted += accepted
        used_iterations = iteration + 1
    return x_aug, y_aug, total_accepted, used_iterations


def run_uncertainty_guided(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    iterations: int | None = None,
    budget_per_round: int | None = None,
) -> ExperimentOutcome:
    _seed_everything(seed)
    start = time.perf_counter()
    iterations = iterations or min(config.framework.max_iterations, 10)
    budget_per_round = budget_per_round or max(64, config.framework.base_quota * len(split.label_to_index))
    x_aug, y_aug, accepted, used = _iterative_quota_baseline(
        config, split, device, "uncertainty", iterations, budget_per_round
    )
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(x_aug, y_aug, split, config, device)
    return ExperimentOutcome(
        method="uncertainty_guided",
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        accepted_samples=accepted,
        iterations=used,
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_curriculum(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    iterations: int | None = None,
    budget_per_round: int | None = None,
) -> ExperimentOutcome:
    _seed_everything(seed)
    start = time.perf_counter()
    iterations = iterations or min(config.framework.max_iterations, 10)
    budget_per_round = budget_per_round or max(64, config.framework.base_quota * len(split.label_to_index))
    x_aug, y_aug, accepted, used = _iterative_quota_baseline(
        config, split, device, "curriculum", iterations, budget_per_round
    )
    metrics, per_class, y_true, y_pred, y_prob = _train_eval(x_aug, y_aug, split, config, device)
    return ExperimentOutcome(
        method="curriculum",
        scenario=config.scenario,
        final_metrics=metrics,
        per_class_f1=per_class,
        accepted_samples=accepted,
        iterations=used,
        elapsed_sec=time.perf_counter() - start,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


BASELINE_METHODS = (
    "no_augmentation",
    "smote",
    "oneshot_wgan_gp",
    "oneshot_cvae",
    "oneshot_ddpm",
    "static_ensemble",
    "gan_smote",
    "uncertainty_guided",
    "curriculum",
)


def run_baseline(
    method: str,
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    device: torch.device,
    budget: int | None = None,
) -> ExperimentOutcome:
    if method == "no_augmentation":
        return run_no_augmentation(config, split, seed, device)
    if method == "smote":
        return run_smote(config, split, seed, device, budget=budget)
    if method.startswith("oneshot_"):
        name = method[len("oneshot_"):]
        return run_oneshot(
            config, split, seed, device, generator_names=(name,), budget=budget
        )
    if method == "static_ensemble":
        return run_static_ensemble(config, split, seed, device, budget=budget)
    if method == "gan_smote":
        return run_gan_smote(config, split, seed, device, budget=budget)
    if method == "uncertainty_guided":
        return run_uncertainty_guided(config, split, seed, device, budget_per_round=budget)
    if method == "curriculum":
        return run_curriculum(config, split, seed, device, budget_per_round=budget)
    raise ValueError(f"Unknown baseline method: {method}")

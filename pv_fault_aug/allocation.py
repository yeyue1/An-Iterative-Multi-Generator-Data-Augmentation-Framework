from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FrameworkConfig
from .feedback import FeedbackSignals


@dataclass
class AllocationPlan:
    difficulty_weights: np.ndarray
    class_quotas: dict[int, int]
    generator_class_quotas: dict[str, dict[int, int]]
    generator_ratios: dict[str, np.ndarray]


def compute_difficulty_weights(feedback: FeedbackSignals, config: FrameworkConfig) -> np.ndarray:
    if not config.use_difficulty_allocation:
        return np.ones_like(feedback.scarcity, dtype=np.float64)

    if not config.use_feedback_allocation:
        return np.ones_like(feedback.scarcity, dtype=np.float64)

    weights = (
        config.alpha_error * feedback.error_rate
        + config.alpha_uncertainty * (1.0 - feedback.confidence)
        + config.alpha_scarcity * feedback.scarcity
    )
    return np.maximum(weights, 1e-8)


def compute_class_quotas(
    difficulty_weights: np.ndarray,
    class_counts: np.ndarray,
    config: FrameworkConfig,
) -> dict[int, int]:
    avg = float(np.mean(difficulty_weights))
    quotas: dict[int, int] = {}
    for class_id, weight in enumerate(difficulty_weights):
        raw = config.base_quota * float(weight / max(avg, 1e-8))
        quota = int(round(np.clip(raw, config.min_quota, config.max_quota)))
        per_round_cap = int(max(config.min_quota, class_counts[class_id] * config.max_per_round_generation_ratio))
        quotas[class_id] = int(min(quota, per_round_cap))
    return quotas


def _integer_split(total: int, ratios: np.ndarray) -> np.ndarray:
    if total <= 0:
        return np.zeros_like(ratios, dtype=int)
    ratios = np.maximum(ratios, 0)
    ratios = ratios / max(ratios.sum(), 1e-12)
    values = np.floor(ratios * total).astype(int)
    remainder = total - int(values.sum())
    if remainder > 0:
        order = np.argsort(ratios * total - values)[::-1]
        values[order[:remainder]] += 1
    return values


def build_allocation_plan(
    feedback: FeedbackSignals,
    generator_names: list[str],
    global_weights: np.ndarray,
    contribution_scores: dict[str, np.ndarray],
    config: FrameworkConfig,
) -> AllocationPlan:
    difficulty = compute_difficulty_weights(feedback, config)
    class_quotas = compute_class_quotas(difficulty, feedback.class_counts, config)

    global_weights = np.asarray(global_weights, dtype=np.float64)
    global_weights = global_weights / max(global_weights.sum(), 1e-12)
    n_classes = len(difficulty)

    generator_ratios = {
        name: np.zeros(n_classes, dtype=np.float64) for name in generator_names
    }
    generator_class_quotas = {
        name: {class_id: 0 for class_id in range(n_classes)} for name in generator_names
    }

    for class_id in range(n_classes):
        local_scores = np.asarray(
            [
                contribution_scores.get(name, np.ones(n_classes))[class_id]
                for name in generator_names
            ],
            dtype=np.float64,
        )
        local = np.maximum(global_weights * np.maximum(local_scores, 1e-6), 1e-8)
        local = local / local.sum()
        split = _integer_split(class_quotas[class_id], local)
        for gen_idx, name in enumerate(generator_names):
            generator_ratios[name][class_id] = local[gen_idx]
            generator_class_quotas[name][class_id] = int(split[gen_idx])

    return AllocationPlan(
        difficulty_weights=difficulty,
        class_quotas=class_quotas,
        generator_class_quotas=generator_class_quotas,
        generator_ratios=generator_ratios,
    )

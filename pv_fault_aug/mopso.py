from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import MopsoConfig


ObjectiveFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MopsoResult:
    weights: np.ndarray
    objectives: np.ndarray
    archive_weights: np.ndarray
    archive_objectives: np.ndarray


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-8)
    return x / x.sum()


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _nondominated(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(len(points)):
            if i != j and _dominates(points[j], points[i]):
                keep[i] = False
                break
    return keep


def _scalar_score(objectives: np.ndarray) -> np.ndarray:
    """Equal-weight normalized Tchebycheff scalarization over the archive.

    Each objective is divided by its ideal-nadir range of the current archive
    and the largest normalized deviation is minimised, following Eq.
    (mopso_select) of the manuscript.  The constant lambda_j = 1/3 does not
    change the argmin and is therefore omitted.
    """
    ideal = objectives.min(axis=0)
    nadir = objectives.max(axis=0)
    scaled = np.abs(objectives - ideal) / np.maximum(nadir - ideal, 1e-12)
    return scaled.max(axis=1)


class MOPSO:
    """A compact MOPSO implementation for generator-weight optimization."""

    def __init__(self, config: MopsoConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def optimize(self, n_weights: int, objective_fn: ObjectiveFn) -> MopsoResult:
        positions = self.rng.dirichlet(np.ones(n_weights), size=self.config.swarm_size)
        velocities = self.rng.normal(0.0, 0.05, size=positions.shape)
        objectives = np.vstack([objective_fn(p) for p in positions])

        pbest = positions.copy()
        pbest_obj = objectives.copy()
        non_dominated = _nondominated(objectives)
        archive_pos = positions[non_dominated]
        archive_obj = objectives[non_dominated]

        for _ in range(self.config.iterations):
            leader_idx = int(np.argmin(_scalar_score(archive_obj)))
            leader = archive_pos[leader_idx]
            r1 = self.rng.random(positions.shape)
            r2 = self.rng.random(positions.shape)
            velocities = (
                self.config.inertia * velocities
                + self.config.cognitive * r1 * (pbest - positions)
                + self.config.social * r2 * (leader - positions)
            )
            positions = np.vstack([_normalize_simplex(p) for p in positions + velocities])
            objectives = np.vstack([objective_fn(p) for p in positions])

            for i in range(len(positions)):
                if _dominates(objectives[i], pbest_obj[i]) or (
                    not _dominates(pbest_obj[i], objectives[i])
                    and _scalar_score(objectives[[i]])[0]
                    < _scalar_score(pbest_obj[[i]])[0]
                ):
                    pbest[i] = positions[i]
                    pbest_obj[i] = objectives[i]

            merged_pos = np.vstack([archive_pos, positions])
            merged_obj = np.vstack([archive_obj, objectives])
            keep = _nondominated(merged_obj)
            archive_pos = merged_pos[keep]
            archive_obj = merged_obj[keep]
            if len(archive_pos) > self.config.swarm_size:
                order = np.argsort(_scalar_score(archive_obj))[: self.config.swarm_size]
                archive_pos = archive_pos[order]
                archive_obj = archive_obj[order]

        best_idx = int(np.argmin(_scalar_score(archive_obj)))
        return MopsoResult(
            weights=archive_pos[best_idx],
            objectives=archive_obj[best_idx],
            archive_weights=archive_pos,
            archive_objectives=archive_obj,
        )


def optimize_generator_weights(
    generator_names: list[str],
    mmd_scores: dict[str, float],
    validation_f1_scores: dict[str, float],
    config: MopsoConfig,
    seed: int,
) -> MopsoResult:
    """Optimize the global generator weights.

    The objective set follows Eqs. (mopso_mmd_agg)--(mopso_entropy).  The
    entropy objective f_3 can be disabled for the corresponding sensitivity
    row, in which case the Tchebycheff selection operates on two objectives.
    """
    names = list(generator_names)

    def objective(weights: np.ndarray) -> np.ndarray:
        mmd = sum(weights[i] * mmd_scores.get(names[i], 1.0) for i in range(len(names)))
        validation_f1 = sum(
            weights[i] * validation_f1_scores.get(names[i], 0.0) for i in range(len(names))
        )
        values = [mmd, -validation_f1]
        if config.use_entropy_objective:
            entropy_objective = float(np.sum(weights * np.log(weights + 1e-12)))
            values.append(entropy_objective)
        return np.asarray(values, dtype=np.float64)

    return MOPSO(config, seed=seed).optimize(len(names), objective)
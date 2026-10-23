from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch

from .allocation import build_allocation_plan, compute_class_quotas, compute_difficulty_weights
from .config import ExperimentConfig
from .data import DatasetSplit
from .feedback import compute_feedback
from .metrics import classification_metrics, feature_center, per_class_f1
from .models.classifier import predict_proba, train_classifier
from .models.generators import build_generators
from .mopso import optimize_generator_weights
from .screening import dual_gate_screen, merge_candidates, rbf_mmd


@dataclass
class IterationRecord:
    iteration: int
    f1_macro: float
    f1_gain: float
    accepted_samples: int
    feature_center_drift: float
    class_quotas: dict[int, int]
    difficulty_weights: list[float]
    global_weights: dict[str, float]
    accepted_counts_by_source: dict[str, int]


@dataclass
class ClosedLoopResult:
    final_metrics: dict[str, float]
    per_class_f1: dict[str, float]
    history: list[IterationRecord]
    test_prob: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    accepted_x: np.ndarray | None = None
    accepted_y: np.ndarray | None = None
    model: object | None = None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ClosedLoopAugmentor:
    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config
        self.device = resolve_device(config.device)

    def _probing_scores(
        self,
        generators,
        probing_quotas: dict[int, int],
        x_aug: np.ndarray,
        y_aug: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        n_classes: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        mmd_scores: dict[str, float] = {}
        f1_scores: dict[str, float] = {}

        for generator in generators:
            probing = generator.sample_many(probing_quotas)
            if len(probing.labels) == 0:
                mmd_scores[generator.name] = float("inf")
                f1_scores[generator.name] = 0.0
                continue

            class_mmd: list[float] = []
            for class_id in range(n_classes):
                real_mask = y_aug == class_id
                probe_mask = probing.labels == class_id
                if np.any(real_mask) and np.any(probe_mask):
                    class_mmd.append(rbf_mmd(x_aug[real_mask], probing.samples[probe_mask]))
            mmd_scores[generator.name] = float(np.mean(class_mmd)) if class_mmd else float("inf")

            x_probe = np.vstack([x_aug, probing.samples])
            y_probe = np.concatenate([y_aug, probing.labels])
            probe_classifier = train_classifier(
                x_probe,
                y_probe,
                x_val,
                y_val,
                input_dim,
                n_classes,
                self.config.classifier,
                self.device,
                epochs=self.config.classifier.feedback_epochs,
                architecture=self.config.classifier.architecture,
            )
            probe_prob = predict_proba(probe_classifier, x_val, self.device)
            probe_pred = probe_prob.argmax(axis=1)
            f1_scores[generator.name] = classification_metrics(
                y_val, probe_pred, probe_prob, n_classes
            )["f1_macro"]

        return mmd_scores, f1_scores

    def run(self, split: DatasetSplit, seed: int) -> ClosedLoopResult:
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_classes = len(split.label_to_index)
        input_dim = split.x_train.shape[1]
        x_aug = split.x_train.copy()
        y_aug = split.y_train.copy()

        generators = (
            []
            if self.config.framework.max_iterations == 0
            else build_generators(input_dim, n_classes, self.config.generators, self.device)
        )
        generator_names = [g.name for g in generators]
        contribution_scores = {
            name: np.ones(n_classes, dtype=np.float64) for name in generator_names
        }
        global_weights = (
            np.ones(len(generator_names), dtype=np.float64) / len(generator_names)
            if generator_names
            else np.empty(0, dtype=np.float64)
        )
        history: list[IterationRecord] = []
        accepted_x_parts: list[np.ndarray] = []
        accepted_y_parts: list[np.ndarray] = []
        previous_center = feature_center(x_aug)
        saturation_count = 0

        for iteration in range(self.config.framework.max_iterations):
            gen_epochs = (
                self.config.generators.quick_epochs
                if iteration > 0
                else min(self.config.generators.epochs, self.config.generators.quick_epochs)
            )
            for generator in generators:
                generator.fit(x_aug, y_aug, epochs=gen_epochs)

            classifier = train_classifier(
                x_aug,
                y_aug,
                split.x_val,
                split.y_val,
                input_dim,
                n_classes,
                self.config.classifier,
                self.device,
                architecture=self.config.classifier.architecture,
            )
            val_prob = predict_proba(classifier, split.x_val, self.device)
            val_pred = val_prob.argmax(axis=1)
            val_metrics = classification_metrics(split.y_val, val_pred, val_prob, n_classes)
            feedback = compute_feedback(
                classifier,
                split.x_val,
                split.y_val,
                x_aug,
                y_aug,
                n_classes,
                self.device,
            )

            difficulty_weights = compute_difficulty_weights(feedback, self.config.framework)
            probing_quotas = compute_class_quotas(
                difficulty_weights, feedback.class_counts, self.config.framework
            )
            mmd_scores, f1_scores = self._probing_scores(
                generators,
                probing_quotas,
                x_aug,
                y_aug,
                split.x_val,
                split.y_val,
                input_dim,
                n_classes,
            )
            mopso = optimize_generator_weights(
                generator_names,
                mmd_scores,
                f1_scores,
                self.config.mopso,
                seed=seed + iteration,
            )
            global_weights = mopso.weights

            allocation = build_allocation_plan(
                feedback,
                generator_names,
                global_weights,
                contribution_scores,
                self.config.framework,
            )

            generated = [
                generator.sample_many(allocation.generator_class_quotas[generator.name])
                for generator in generators
            ]
            candidates = merge_candidates(generated, input_dim)
            screened = dual_gate_screen(
                x_aug,
                y_aug,
                split.x_val,
                split.y_val,
                classifier,
                candidates,
                input_dim,
                n_classes,
                self.config.classifier,
                self.config.framework,
                self.device,
            )

            if len(screened.accepted_y) > 0:
                x_aug = np.vstack([x_aug, screened.accepted_x])
                y_aug = np.concatenate([y_aug, screened.accepted_y])
                accepted_x_parts.append(screened.accepted_x)
                accepted_y_parts.append(screened.accepted_y)

            if len(screened.accepted_y) > 0:
                round_classifier = train_classifier(
                    x_aug,
                    y_aug,
                    split.x_val,
                    split.y_val,
                    input_dim,
                    n_classes,
                    self.config.classifier,
                    self.device,
                    epochs=self.config.classifier.feedback_epochs,
                    architecture=self.config.classifier.architecture,
                )
                round_prob = predict_proba(round_classifier, split.x_val, self.device)
                round_pred = round_prob.argmax(axis=1)
                round_metrics = classification_metrics(
                    split.y_val, round_pred, round_prob, n_classes
                )
                round_f1_gain = (round_metrics["f1_macro"] - val_metrics["f1_macro"]) / 100.0
            else:
                round_metrics = val_metrics
                round_f1_gain = 0.0

            current_center = feature_center(x_aug)
            drift = float(np.linalg.norm(current_center - previous_center))
            previous_center = current_center

            generated_counts = {
                name: np.bincount(result.labels, minlength=n_classes).astype(np.float64)
                for name, result in zip(generator_names, generated)
            }
            accepted_counts = {
                name: np.bincount(
                    screened.accepted_y[screened.accepted_source == name],
                    minlength=n_classes,
                ).astype(np.float64)
                for name in generator_names
            }
            for name in generator_names:
                contribution_scores[name] = (accepted_counts[name] + 1.0) / (
                    generated_counts[name] + 2.0
                )

            if round_f1_gain < self.config.framework.min_f1_gain:
                saturation_count += 1
            else:
                saturation_count = 0

            history.append(
                IterationRecord(
                    iteration=iteration,
                    f1_macro=round_metrics["f1_macro"],
                    f1_gain=round_f1_gain * 100.0,
                    accepted_samples=int(len(screened.accepted_y)),
                    feature_center_drift=drift,
                    class_quotas=allocation.class_quotas,
                    difficulty_weights=allocation.difficulty_weights.tolist(),
                    global_weights={
                        name: float(global_weights[i]) for i, name in enumerate(generator_names)
                    },
                    accepted_counts_by_source=screened.accepted_counts_by_source,
                )
            )

            if self.config.framework.use_convergence_control:
                f1_saturated = saturation_count >= self.config.framework.f1_saturation_window
                center_converged = iteration > 0 and drift <= self.config.framework.center_drift_threshold
                if f1_saturated and center_converged:
                    break

        final_classifier = train_classifier(
            x_aug,
            y_aug,
            split.x_val,
            split.y_val,
            input_dim,
            n_classes,
            self.config.classifier,
            self.device,
            architecture=self.config.classifier.architecture,
        )
        test_prob = predict_proba(final_classifier, split.x_test, self.device)
        test_pred = test_prob.argmax(axis=1)
        final_metrics = classification_metrics(split.y_test, test_pred, test_prob, n_classes)
        class_f1 = per_class_f1(split.y_test, test_pred, n_classes)
        per_class = {
            split.index_to_label[idx]: float(class_f1[idx]) for idx in range(n_classes)
        }

        accepted_x = np.vstack(accepted_x_parts) if accepted_x_parts else np.empty((0, input_dim), dtype=np.float32)
        accepted_y = np.concatenate(accepted_y_parts) if accepted_y_parts else np.empty(0, dtype=np.int64)
        return ClosedLoopResult(
            final_metrics=final_metrics,
            per_class_f1=per_class,
            history=history,
            test_prob=test_prob,
            test_pred=test_pred,
            accepted_x=accepted_x,
            accepted_y=accepted_y,
            model=final_classifier,
        )

    def save_result(self, result: ClosedLoopResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "final_metrics": result.final_metrics,
            "per_class_f1": result.per_class_f1,
            "history": [asdict(row) for row in result.history],
        }
        with (output_dir / "closed_loop_result.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=_json_ready, indent=2)
        arrays = {}
        if result.test_prob is not None:
            arrays["test_prob"] = result.test_prob
        if result.test_pred is not None:
            arrays["test_pred"] = result.test_pred
        if result.accepted_x is not None:
            arrays["accepted_x"] = result.accepted_x
            arrays["accepted_y"] = result.accepted_y
        if arrays:
            np.savez_compressed(output_dir / "run_arrays.npz", **arrays)

        history_path = output_dir / "iteration_history.csv"
        with history_path.open("w", encoding="utf-8") as fh:
            fh.write("iteration,f1_macro,f1_gain,accepted_samples,feature_center_drift\n")
            for row in result.history:
                fh.write(
                    f"{row.iteration},{row.f1_macro:.6f},{row.f1_gain:.6f},"
                    f"{row.accepted_samples},{row.feature_center_drift:.8f}\n"
                )

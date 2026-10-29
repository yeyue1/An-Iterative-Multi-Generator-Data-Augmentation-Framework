from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from pv_fault_aug.config import (
    ExperimentConfig,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_GENERATORS,
)
from pv_fault_aug.data import load_scenario_split
from pv_fault_aug.metrics import classification_metrics
from pv_fault_aug.models.classifier import predict_proba, train_classifier
from pv_fault_aug.pipeline import ClosedLoopAugmentor, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifier-generality analysis of Table 10 (scarce scenario by default)."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", default="scarce", choices=["balanced", "imbalanced", "scarce"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--classifiers",
        default=",".join(SUPPORTED_CLASSIFIERS),
        help="Comma-separated classifier architectures.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classifier_generality"))
    parser.add_argument(
        "--partition-seed", type=int, default=None, help="Shared partition seed."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generators", default=",".join(SUPPORTED_GENERATORS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-prefix", default=None)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--scarce-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_config(args: argparse.Namespace, architecture: str) -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.data_path = args.data_path
    config.data.label_column = args.label_column
    config.data.feature_prefix = args.feature_prefix
    config.data.feature_dim = args.feature_dim
    config.data.scarce_ratio = args.scarce_ratio
    config.data.n_runs = args.runs
    config.scenario = args.scenario
    config.device = args.device
    config.classifier.architecture = architecture
    config.generators.names = tuple(
        name.strip() for name in args.generators.split(",") if name.strip()
    )
    if args.partition_seed is not None:
        config.data.random_seed = args.partition_seed
    if args.quick:
        config.classifier.epochs = 20
        config.classifier.feedback_epochs = 5
        config.classifier.patience = 5
        config.generators.epochs = 10
        config.generators.quick_epochs = 5
        config.framework.max_iterations = 3
        config.framework.gate_eval_runs = 1
        config.mopso.iterations = 10
        config.mopso.swarm_size = 12
    config.validate()
    return config


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.nanmean(array)), std


def main() -> None:
    args = parse_args()
    architectures = [name.strip() for name in args.classifiers.split(",") if name.strip()]
    unknown = sorted(set(architectures) - set(SUPPORTED_CLASSIFIERS))
    if unknown:
        raise ValueError(f"Unknown classifiers: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = build_config(args, architectures[0])
    device = resolve_device(base.device)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(base), fh, default=str, indent=2)

    split = load_scenario_split(base.data, args.scenario, base.data.random_seed)
    n_classes = len(split.label_to_index)
    rows = []
    for architecture in architectures:
        config = build_config(args, architecture)
        no_aug_f1: list[float] = []
        proposed_f1: list[float] = []
        for run_idx in range(args.runs):
            seed = base.data.random_seed + run_idx * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            baseline = train_classifier(
                split.x_train,
                split.y_train,
                split.x_val,
                split.y_val,
                split.x_train.shape[1],
                n_classes,
                config.classifier,
                device,
                architecture=architecture,
            )
            prob = predict_proba(baseline, split.x_test, device)
            metrics = classification_metrics(
                split.y_test, prob.argmax(axis=1), prob, n_classes
            )
            no_aug_f1.append(float(metrics["f1_macro"]))

            result = ClosedLoopAugmentor(config).run(split, seed)
            proposed_f1.append(float(result.final_metrics["f1_macro"]))
        before_mean, before_std = _mean_std(no_aug_f1)
        after_mean, after_std = _mean_std(proposed_f1)
        rows.append(
            {
                "classifier": architecture,
                "no_augmentation_mean": before_mean,
                "no_augmentation_std": before_std,
                "proposed_mean": after_mean,
                "proposed_std": after_std,
                "gain": after_mean - before_mean,
            }
        )
        print(
            f"[{architecture}] no-aug={before_mean:.2f} +/- {before_std:.2f} "
            f"proposed={after_mean:.2f} +/- {after_std:.2f}"
        )

    with (args.output_dir / "classifier_generality.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "classifier,no_augmentation_mean,no_augmentation_std,"
            "proposed_mean,proposed_std,gain\n"
        )
        for row in rows:
            fh.write(
                f"{row['classifier']},{row['no_augmentation_mean']:.3f},"
                f"{row['no_augmentation_std']:.3f},{row['proposed_mean']:.3f},"
                f"{row['proposed_std']:.3f},{row['gain']:.3f}\n"
            )
    print(f"results written to {args.output_dir}")


if __name__ == "__main__":
    main()
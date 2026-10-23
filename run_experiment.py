from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json

from pv_fault_aug.config import (
    ExperimentConfig,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_GENERATORS,
    SUPPORTED_VARIANTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop multi-generator augmentation for PV fault diagnosis."
    )
    parser.add_argument("--data-path", type=Path, required=True, help="CSV or NPZ dataset path.")
    parser.add_argument("--label-column", default="label", help="Label column name for CSV input.")
    parser.add_argument(
        "--feature-prefix",
        default=None,
        help="Optional CSV feature prefix. If omitted, all non-label columns are used.",
    )
    parser.add_argument("--feature-dim", type=int, default=100, help="Number of I-V sampling points.")
    parser.add_argument("--scarce-ratio", type=float, default=0.10, help="Scarce scenario sampling ratio.")
    parser.add_argument(
        "--scenario",
        choices=["balanced", "imbalanced", "scarce"],
        default="scarce",
        help="Real-sample scenario before augmentation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--classifier",
        default="cnn1d",
        choices=SUPPORTED_CLASSIFIERS,
        help="Downstream classifier architecture.",
    )
    parser.add_argument(
        "--metadata-columns",
        default="",
        help="Comma-separated metadata columns to carry through the split "
        "(e.g. irradiance,temperature,module_id,condition_id).",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of stratified random runs.")
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="full",
        help="Experiment variant for baselines and ablations.",
    )
    parser.add_argument(
        "--generators",
        default=",".join(SUPPORTED_GENERATORS),
        help="Comma-separated generator subset, e.g. wgan_gp,cvae,ddpm.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter training loops for smoke testing before full experiments.",
    )
    parser.add_argument(
        "--make-demo-data",
        action="store_true",
        help="Create a small synthetic NPZ dataset at --data-path and exit.",
    )
    return parser.parse_args()


def make_demo_data(path: Path, feature_dim: int, seed: int = 42) -> None:
    import numpy as np

    from pv_fault_aug.config import FAULT_CLASSES

    rng = np.random.default_rng(seed)
    counts = [180, 120, 110, 100, 100, 100, 100]
    grid = np.linspace(0.0, 1.0, feature_dim, dtype=np.float32)
    samples: list[np.ndarray] = []
    labels: list[str] = []
    for class_id, count in enumerate(counts):
        slope = 1.0 - 0.06 * class_id
        knee = 0.18 * np.tanh((grid - 0.55 + 0.03 * class_id) * 8.0)
        ripple = 0.03 * np.sin((class_id + 1) * np.pi * grid)
        base = np.clip(slope * (1.0 - grid) + knee + ripple, 0.0, 1.0)
        for _ in range(count):
            noise = rng.normal(0.0, 0.015, size=feature_dim).astype(np.float32)
            shift = rng.normal(0.0, 0.02)
            samples.append(np.clip(base + shift + noise, 0.0, 1.0).astype(np.float32))
            labels.append(FAULT_CLASSES[class_id])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=np.vstack(samples).astype(np.float32), y=np.asarray(labels, dtype=object))
    print(f"demo dataset written to {path}")


def main() -> None:
    args = parse_args()

    import numpy as np

    if args.make_demo_data:
        make_demo_data(args.data_path, args.feature_dim)
        return

    from pv_fault_aug.data import load_scenario_split
    from pv_fault_aug.pipeline import ClosedLoopAugmentor

    config = ExperimentConfig()
    config.data.data_path = args.data_path
    config.data.label_column = args.label_column
    config.data.feature_prefix = args.feature_prefix
    config.data.feature_dim = args.feature_dim
    config.data.scarce_ratio = args.scarce_ratio
    config.scenario = args.scenario
    config.output_dir = args.output_dir
    config.device = args.device
    config.data.n_runs = args.runs
    config.data.metadata_columns = tuple(
        name.strip() for name in args.metadata_columns.split(",") if name.strip()
    )
    config.classifier.architecture = args.classifier
    config.variant = args.variant
    config.generators.names = tuple(
        name.strip() for name in args.generators.split(",") if name.strip()
    )

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

    config.apply_variant()
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, default=str, indent=2)

    all_metrics: list[dict[str, float]] = []
    for run_idx in range(config.data.n_runs):
        seed = config.data.random_seed + run_idx * 100
        split = load_scenario_split(config.data, config.scenario, seed)
        runner = ClosedLoopAugmentor(config)
        result = runner.run(split, seed)
        run_dir = config.output_dir / f"{config.scenario}_run_{run_idx + 1}"
        runner.save_result(result, run_dir)
        all_metrics.append(result.final_metrics)
        print(f"[run {run_idx + 1}] {result.final_metrics}")

    summary = {}
    for key in all_metrics[0]:
        values = np.asarray([m[key] for m in all_metrics], dtype=np.float64)
        summary[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    with (config.output_dir / f"{config.scenario}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("summary:", summary)


if __name__ == "__main__":
    main()

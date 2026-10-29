from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from pv_fault_aug.config import (
    ExperimentConfig,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_GENERATORS,
)
from pv_fault_aug.data import load_scenario_split
from pv_fault_aug.pipeline import ClosedLoopAugmentor, resolve_device


SENSITIVITY_ROWS = (
    ("default", {"alpha": (0.4, 0.3, 0.3)}),
    ("uniform_difficulty", {"alpha": (1 / 3, 1 / 3, 1 / 3)}),
    ("error_only", {"alpha": (1.0, 0.0, 0.0)}),
    ("scarcity_only", {"alpha": (0.0, 0.0, 1.0)}),
    ("mopso_without_entropy", {"use_entropy_objective": False}),
    ("either_condition_stopping", {"stop_rule": "either"}),
    ("joint_tau_stop_0.005", {"center_drift_threshold": 0.005}),
    ("joint_tau_stop_0.020", {"center_drift_threshold": 0.020}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis of Table 11 (scarce scenario by default)."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", default="scarce", choices=["balanced", "imbalanced", "scarce"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument(
        "--partition-seed", type=int, default=None, help="Shared partition seed."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier", default="cnn1d", choices=SUPPORTED_CLASSIFIERS)
    parser.add_argument("--generators", default=",".join(SUPPORTED_GENERATORS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-prefix", default=None)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--scarce-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.data_path = args.data_path
    config.data.label_column = args.label_column
    config.data.feature_prefix = args.feature_prefix
    config.data.feature_dim = args.feature_dim
    config.data.scarce_ratio = args.scarce_ratio
    config.data.n_runs = args.runs
    config.scenario = args.scenario
    config.device = args.device
    config.classifier.architecture = args.classifier
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


def variant_config(
    base: ExperimentConfig,
    overrides: dict,
) -> ExperimentConfig:
    framework = replace(base.framework)
    mopso = replace(base.mopso)
    if "alpha" in overrides:
        framework.alpha_error, framework.alpha_uncertainty, framework.alpha_scarcity = (
            overrides["alpha"]
        )
    if "stop_rule" in overrides:
        framework.stop_rule = overrides["stop_rule"]
    if "stop_f1_saturation" in overrides:
        framework.stop_f1_saturation = overrides["stop_f1_saturation"]
    if "center_drift_threshold" in overrides:
        framework.center_drift_threshold = overrides["center_drift_threshold"]
    if "use_entropy_objective" in overrides:
        mopso.use_entropy_objective = overrides["use_entropy_objective"]
    return replace(base, framework=framework, mopso=mopso)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.nanmean(array)), std


def main() -> None:
    args = parse_args()
    base = build_config(args)
    device = resolve_device(base.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = load_scenario_split(base.data, args.scenario, base.data.random_seed)

    rows = []
    for name, overrides in SENSITIVITY_ROWS:
        config = variant_config(base, overrides)
        config.validate()
        f1_values: list[float] = []
        iteration_values: list[float] = []
        max_weight_values: list[float] = []
        for run_idx in range(args.runs):
            seed = base.data.random_seed + run_idx * 100
            result = ClosedLoopAugmentor(config).run(split, seed)
            f1_values.append(float(result.final_metrics["f1_macro"]))
            iteration_values.append(float(len(result.history)))
            if result.history:
                max_weight_values.append(
                    max(
                        max(row.global_weights.values())
                        for row in result.history
                        if row.global_weights
                    )
                )
        f1_mean, f1_std = _mean_std(f1_values)
        iter_mean, iter_std = _mean_std(iteration_values)
        weight_mean = float(np.mean(max_weight_values)) if max_weight_values else float("nan")
        rows.append(
            {
                "configuration": name,
                "f1_macro_mean": f1_mean,
                "f1_macro_std": f1_std,
                "iterations_mean": iter_mean,
                "iterations_std": iter_std,
                "max_generator_weight": weight_mean,
            }
        )
        print(
            f"[{name}] F1={f1_mean:.2f} +/- {f1_std:.2f} "
            f"iterations={iter_mean:.1f} +/- {iter_std:.1f}"
        )

    with (args.output_dir / "sensitivity.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "configuration,f1_macro_mean,f1_macro_std,iterations_mean,"
            "iterations_std,max_generator_weight\n"
        )
        for row in rows:
            fh.write(
                f"{row['configuration']},{row['f1_macro_mean']:.3f},"
                f"{row['f1_macro_std']:.3f},{row['iterations_mean']:.2f},"
                f"{row['iterations_std']:.2f},{row['max_generator_weight']:.3f}\n"
            )
    with (args.output_dir / "sensitivity.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(base), "rows": rows}, fh, indent=2, default=str)
    print(f"results written to {args.output_dir}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from pv_fault_aug.analysis import (
    IRRADIANCE_BINS,
    TEMPERATURE_BINS,
    PERTURBATION_SETTINGS,
    apply_perturbation,
    confusion_counts,
    evaluate_perturbations,
    make_group_heldout_split,
    normalize_confusion_counts,
    regime_report,
    summarize_physical,
)
from pv_fault_aug.config import (
    ExperimentConfig,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_GENERATORS,
    SUPPORTED_VARIANTS,
)
from pv_fault_aug.data import load_fault_dataset, load_scenario_split
from pv_fault_aug.figures import (
    plot_confusion_matrices,
    plot_quota_distribution,
    plot_real_vs_generated,
    plot_run_level_f1,
)
from pv_fault_aug.metrics import classification_metrics
from pv_fault_aug.models.classifier import predict_proba, train_classifier
from pv_fault_aug.pipeline import ClosedLoopAugmentor, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run error analysis, robustness tests, and physical-consistency checks."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", default="scarce", choices=["balanced", "imbalanced", "scarce"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier", default="cnn1d", choices=SUPPORTED_CLASSIFIERS)
    parser.add_argument("--generators", default=",".join(SUPPORTED_GENERATORS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--variant", default="full", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-prefix", default=None)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--scarce-ratio", type=float, default=0.10)
    parser.add_argument(
        "--metadata-columns",
        default="irradiance,temperature,module_id,condition_id",
        help="Comma-separated metadata columns expected in the dataset (optional).",
    )
    parser.add_argument("--irradiance-column", default="irradiance")
    parser.add_argument("--temperature-column", default="temperature")
    parser.add_argument("--module-column", default="module_id")
    parser.add_argument("--condition-column", default="condition_id")
    parser.add_argument("--voltage-max", type=float, default=None)
    parser.add_argument("--current-max", type=float, default=None)
    parser.add_argument(
        "--no-group-heldout",
        action="store_true",
        help="Skip the unseen-module/condition group-held-out experiment.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.data_path = args.data_path
    config.data.label_column = args.label_column
    config.data.feature_prefix = args.feature_prefix
    config.data.feature_dim = args.feature_dim
    config.data.scarce_ratio = args.scarce_ratio
    config.data.n_runs = args.runs
    config.data.metadata_columns = tuple(
        name.strip() for name in args.metadata_columns.split(",") if name.strip()
    )
    config.data.irradiance_column = args.irradiance_column
    config.data.temperature_column = args.temperature_column
    config.data.module_column = args.module_column
    config.data.condition_column = args.condition_column
    config.scenario = args.scenario
    config.device = args.device
    config.classifier.architecture = args.classifier
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
    config.variant = args.variant
    config.apply_variant()
    config.validate()
    return config


def train_no_augmentation(config, split, device):
    return train_classifier(
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        split.x_train.shape[1],
        len(split.label_to_index),
        config.classifier,
        device,
        architecture=config.classifier.architecture,
    )


def collect_predictions(model, split, device):
    prob = predict_proba(model, split.x_test, device)
    pred = prob.argmax(axis=1)
    n_classes = len(split.label_to_index)
    return pred, prob, classification_metrics(split.y_test, pred, prob, n_classes)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    device = resolve_device(config.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, default=str, indent=2)

    n_classes = len(config.data.class_names)
    class_names = list(config.data.class_names)
    accum = {
        "confusion_before": np.zeros((n_classes, n_classes)),
        "confusion_after": np.zeros((n_classes, n_classes)),
        "perturbations_before": {},
        "perturbations_after": {},
        "physical": [],
    }
    f1_runs = []
    stop_runs = []
    class_totals = []
    iteration_totals = []
    first_run_artifacts = None

    for run_idx in range(args.runs):
        seed = config.data.random_seed + run_idx * 100
        split = load_scenario_split(config.data, args.scenario, seed)
        runner = ClosedLoopAugmentor(config)
        result = runner.run(split, seed)
        runner.save_result(result, args.output_dir / f"run_{run_idx + 1}")
        baseline_model = train_no_augmentation(config, split, device)

        pred_before, prob_before, metrics_before = collect_predictions(
            baseline_model, split, device
        )
        pred_after, prob_after, metrics_after = collect_predictions(
            result.model, split, device
        )
        accum["confusion_before"] += confusion_counts(split.y_test, pred_before, n_classes)
        accum["confusion_after"] += confusion_counts(split.y_test, pred_after, n_classes)

        for key, pred, prob in (("before", pred_before, prob_before), ("after", pred_after, prob_after)):
            rows = evaluate_perturbations(
                lambda x, m=(result.model if key == "after" else baseline_model): predict_proba(
                    m, x, device
                ),
                split.x_test,
                split.y_test,
                n_classes,
                settings=PERTURBATION_SETTINGS,
                seed=seed,
            )
            bucket = accum[f"perturbations_{key}"]
            for row in rows:
                key2 = f"{row['kind']}:{row['level']}"
                bucket.setdefault(key2, []).append(row["f1_macro"])

        if len(result.accepted_y) > 0:
            physical = summarize_physical(
                split.x_test,
                result.accepted_x,
                voltage_max=args.voltage_max,
                current_max=args.current_max,
            )
            accum["physical"].append(physical)
            class_totals.append(
                np.bincount(result.accepted_y, minlength=n_classes).astype(np.float64)
            )
            iteration_totals.append(
                [row.accepted_samples for row in result.history]
            )
        else:
            class_totals.append(np.zeros(n_classes))
            iteration_totals.append([row.accepted_samples for row in result.history])

        f1_runs.append(metrics_after["f1_macro"])
        stop_runs.append(len(result.history))
        if run_idx == 0:
            first_run_artifacts = {
                "split": split,
                "result": result,
                "pred_before": pred_before,
                "pred_after": pred_after,
            }
        print(
            f"[run {run_idx + 1}] before F1={metrics_before['f1_macro']:.2f} "
            f"after F1={metrics_after['f1_macro']:.2f} iterations={len(result.history)}"
        )

    report: dict = {}
    conf_before = normalize_confusion_counts(accum["confusion_before"])
    conf_after = normalize_confusion_counts(accum["confusion_after"])
    report["confusion_before"] = conf_before.round(2).tolist()
    report["confusion_after"] = conf_after.round(2).tolist()
    # Class-pair table from the pooled confusion counts.
    def pairs_from_counts(counts: np.ndarray) -> list[dict]:
        matrix = normalize_confusion_counts(counts)
        pairs = []
        for true_id in range(n_classes):
            for pred_id in range(n_classes):
                if true_id != pred_id and matrix[true_id, pred_id] > 0:
                    pairs.append(
                        {
                            "true": class_names[true_id],
                            "predicted": class_names[pred_id],
                            "rate": float(matrix[true_id, pred_id]),
                        }
                    )
        pairs.sort(key=lambda item: item["rate"], reverse=True)
        return pairs[:10]

    report["class_pairs_before"] = pairs_from_counts(accum["confusion_before"])
    report["class_pairs_after"] = pairs_from_counts(accum["confusion_after"])

    for key in ("perturbations_before", "perturbations_after"):
        report[key] = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for name, values in accum[key].items()
        }

    if accum["physical"]:
        keys = accum["physical"][0].keys()
        report["physical_parameters"] = {
            key: {
                metric: float(
                    np.mean([run[key][metric] for run in accum["physical"]])
                )
                for metric in ("real_mean", "generated_mean", "relative_error_pct")
            }
            for key in keys
        }

    report["f1_macro"] = {
        "mean": float(np.mean(f1_runs)),
        "std": float(np.std(f1_runs, ddof=1)) if len(f1_runs) > 1 else 0.0,
    }
    report["stopping_iterations"] = {
        "mean": float(np.mean(stop_runs)),
        "std": float(np.std(stop_runs, ddof=1)) if len(stop_runs) > 1 else 0.0,
    }
    report["accepted_by_class"] = {
        class_names[i]: float(np.mean([row[i] for row in class_totals]))
        for i in range(n_classes)
    }

    # Operating regimes, when metadata are available.
    if first_run_artifacts is not None:
        split = first_run_artifacts["split"]
        meta = split.test_metadata
        if args.irradiance_column in meta:
            report["irradiance_regimes"] = regime_report(
                split.y_test,
                first_run_artifacts["pred_after"],
                np.asarray(meta[args.irradiance_column], dtype=float),
                IRRADIANCE_BINS,
                n_classes,
            )
        if args.temperature_column in meta:
            report["temperature_regimes"] = regime_report(
                split.y_test,
                first_run_artifacts["pred_after"],
                np.asarray(meta[args.temperature_column], dtype=float),
                TEMPERATURE_BINS,
                n_classes,
            )

    # Group-held-out shifts.
    if not args.no_group_heldout:
        report["group_heldout"] = run_group_heldout(args, config, device)

    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    plot_confusion_matrices(
        conf_before,
        conf_after,
        class_names,
        figures_dir / "confusion_matrices.png",
    )
    if accum["physical"] and first_run_artifacts is not None:
        result = first_run_artifacts["result"]
        split = first_run_artifacts["split"]
        real_by_class = {
            c: split.x_test[split.y_test == c] for c in range(n_classes) if np.any(split.y_test == c)
        }
        generated_by_class = {
            c: result.accepted_x[result.accepted_y == c]
            for c in range(n_classes)
            if np.any(result.accepted_y == c)
        }
        try:
            plot_real_vs_generated(
                real_by_class, generated_by_class, class_names, figures_dir / "real_vs_generated.png"
            )
        except Exception as exc:  # plotting is best-effort for small demo data
            print(f"skipping real-vs-generated figure: {exc}")
    plot_quota_distribution(
        class_totals, iteration_totals, class_names, figures_dir / "quota_distribution.png"
    )
    plot_run_level_f1(
        {"proposed": f1_runs},
        {"proposed": stop_runs},
        figures_dir / "f1_convergence.png",
    )
    print(f"analysis written to {args.output_dir}")


def run_group_heldout(args, config, device) -> dict:
    """Evaluate unseen modules and unseen operating conditions."""
    x, y, _, metadata = load_fault_dataset(config.data)
    results: dict = {}
    for label, column in (("unseen_module", args.module_column), ("unseen_condition", args.condition_column)):
        if column not in metadata:
            continue
        split = make_group_heldout_split(
            x, y, np.asarray(metadata[column]), config.data, config.data.random_seed
        )
        baseline = train_no_augmentation(config, split, device)
        pred, prob, metrics = collect_predictions(baseline, split, device)
        results[label] = {"no_augmentation": metrics["f1_macro"]}
        runner = ClosedLoopAugmentor(config)
        outcome = runner.run(split, config.data.random_seed)
        results[label]["proposed"] = outcome.final_metrics["f1_macro"]
    return results


if __name__ == "__main__":
    main()

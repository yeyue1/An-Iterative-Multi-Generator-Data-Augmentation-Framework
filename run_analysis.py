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
    class_pair_table_from_counts,
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
)
from pv_fault_aug.metrics import classification_metrics, per_class_f1
from pv_fault_aug.models.classifier import predict_proba, train_classifier
from pv_fault_aug.pipeline import ClosedLoopAugmentor, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run error analysis, robustness tests, and physical-consistency checks."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", default="scarce", choices=["balanced", "imbalanced", "scarce"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--group-heldout-runs",
        type=int,
        default=None,
        help="Runs for the unseen-module/condition shifts. Defaults to --runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="Seed of the shared partition. Every run reuses it.",
    )
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
    config.variant = args.variant
    config.apply_variant()
    config.validate()
    return config


def train_no_augmentation(config, split, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.nanmean(array)), std


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
        "regime_irradiance": {},
        "regime_temperature": {},
    }
    per_class_before: dict[str, list[float]] = {name: [] for name in class_names}
    per_class_after: dict[str, list[float]] = {name: [] for name in class_names}
    class_totals: list[np.ndarray] = []
    iteration_totals: list[list[int]] = []
    clean_proposed_f1: list[float] = []
    real_by_class: dict[int, np.ndarray] = {}
    generated_by_class: dict[int, list[np.ndarray]] = {c: [] for c in range(n_classes)}

    split = load_scenario_split(
        config.data, args.scenario, config.data.random_seed
    )
    for class_id in range(n_classes):
        if np.any(split.y_train == class_id):
            real_by_class[class_id] = split.x_train[split.y_train == class_id]

    for run_idx in range(args.runs):
        seed = config.data.random_seed + run_idx * 100
        runner = ClosedLoopAugmentor(config)
        result = runner.run(split, seed)
        runner.save_result(result, args.output_dir / f"run_{run_idx + 1}")
        baseline_model = train_no_augmentation(config, split, device, seed)

        pred_before, prob_before, metrics_before = collect_predictions(
            baseline_model, split, device
        )
        pred_after, prob_after, metrics_after = collect_predictions(
            result.model, split, device
        )
        clean_proposed_f1.append(float(metrics_after["f1_macro"]))
        accum["confusion_before"] += confusion_counts(split.y_test, pred_before, n_classes)
        accum["confusion_after"] += confusion_counts(split.y_test, pred_after, n_classes)

        f1_before = per_class_f1(split.y_test, pred_before, n_classes)
        f1_after = per_class_f1(split.y_test, pred_after, n_classes)
        for class_id, name in enumerate(class_names):
            per_class_before[name].append(float(f1_before[class_id]))
            per_class_after[name].append(float(f1_after[class_id]))

        for key, model in (("before", baseline_model), ("after", result.model)):
            rows = evaluate_perturbations(
                lambda x, m=model: predict_proba(m, x, device),
                split.x_test,
                split.y_test,
                n_classes,
                settings=PERTURBATION_SETTINGS,
                seed=seed,
            )
            bucket = accum[f"perturbations_{key}"]
            for row in rows:
                setting = f"{row['kind']}:{row['level']}"
                bucket.setdefault(setting, []).append(row["f1_macro"])

        meta = split.test_metadata
        for column, bins, bucket_name in (
            (args.irradiance_column, IRRADIANCE_BINS, "regime_irradiance"),
            (args.temperature_column, TEMPERATURE_BINS, "regime_temperature"),
        ):
            if column not in meta:
                continue
            values = np.asarray(meta[column], dtype=float)
            for label, pred in (("before", pred_before), ("after", pred_after)):
                rows = regime_report(
                    split.y_test, pred, values, bins, n_classes
                )
                bucket = accum[bucket_name]
                for row in rows:
                    entry = bucket.setdefault(
                        (row["low"], row["high"]),
                        {
                            "samples": 0,
                            "before": [],
                            "after": [],
                        },
                    )
                    # Count each test record once; the before/after predictions
                    # cover the same records.
                    if label == "before":
                        entry["samples"] += row["samples"]
                    entry[label].append(row["f1_macro"])

        if len(result.accepted_y) == 0:
            class_totals.append(np.zeros(n_classes, dtype=np.float64))
            iteration_totals.append([row.accepted_samples for row in result.history])
        else:
            for class_id in range(n_classes):
                if np.any(result.accepted_y == class_id):
                    generated_by_class[class_id].append(
                        result.accepted_x[result.accepted_y == class_id]
                    )
            class_totals.append(
                np.bincount(result.accepted_y, minlength=n_classes).astype(np.float64)
            )
            iteration_totals.append([row.accepted_samples for row in result.history])

            normal_real = split.x_train[split.y_train == 0]
            normal_gen = result.accepted_x[result.accepted_y == 0]
            if len(normal_real) and len(normal_gen):
                accum["physical"].append(
                    summarize_physical(
                        normal_real,
                        normal_gen,
                        voltage_max=args.voltage_max,
                        current_max=args.current_max,
                    )
                )
        print(
            f"[run {run_idx + 1}] before F1={metrics_before['f1_macro']:.2f} "
            f"after F1={metrics_after['f1_macro']:.2f} iterations={len(result.history)}"
        )

    report = build_report(
        args, config, accum, per_class_before, per_class_after, class_names, class_totals
    )
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    write_tables(args.output_dir, report, class_names)

    if not args.no_group_heldout:
        clean_mean = float(np.mean(clean_proposed_f1)) if clean_proposed_f1 else None
        report["group_heldout"] = run_group_heldout(
            args, config, device, clean_proposed_mean=clean_mean
        )
        with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        write_group_heldout(args.output_dir, report["group_heldout"])

    conf_before = normalize_confusion_counts(accum["confusion_before"])
    conf_after = normalize_confusion_counts(accum["confusion_after"])
    plot_confusion_matrices(
        conf_before, conf_after, class_names, figures_dir / "confusion_matrices.png"
    )
    if real_by_class:
        generated_means = {
            c: np.vstack(parts) for c, parts in generated_by_class.items() if parts
        }
        try:
            plot_real_vs_generated(
                real_by_class,
                generated_means,
                class_names,
                figures_dir / "real_vs_generated.png",
                physical_class=0,
                voltage_max=args.voltage_max,
                current_max=args.current_max,
            )
        except Exception as exc:  # plotting is best-effort for small demo data
            print(f"skipping real-vs-generated figure: {exc}")
    if args.scenario == "scarce" and class_totals:
        plot_quota_distribution(
            class_totals,
            iteration_totals,
            class_names,
            figures_dir / "quota_distribution.png",
        )
    print(f"analysis written to {args.output_dir}")


def build_report(
    args,
    config,
    accum,
    per_class_before,
    per_class_after,
    class_names,
    class_totals,
) -> dict:
    report: dict = {}
    report["confusion_before"] = normalize_confusion_counts(
        accum["confusion_before"]
    ).round(2).tolist()
    report["confusion_after"] = normalize_confusion_counts(
        accum["confusion_after"]
    ).round(2).tolist()
    report["confusion_counts_before"] = accum["confusion_before"].astype(int).tolist()
    report["confusion_counts_after"] = accum["confusion_after"].astype(int).tolist()
    report["per_class_f1"] = {
        name: {
            "no_augmentation": dict(
                zip(("mean", "std"), _mean_std(per_class_before[name]))
            ),
            "proposed": dict(zip(("mean", "std"), _mean_std(per_class_after[name]))),
            "gain": float(np.mean(per_class_after[name]) - np.mean(per_class_before[name])),
        }
        for name in class_names
    }
    report["class_pairs_before"] = class_pair_table_from_counts(
        accum["confusion_before"], class_names
    )
    report["class_pairs_after"] = class_pair_table_from_counts(
        accum["confusion_after"], class_names
    )

    for key in ("perturbations_before", "perturbations_after"):
        report[key] = {
            name: dict(zip(("mean", "std"), _mean_std(values)))
            for name, values in accum[key].items()
        }
    clean = report.get("perturbations_after", {}).get("none:0.0", {}).get("mean")
    if clean is not None:
        for name, stats in report["perturbations_after"].items():
            stats["delta_vs_clean"] = float(stats["mean"] - clean)

    if accum["physical"]:
        keys = accum["physical"][0].keys()
        report["physical_parameters"] = {
            key: {
                metric: float(np.mean([run[key][metric] for run in accum["physical"]]))
                for metric in ("real_mean", "generated_mean", "relative_error_pct")
            }
            for key in keys
        }

    regimes = {}
    for bucket_name, label in (
        ("regime_irradiance", "irradiance"),
        ("regime_temperature", "temperature"),
    ):
        bucket = accum[bucket_name]
        if not bucket:
            continue
        rows = []
        for (low, high), entry in sorted(bucket.items()):
            before_mean, before_std = _mean_std(entry["before"])
            after_mean, after_std = _mean_std(entry["after"])
            rows.append(
                {
                    "regime": label,
                    "low": low,
                    "high": high,
                    "samples_pooled": int(entry["samples"]),
                    "no_augmentation_mean": before_mean,
                    "no_augmentation_std": before_std,
                    "proposed_mean": after_mean,
                    "proposed_std": after_std,
                    "gain": float(after_mean - before_mean),
                }
            )
        regimes[label] = rows
    if regimes:
        report["regimes"] = regimes

    report["accepted_by_class"] = {
        class_names[i]: {
            "mean": float(np.mean([row[i] for row in class_totals])),
            "std": float(np.std([row[i] for row in class_totals], ddof=1))
            if len(class_totals) > 1
            else 0.0,
        }
        for i in range(len(class_names))
    }
    return report


def write_tables(output_dir: Path, report: dict, class_names: list[str]) -> None:
    with (output_dir / "per_class_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("class,no_augmentation_mean,no_augmentation_std,proposed_mean,proposed_std,gain\n")
        for name, stats in report["per_class_f1"].items():
            fh.write(
                f"{name},{stats['no_augmentation']['mean']:.3f},"
                f"{stats['no_augmentation']['std']:.3f},"
                f"{stats['proposed']['mean']:.3f},{stats['proposed']['std']:.3f},"
                f"{stats['gain']:.3f}\n"
            )

    after_counts = np.asarray(report["confusion_counts_after"], dtype=np.float64)
    after_full = {
        (row["true"], row["predicted"]): row
        for row in class_pair_table_from_counts(
            after_counts, class_names, top_k=10_000
        )
    }
    with (output_dir / "class_pairs.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "true,predicted,before_count,support,before_pct,after_count,after_pct,reduction_pp\n"
        )
        for row in report["class_pairs_before"]:
            after = after_full.get((row["true"], row["predicted"]))
            after_count = after["count"] if after else 0
            after_pct = after["rate"] if after else 0.0
            fh.write(
                f"{row['true']},{row['predicted']},{row['count']},{row['support']},"
                f"{row['rate']:.2f},{after_count},{after_pct:.2f},"
                f"{row['rate'] - after_pct:.2f}\n"
            )

    if "regimes" in report:
        with (output_dir / "regimes.csv").open("w", encoding="utf-8") as fh:
            fh.write(
                "regime,low,high,samples_pooled,no_augmentation_mean,"
                "no_augmentation_std,proposed_mean,proposed_std,gain\n"
            )
            for rows in report["regimes"].values():
                for row in rows:
                    fh.write(
                        f"{row['regime']},{row['low']},{row['high']},"
                        f"{row['samples_pooled']},"
                        f"{row['no_augmentation_mean']:.3f},"
                        f"{row['no_augmentation_std']:.3f},"
                        f"{row['proposed_mean']:.3f},{row['proposed_std']:.3f},"
                        f"{row['gain']:.3f}\n"
                    )

    with (output_dir / "perturbations.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "setting,no_augmentation_mean,no_augmentation_std,"
            "proposed_mean,proposed_std,delta_vs_clean\n"
        )
        for name in report["perturbations_after"]:
            after = report["perturbations_after"][name]
            before = report["perturbations_before"].get(name, {})
            fh.write(
                f"{name},{before.get('mean', float('nan')):.3f},"
                f"{before.get('std', float('nan')):.3f},"
                f"{after['mean']:.3f},{after['std']:.3f},"
                f"{after.get('delta_vs_clean', float('nan')):+.3f}\n"
            )

    if "physical_parameters" in report:
        with (output_dir / "physical_parameters.csv").open("w", encoding="utf-8") as fh:
            fh.write("parameter,real_mean,generated_mean,relative_error_pct\n")
            for key, row in report["physical_parameters"].items():
                fh.write(
                    f"{key},{row['real_mean']:.4f},{row['generated_mean']:.4f},"
                    f"{row['relative_error_pct']:.2f}\n"
                )


def run_group_heldout(args, config, device, clean_proposed_mean=None) -> dict:
    """Evaluate unseen modules and unseen operating conditions.

    Each shift uses its own group-held-out partition; the reported value is the
    mean and sample standard deviation over the configured number of runs.
    ``delta_vs_clean`` follows Table 15: the proposed group-held-out F1 minus
    the proposed F1 on the clean split, not the difference against the
    no-augmentation baseline of the same shifted split.
    """
    x, y, _, metadata = load_fault_dataset(config.data)
    n_runs = args.group_heldout_runs or args.runs
    results: dict = {}
    for label, column in (
        ("unseen_module", args.module_column),
        ("unseen_condition", args.condition_column),
    ):
        if column not in metadata:
            print(f"[skip] metadata column '{column}' not available")
            continue
        before_values: list[float] = []
        after_values: list[float] = []
        for run_idx in range(n_runs):
            seed = config.data.random_seed + run_idx * 100
            try:
                split = make_group_heldout_split(
                    x, y, np.asarray(metadata[column]), config.data, seed
                )
                baseline = train_no_augmentation(config, split, device, seed)
                _, _, metrics = collect_predictions(baseline, split, device)
                runner = ClosedLoopAugmentor(config)
                outcome = runner.run(split, seed)
            except ValueError as exc:
                print(f"[skip] {label} run {run_idx + 1}: {exc}")
                continue
            before_values.append(float(metrics["f1_macro"]))
            after_values.append(float(outcome.final_metrics["f1_macro"]))
        if not before_values:
            continue
        before_mean, before_std = _mean_std(before_values)
        after_mean, after_std = _mean_std(after_values)
        results[label] = {
            "no_augmentation_mean": before_mean,
            "no_augmentation_std": before_std,
            "proposed_mean": after_mean,
            "proposed_std": after_std,
            "gain_vs_no_augmentation": float(after_mean - before_mean),
            "delta_vs_clean": (
                float(after_mean - clean_proposed_mean)
                if clean_proposed_mean is not None
                else None
            ),
            "runs": len(after_values),
        }
    return results


def write_group_heldout(output_dir: Path, results: dict) -> None:
    if not results:
        return
    with (output_dir / "group_heldout.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "shift,no_augmentation_mean,no_augmentation_std,"
            "proposed_mean,proposed_std,gain_vs_no_augmentation,"
            "delta_vs_clean,runs\n"
        )
        for label, row in results.items():
            delta = row.get("delta_vs_clean")
            delta_text = f"{delta:.3f}" if delta is not None else ""
            fh.write(
                f"{label},{row['no_augmentation_mean']:.3f},"
                f"{row['no_augmentation_std']:.3f},"
                f"{row['proposed_mean']:.3f},{row['proposed_std']:.3f},"
                f"{row['gain_vs_no_augmentation']:.3f},{delta_text},"
                f"{row['runs']}\n"
            )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from pv_fault_aug.baselines import BASELINE_METHODS, run_baseline
from pv_fault_aug.config import (
    ExperimentConfig,
    FAULT_CLASSES,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_GENERATORS,
)
from pv_fault_aug.data import load_scenario_split
from pv_fault_aug.pipeline import ClosedLoopAugmentor, resolve_device
from pv_fault_aug.stats import summarize_paired


ALL_METHODS = ("proposed",) + BASELINE_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the comparison table (baselines plus the proposed closed-loop framework)."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", default="all", choices=["balanced", "imbalanced", "scarce", "all"])
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="Seed of the shared partition. All methods and runs reuse it.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier", default="cnn1d", choices=SUPPORTED_CLASSIFIERS)
    parser.add_argument("--generators", default=",".join(SUPPORTED_GENERATORS))
    parser.add_argument(
        "--budget",
        type=int,
        default=500,
        help="Total synthetic budget used by one-shot/static baselines "
        "(and the per-round budget for iterative baselines).",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-prefix", default=None)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--scarce-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_config(args: argparse.Namespace, scenario: str) -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.data_path = args.data_path
    config.data.label_column = args.label_column
    config.data.feature_prefix = args.feature_prefix
    config.data.feature_dim = args.feature_dim
    config.data.scarce_ratio = args.scarce_ratio
    config.data.n_runs = args.runs
    config.scenario = scenario
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


def main() -> None:
    args = parse_args()
    scenarios = ["balanced", "imbalanced", "scarce"] if args.scenario == "all" else [args.scenario]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    for scenario in scenarios:
        config = build_config(args, scenario)
        device = resolve_device(config.device)
        with (args.output_dir / f"config_{scenario}.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(config), fh, default=str, indent=2)
        # Fixed shared partition; seeds below only change model initialization.
        split = load_scenario_split(config.data, scenario, config.data.random_seed)
        for run_idx in range(args.runs):
            seed = config.data.random_seed + run_idx * 100
            for method in methods:
                if method == "proposed":
                    import time

                    started = time.perf_counter()
                    runner = ClosedLoopAugmentor(config)
                    result = runner.run(split, seed)
                    elapsed = time.perf_counter() - started
                    metrics = result.final_metrics
                    per_class = result.per_class_f1
                    row = {
                        "method": "proposed",
                        "scenario": scenario,
                        "run": run_idx + 1,
                        "accepted_samples": int(len(result.accepted_y)),
                        "iterations": len(result.history),
                        "elapsed_sec": elapsed,
                        "per_class_f1": per_class,
                        **metrics,
                    }
                    if result.test_prob is not None:
                        np.savez_compressed(
                            predictions_dir / f"{scenario}_{method}_run{run_idx + 1}.npz",
                            y_true=split.y_test,
                            y_pred=result.test_pred,
                            y_prob=result.test_prob,
                        )
                else:
                    outcome = run_baseline(
                        method, config, split, seed, device, budget=args.budget
                    )
                    row = {
                        "method": outcome.method,
                        "scenario": scenario,
                        "run": run_idx + 1,
                        "accepted_samples": outcome.accepted_samples,
                        "iterations": outcome.iterations,
                        "elapsed_sec": outcome.elapsed_sec,
                        "per_class_f1": outcome.per_class_f1,
                        **outcome.final_metrics,
                    }
                    if outcome.y_prob is not None:
                        np.savez_compressed(
                            predictions_dir / f"{scenario}_{method}_run{run_idx + 1}.npz",
                            y_true=outcome.y_true,
                            y_pred=outcome.y_pred,
                            y_prob=outcome.y_prob,
                        )
                rows.append(row)
                print(
                    f"[{scenario} run {run_idx + 1}] {method}: "
                    f"F1={row['f1_macro']:.2f} time={row['elapsed_sec']:.1f}s"
                )

    write_outputs(rows, args.output_dir, methods, scenarios)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.nanmean(array)), std


def write_outputs(rows: list[dict], output_dir: Path, methods: list[str], scenarios: list[str]) -> None:
    metric_keys = [k for k in rows[0] if k not in ("method", "scenario", "run", "per_class_f1")]
    with (output_dir / "runs.csv").open("w", encoding="utf-8") as fh:
        header = ["method", "scenario", "run", *metric_keys]
        fh.write(",".join(header) + "\n")
        for row in rows:
            values = [str(row.get(key, "")) for key in header]
            fh.write(",".join(values) + "\n")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        summary[method] = {}
        for scenario in scenarios:
            subset = [r for r in rows if r["method"] == method and r["scenario"] == scenario]
            if not subset:
                continue
            stats = {}
            for key in metric_keys:
                values = [r[key] for r in subset]
                mean, std = _mean_std(values)
                stats[key] = {"mean": mean, "std": std}
            summary[method][scenario] = stats
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with (output_dir / "comparison_table.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "method,scenario,f1_macro_mean,f1_macro_std,accuracy_mean,"
            "elapsed_sec_mean,iterations_mean\n"
        )
        for method in methods:
            for scenario in scenarios:
                stats = summary.get(method, {}).get(scenario)
                if not stats:
                    continue
                fh.write(
                    f"{method},{scenario},"
                    f"{stats['f1_macro']['mean']:.3f},{stats['f1_macro']['std']:.3f},"
                    f"{stats['accuracy']['mean']:.3f},"
                    f"{stats['elapsed_sec']['mean']:.2f},"
                    f"{stats['iterations']['mean']:.2f}\n"
                )

    write_per_class(rows, output_dir, methods, scenarios)
    write_cost_table(rows, output_dir, methods)

    if "proposed" in methods:
        paired_rows = []
        for scenario in scenarios:
            proposed = np.asarray(
                [
                    r["f1_macro"]
                    for r in rows
                    if r["method"] == "proposed" and r["scenario"] == scenario
                ],
                dtype=np.float64,
            )
            if len(proposed) == 0:
                continue
            for method in methods:
                if method == "proposed":
                    continue
                other = np.asarray(
                    [
                        r["f1_macro"]
                        for r in rows
                        if r["method"] == method and r["scenario"] == scenario
                    ],
                    dtype=np.float64,
                )
                if len(other) != len(proposed):
                    continue
                stats = summarize_paired(proposed, other)
                paired_rows.append({"method": method, "scenario": scenario, **stats})
        if paired_rows:
            keys = [
                "method",
                "scenario",
                "mean_a",
                "mean_b",
                "mean_difference",
                "std_difference",
                "ci_low",
                "ci_high",
                "n_runs",
            ]
            with (output_dir / "paired_vs_proposed.csv").open("w", encoding="utf-8") as fh:
                fh.write(",".join(keys) + "\n")
                for row in paired_rows:
                    fh.write(",".join(str(row[key]) for key in keys) + "\n")
    print(f"results written to {output_dir}")


def write_per_class(
    rows: list[dict], output_dir: Path, methods: list[str], scenarios: list[str]
) -> None:
    """Per-class F1 mean and standard deviation across runs (Table 8)."""
    with (output_dir / "per_class_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("method,scenario,class,f1_mean,f1_std\n")
        for method in methods:
            for scenario in scenarios:
                subset = [
                    r for r in rows if r["method"] == method and r["scenario"] == scenario
                ]
                if not subset:
                    continue
                for name in FAULT_CLASSES:
                    mean, std = _mean_std([r["per_class_f1"][name] for r in subset])
                    fh.write(f"{method},{scenario},{name},{mean:.4f},{std:.4f}\n")


def write_cost_table(rows: list[dict], output_dir: Path, methods: list[str]) -> None:
    """Offline cost table of Table 6 for the scarce scenario."""
    scarce = [r for r in rows if r["scenario"] == "scarce"]
    if not scarce:
        return
    ordered: list[str] = []
    for method in ("no_augmentation", "proposed", *methods):
        if method not in ordered:
            ordered.append(method)
    baseline_time = None
    lines = [
        "method,time_min_mean,time_min_std,iterations_mean,"
        "iterations_std,relative_cost,f1_macro_mean,f1_macro_std"
    ]
    for method in ordered:
        subset = [r for r in scarce if r["method"] == method]
        if not subset:
            continue
        time_mean, time_std = _mean_std([r["elapsed_sec"] / 60.0 for r in subset])
        f1_mean, f1_std = _mean_std([r["f1_macro"] for r in subset])
        if method == "proposed":
            iterations_mean, iterations_std = _mean_std(
                [r["iterations"] for r in subset]
            )
        else:
            iterations_mean, iterations_std = float("nan"), float("nan")
        if method == "no_augmentation":
            baseline_time = time_mean
        relative = (
            time_mean / baseline_time
            if baseline_time is not None and baseline_time > 0
            else float("nan")
        )
        lines.append(
            f"{method},{time_mean:.3f},{time_std:.3f},"
            f"{iterations_mean:.2f},{iterations_std:.2f},"
            f"{relative:.2f},{f1_mean:.3f},{f1_std:.3f}"
        )
    (output_dir / "cost_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
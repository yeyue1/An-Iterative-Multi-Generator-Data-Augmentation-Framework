from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from pv_fault_aug.baselines import run_no_augmentation, run_oneshot
from pv_fault_aug.config import (
    ExperimentConfig,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_GENERATORS,
)
from pv_fault_aug.data import load_scenario_split
from pv_fault_aug.pipeline import ClosedLoopAugmentor, resolve_device


DEFAULT_POOLS = (
    ("cvae",),
    ("wgan_gp",),
    ("ctgan",),
    ("vaegan",),
    ("ddpm",),
    ("cddpm",),
    ("cvae", "wgan_gp"),
    ("cvae", "wgan_gp", "ctgan"),
    ("cvae", "wgan_gp", "ctgan", "vaegan"),
    ("cvae", "wgan_gp", "ctgan", "vaegan", "ddpm"),
    ("cvae", "wgan_gp", "ctgan", "vaegan", "ddpm", "cddpm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generator-pool comparison under a fixed or natural augmentation budget."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scenario", default="scarce", choices=["balanced", "imbalanced", "scarce"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generator_pool"))
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help="Seed of the shared partition. Every pool and run reuses it.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier", default="cnn1d", choices=SUPPORTED_CLASSIFIERS)
    parser.add_argument(
        "--mode",
        default="fixed",
        choices=["fixed", "natural"],
        help="fixed: identical accepted-sample budget for every pool (Table 13); "
        "natural: budget scales with the pool (Table 12).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=500,
        help="Accepted synthetic-sample budget per run in fixed mode.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--pools",
        default=None,
        help="Optional semicolon-separated pools, e.g. 'cvae;cvae+wgan_gp'. "
        "Defaults to the manuscript pools.",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-prefix", default=None)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--scarce-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_config(
    args: argparse.Namespace, names: tuple[str, ...], output_dir: Path | None = None
) -> ExperimentConfig:
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
    config.generators.names = names
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
    unknown = sorted({n for pool in DEFAULT_POOLS for n in pool} - set(SUPPORTED_GENERATORS))
    if unknown:
        raise ValueError(f"Unknown generators in default pools: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    pools = DEFAULT_POOLS
    if args.pools:
        pools = tuple(
            tuple(name for name in pool.split("+") if name)
            for pool in args.pools.split(";")
            if pool.strip()
        )
    if args.mode == "natural" and not pools:
        raise ValueError("At least one generator pool is required")

    # One fixed partition shared by every pool and run (Section 3.3).
    base_config = build_config(args, ("cvae",))
    split = load_scenario_split(
        base_config.data, args.scenario, base_config.data.random_seed
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(base_config), fh, default=str, indent=2)

    rows: list[dict] = []
    for run_idx in range(args.runs):
        seed = base_config.data.random_seed + run_idx * 100
        no_aug = run_no_augmentation(base_config, split, seed, device)
        rows.append(
            {
                "pool": "No augmentation",
                "pool_size": 0,
                "scenario": args.scenario,
                "mode": args.mode,
                "run": run_idx + 1,
                "accepted_samples": 0,
                "iterations": 0,
                "f1_macro": no_aug.final_metrics["f1_macro"],
                "accuracy": no_aug.final_metrics["accuracy"],
                "elapsed_sec": no_aug.elapsed_sec,
            }
        )
        for pool in pools:
            config = build_config(args, pool)
            budget = args.budget if args.mode == "fixed" else None
            if args.mode == "natural":
                import time

                started = time.perf_counter()
                runner = ClosedLoopAugmentor(config)
                result = runner.run(split, seed)
                elapsed = time.perf_counter() - started
                accepted = int(len(result.accepted_y))
                iterations = len(result.history)
                metrics = result.final_metrics
            else:
                outcome = run_oneshot(
                    config,
                    split,
                    seed,
                    device,
                    generator_names=pool,
                    budget=budget,
                    method_name="+".join(pool),
                )
                elapsed = outcome.elapsed_sec
                accepted = outcome.accepted_samples
                iterations = 1
                metrics = outcome.final_metrics
            rows.append(
                {
                    "pool": "+".join(pool),
                    "pool_size": len(pool),
                    "scenario": args.scenario,
                    "mode": args.mode,
                    "run": run_idx + 1,
                    "accepted_samples": accepted,
                    "iterations": iterations,
                    "f1_macro": metrics["f1_macro"],
                    "accuracy": metrics["accuracy"],
                    "elapsed_sec": elapsed,
                }
            )
            print(
                f"[{'+'.join(pool)} run {run_idx + 1}] F1={metrics['f1_macro']:.2f} "
                f"accepted={accepted}"
            )

    write_outputs(rows, base_config, args, pools)


def write_outputs(
    rows: list[dict],
    config: ExperimentConfig,
    args: argparse.Namespace,
    pools: tuple[tuple[str, ...], ...],
) -> None:
    header = [
        "pool",
        "pool_size",
        "mode",
        "run",
        "accepted_samples",
        "iterations",
        "f1_macro",
        "accuracy",
        "elapsed_sec",
    ]
    with (args.output_dir / f"pool_{args.mode}_runs.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(row[key]) for key in header) + "\n")

    order = ["+".join(pool) for pool in pools]
    if args.mode == "fixed":
        order = ["No augmentation"] + order
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for pool in order:
        subset = [r for r in rows if r["pool"] == pool]
        if not subset:
            continue
        stats = summary.setdefault(pool, {})
        for key in ("f1_macro", "accuracy", "accepted_samples", "iterations", "elapsed_sec"):
            values = np.asarray([r[key] for r in subset], dtype=np.float64)
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    with (args.output_dir / f"pool_{args.mode}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    baseline = np.asarray(
        [r["f1_macro"] for r in rows if r["pool"] == "No augmentation"],
        dtype=np.float64,
    )
    with (args.output_dir / f"pool_{args.mode}_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "pool,accepted_mean,accepted_std,f1_macro_mean,f1_macro_std,"
            "delta_vs_no_augmentation,iterations_mean,iterations_std\n"
        )
        for pool in order:
            stats = summary.get(pool)
            if not stats:
                continue
            if pool == "No augmentation":
                delta = 0.0
            else:
                values = np.asarray(
                    [r["f1_macro"] for r in rows if r["pool"] == pool], dtype=np.float64
                )
                delta = float(np.mean(values - baseline)) if len(values) == len(baseline) else float("nan")
            fh.write(
                f"{pool},{stats['accepted_samples']['mean']:.2f},"
                f"{stats['accepted_samples']['std']:.2f},"
                f"{stats['f1_macro']['mean']:.3f},{stats['f1_macro']['std']:.3f},"
                f"{delta:+.3f},"
                f"{stats['iterations']['mean']:.2f},"
                f"{stats['iterations']['std']:.2f}\n"
            )
    print(f"results written to {args.output_dir}")


if __name__ == "__main__":
    main()
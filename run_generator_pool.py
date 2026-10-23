from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from pv_fault_aug.baselines import run_oneshot
from pv_fault_aug.config import ExperimentConfig, SUPPORTED_CLASSIFIERS, SUPPORTED_GENERATORS
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier", default="cnn1d", choices=SUPPORTED_CLASSIFIERS)
    parser.add_argument(
        "--mode",
        default="fixed",
        choices=["fixed", "natural"],
        help="fixed: identical accepted-sample budget for every pool; natural: budget scales with the pool.",
    )
    parser.add_argument("--budget", type=int, default=500, help="Budget per run in fixed mode.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--pools",
        default=None,
        help="Optional semicolon-separated pools, e.g. 'cvae;cvae+wgan_gp'. Defaults to the manuscript pools.",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-prefix", default=None)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--scarce-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_config(args: argparse.Namespace, names: tuple[str, ...]) -> ExperimentConfig:
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
    if args.quick:
        config.classifier.epochs = 20
        config.classifier.feedback_epochs = 5
        config.classifier.patience = 5
        config.generators.epochs = 10
        config.generators.quick_epochs = 5
    config.validate()
    return config


def main() -> None:
    args = parse_args()
    unknown = sorted({n for pool in DEFAULT_POOLS for n in pool} - set(SUPPORTED_GENERATORS))
    if unknown:
        raise ValueError(f"Unknown generators in default pools: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rows: list[dict] = []
    pools = DEFAULT_POOLS
    if args.pools:
        pools = tuple(
            tuple(name for name in pool.split("+") if name)
            for pool in args.pools.split(";")
            if pool.strip()
        )

    for pool in pools:
        config = build_config(args, pool)
        budget = args.budget if args.mode == "fixed" else None
        for run_idx in range(args.runs):
            seed = config.data.random_seed + run_idx * 100
            split = load_scenario_split(config.data, args.scenario, seed)
            if args.mode == "natural":
                import time

                started = time.perf_counter()
                runner = ClosedLoopAugmentor(config)
                result = runner.run(split, seed)
                elapsed = time.perf_counter() - started
                accepted = int(len(result.accepted_y)) if result.accepted_y is not None else 0
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
                metrics = outcome.final_metrics
            rows.append(
                {
                    "pool": "+".join(pool),
                    "pool_size": len(pool),
                    "scenario": args.scenario,
                    "mode": args.mode,
                    "run": run_idx + 1,
                    "accepted_samples": accepted,
                    "f1_macro": metrics["f1_macro"],
                    "accuracy": metrics["accuracy"],
                    "elapsed_sec": elapsed,
                }
            )
            print(
                f"[{'+'.join(pool)} run {run_idx + 1}] F1={metrics['f1_macro']:.2f} "
                f"accepted={accepted}"
            )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        stats = summary.setdefault(row["pool"], {})
        for key in ("f1_macro", "accuracy", "accepted_samples", "elapsed_sec"):
            stats.setdefault(key, {"values": []})["values"].append(row[key])
    for pool, stats in summary.items():
        for key, entry in stats.items():
            values = np.asarray(entry.pop("values"), dtype=np.float64)
            entry["mean"] = float(np.mean(values))
            entry["std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    with (args.output_dir / f"pool_{args.mode}_runs.csv").open("w", encoding="utf-8") as fh:
        header = ["pool", "pool_size", "mode", "run", "accepted_samples", "f1_macro", "accuracy", "elapsed_sec"]
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(row[key]) for key in header) + "\n")
    with (args.output_dir / f"pool_{args.mode}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"results written to {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pv_fault_aug.config import FAULT_CLASSES
from pv_fault_aug.figures import plot_quota_distribution, plot_run_level_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build run-level convergence and allocation figures from run_experiment outputs."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--scenarios", default="balanced,imbalanced,scarce")
    parser.add_argument("--class-names", default=",".join(FAULT_CLASSES))
    return parser.parse_args()


def collect_run_dirs(root: Path, scenario: str) -> list[Path]:
    return sorted(
        [p for p in root.glob(f"{scenario}_run_*") if p.is_dir()],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [name for name in args.class_names.split(",") if name]
    n_classes = len(class_names)

    f1_values: dict[str, list[float]] = {}
    stop_values: dict[str, list[float]] = {}
    class_totals: list[np.ndarray] = []
    iteration_totals: list[list[int]] = []
    found_any = False

    for scenario in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        run_dirs = collect_run_dirs(args.runs_root, scenario)
        if not run_dirs:
            print(f"[skip] no run directories for scenario '{scenario}' under {args.runs_root}")
            continue
        found_any = True
        f1_values[scenario] = []
        stop_values[scenario] = []
        for run_dir in run_dirs:
            with (run_dir / "closed_loop_result.json").open(encoding="utf-8") as fh:
                payload = json.load(fh)
            f1_values[scenario].append(float(payload["final_metrics"]["f1_macro"]))
            history = payload.get("history", [])
            stop_values[scenario].append(float(len(history)))
            iteration_totals.append([int(row["accepted_samples"]) for row in history])
            arrays_path = run_dir / "run_arrays.npz"
            if arrays_path.exists():
                arrays = np.load(arrays_path, allow_pickle=True)
                if "accepted_y" in arrays.files:
                    class_totals.append(
                        np.bincount(arrays["accepted_y"], minlength=n_classes).astype(float)
                    )

    if not found_any:
        raise SystemExit(f"No run directories found under {args.runs_root}")

    plot_run_level_f1(f1_values, stop_values, args.output_dir / "f1_convergence.png")
    if class_totals and iteration_totals:
        plot_quota_distribution(
            class_totals,
            iteration_totals,
            class_names,
            args.output_dir / "quota_distribution.png",
        )
    print(f"figures written to {args.output_dir}")


if __name__ == "__main__":
    main()

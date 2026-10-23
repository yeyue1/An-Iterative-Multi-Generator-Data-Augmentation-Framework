from __future__ import annotations

"""Figure generation for the manuscript.

All functions write PNG files and use the same visual conventions as the
manuscript: no titles inside the figures, one-colour family for each scenario,
and explicit axis labels/legends.
"""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO_COLORS = {
    "balanced": "#1f77b4",
    "imbalanced": "#ff7f0e",
    "scarce": "#2ca02c",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_run_level_f1(
    f1_values: dict[str, list[float]],
    stop_values: dict[str, list[float]],
    path: Path,
    ylabel: str = "Final F1-macro (%)",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    scenarios = list(f1_values)
    for ax, values, axis_label in (
        (axes[0], f1_values, ylabel),
        (axes[1], stop_values, "Stopping iteration"),
    ):
        for idx, scenario in enumerate(scenarios):
            data = np.asarray(values.get(scenario, []), dtype=float)
            if len(data) == 0:
                continue
            jitter = np.linspace(-0.08, 0.08, len(data))
            ax.scatter(
                np.full(len(data), idx) + jitter,
                data,
                s=18,
                color=SCENARIO_COLORS.get(scenario, f"C{idx}"),
                label=scenario.capitalize(),
                zorder=3,
            )
            mean = data.mean()
            std = data.std(ddof=1) if len(data) > 1 else 0.0
            ax.errorbar(
                idx,
                mean,
                yerr=std,
                fmt="none",
                color="black",
                capsize=6,
                linewidth=1.2,
                zorder=2,
            )
            ax.hlines(mean, idx - 0.18, idx + 0.18, color="black", linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.set_ylabel(axis_label)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    _save(fig, path)


def plot_stopping_iterations(run_values: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    scenarios = list(run_values)
    for idx, scenario in enumerate(scenarios):
        data = np.asarray(run_values[scenario], dtype=float)
        if len(data) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(data))
        ax.scatter(
            np.full(len(data), idx) + jitter,
            data,
            s=18,
            color=SCENARIO_COLORS.get(scenario, f"C{idx}"),
            zorder=3,
        )
        mean = data.mean()
        std = data.std(ddof=1) if len(data) > 1 else 0.0
        ax.errorbar(idx, mean, yerr=std, fmt="none", color="black", capsize=6, zorder=2)
        ax.hlines(mean, idx - 0.18, idx + 0.18, color="black", zorder=4)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel("Stopping iteration")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    _save(fig, path)


def plot_quota_distribution(
    per_run_class_totals: list[np.ndarray],
    per_run_iteration_totals: list[list[int]],
    class_names: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    matrix = np.vstack(per_run_class_totals)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
    positions = np.arange(len(class_names))
    axes[0].bar(positions, mean, yerr=std, capsize=4, color="#4c72b0", alpha=0.85)
    for row in matrix:
        axes[0].scatter(
            positions + np.linspace(-0.12, 0.12, len(matrix)),
            row,
            s=14,
            color="black",
            alpha=0.6,
            zorder=3,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(class_names, rotation=30, ha="right")
    axes[0].set_ylabel("Cumulative accepted samples")

    max_iter = max((len(run) for run in per_run_iteration_totals), default=0)
    for run in per_run_iteration_totals:
        axes[1].plot(range(1, len(run) + 1), run, color="#8c8c8c", linewidth=0.9, alpha=0.8)
    if per_run_iteration_totals:
        grid = np.full((len(per_run_iteration_totals), max_iter), np.nan)
        for row, run in enumerate(per_run_iteration_totals):
            grid[row, : len(run)] = run
        mean_curve = np.nanmean(grid, axis=0)
        std_curve = np.nanstd(grid, axis=0)
        iterations = np.arange(1, max_iter + 1)
        axes[1].plot(iterations, mean_curve, color="#c44e52", linewidth=2.0)
        axes[1].fill_between(
            iterations,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="#c44e52",
            alpha=0.2,
        )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accepted samples per iteration")
    _save(fig, path)


def plot_confusion_matrices(
    before: np.ndarray,
    after: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, matrix, label in ((axes[0], before, "(a)"), (axes[1], after, "(b)")):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if value >= 0.5:
                    ax.text(
                        col,
                        row,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > 50 else "black",
                    )
        ax.set_title(label, loc="left", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalised (%)")
    _save(fig, path)


def plot_real_vs_generated(
    real_by_class: dict[int, np.ndarray],
    generated_by_class: dict[int, np.ndarray],
    class_names: list[str],
    path: Path,
) -> None:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    classes = sorted(set(real_by_class) & set(generated_by_class))
    if not classes:
        return
    real_x = np.vstack([real_by_class[c] for c in classes])
    gen_x = np.vstack([generated_by_class[c] for c in classes])
    real_y = np.concatenate([np.full(len(real_by_class[c]), c) for c in classes])
    gen_y = np.concatenate([np.full(len(generated_by_class[c]), c) for c in classes])
    features = np.vstack([real_x, gen_x])
    labels = np.concatenate([real_y, gen_y])
    projection = LinearDiscriminantAnalysis(n_components=2).fit_transform(features, labels)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    n_real = len(real_x)
    axes[0].scatter(
        projection[:n_real, 0], projection[:n_real, 1], s=10, alpha=0.5, label="Real"
    )
    axes[0].scatter(
        projection[n_real:, 0],
        projection[n_real:, 1],
        s=14,
        marker="^",
        alpha=0.6,
        label="Generated",
    )
    axes[0].set_xlabel("LD 1")
    axes[0].set_ylabel("LD 2")
    axes[0].legend(frameon=False, fontsize=8)

    sample_class = classes[-1]
    for curves, label, style in (
        (real_by_class[sample_class], "Real", "-"),
        (generated_by_class[sample_class], "Generated", "--"),
    ):
        mean_curve = np.mean(curves, axis=0)
        axes[1].plot(np.linspace(0, 1, len(mean_curve)), mean_curve, style, label=label)
    axes[1].set_xlabel("Normalised voltage")
    axes[1].set_ylabel("Normalised current")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, path)

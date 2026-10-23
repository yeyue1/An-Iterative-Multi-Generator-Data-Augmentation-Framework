from __future__ import annotations

"""Run-matched descriptive statistics.

The manuscript reports means, standard deviations, and the magnitude of the
gains relative to between-run variability; it does not attach p-values.  These
helpers therefore provide paired differences and bootstrap confidence
intervals rather than significance tests.
"""

import numpy as np


def paired_differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have the same shape, got {a.shape} and {b.shape}")
    return a - b


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lower, upper


def summarize_paired(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Summarise a - b over matched runs."""
    diffs = paired_differences(a, b)
    low, high = bootstrap_ci(diffs, seed=seed)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference": float(np.mean(diffs)),
        "std_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "n_runs": int(len(diffs)),
    }

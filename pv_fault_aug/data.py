from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]


def load_fault_dataset(config: DataConfig) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load a PV fault dataset from CSV or NPZ.

    CSV format:
        label,f0,f1,...,f99

    NPZ format:
        arrays named x and y. y may contain string labels or integer labels.
    """
    if config.data_path is None:
        raise FileNotFoundError(
            "No data path was provided. Pass --data-path to a CSV/NPZ file after "
            "the real I-V dataset is available."
        )

    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        x = np.asarray(data["x"], dtype=np.float32)
        y_raw = np.asarray(data["y"])
    elif path.suffix.lower() == ".csv":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CSV loading requires pandas. Install dependencies with "
                "`pip install -r requirements.txt`, or provide an NPZ file."
            ) from exc

        frame = pd.read_csv(path)
        if config.label_column not in frame.columns:
            raise ValueError(f"Missing label column: {config.label_column}")
        y_raw = frame[config.label_column].to_numpy()
        if config.feature_prefix:
            feature_cols = [c for c in frame.columns if c.startswith(config.feature_prefix)]
        else:
            feature_cols = [c for c in frame.columns if c != config.label_column]
        x = frame[feature_cols].to_numpy(dtype=np.float32)
    else:
        raise ValueError("Supported dataset formats are .csv and .npz")

    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D [samples, features], got shape {x.shape}")
    if x.shape[1] != config.feature_dim:
        raise ValueError(
            f"Expected {config.feature_dim} I-V points per sample, got {x.shape[1]}"
        )

    label_to_index = {name: idx for idx, name in enumerate(config.class_names)}
    if np.issubdtype(y_raw.dtype, np.integer):
        y = y_raw.astype(np.int64)
    else:
        missing = sorted(set(map(str, y_raw)) - set(label_to_index))
        if missing:
            raise ValueError(f"Unknown labels in dataset: {missing}")
        y = np.asarray([label_to_index[str(label)] for label in y_raw], dtype=np.int64)

    return x, y, label_to_index


def construct_scenario(
    x: np.ndarray,
    y: np.ndarray,
    scenario: str,
    config: DataConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    labels = np.unique(y)

    if scenario == "imbalanced":
        return x.copy(), y.copy()

    for label in labels:
        class_indices = np.where(y == label)[0]
        if scenario == "balanced":
            n_select = min(np.sum(y == current_label) for current_label in labels)
        elif scenario == "scarce":
            n_select = max(1, int(round(len(class_indices) * config.scarce_ratio)))
        else:
            raise ValueError("scenario must be balanced, imbalanced, or scarce")
        selected.extend(rng.choice(class_indices, size=n_select, replace=False).tolist())

    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    return x[selected_array].copy(), y[selected_array].copy()


def make_split(
    x: np.ndarray,
    y: np.ndarray,
    label_to_index: dict[str, int],
    config: DataConfig,
    seed: int,
) -> DatasetSplit:
    try:
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Dataset splitting requires scikit-learn. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x,
        y,
        train_size=config.train_ratio,
        random_state=seed,
        stratify=y,
    )
    relative_val = config.val_ratio / (config.val_ratio + config.test_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        train_size=relative_val,
        random_state=seed + 1,
        stratify=y_tmp,
    )
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return DatasetSplit(
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        label_to_index=label_to_index,
        index_to_label=index_to_label,
    )


def load_scenario_split(
    data_config: DataConfig,
    scenario: str,
    seed: int,
) -> DatasetSplit:
    x, y, label_to_index = load_fault_dataset(data_config)
    x_scenario, y_scenario = construct_scenario(x, y, scenario, data_config, seed)
    return make_split(x_scenario, y_scenario, label_to_index, data_config, seed)

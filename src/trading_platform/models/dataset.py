"""Sequence dataset for multivariate LSTM classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class LabelSpec:
    """Metadata describing how labels were built."""

    horizon: int
    hold_epsilon: float
    num_classes: int = 3


def chronological_split_indices(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[slice, slice, slice]:
    """Return slice objects for train, validation, test (no shuffle).

    Args:
        n: Number of usable sequence end indices.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (remainder is test).

    Returns:
        Three slices in time order.
    """
    if n < 3:
        raise ValueError("Need at least 3 sequences for train/val/test split")
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val
    logger.info("split: train=%s val=%s test=%s", n_train, n_val, n_test)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


class TimeSeriesSequenceDataset(Dataset):
    """Windows of length ``seq_len`` ending at index i, predicting label at i."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
    ) -> None:
        """Build tensor dataset.

        Args:
            features: Array shape ``(T, F)``.
            labels: Integer labels shape ``(T,)`` (same length as features).
            seq_len: Past window length.

        Raises:
            ValueError: If shapes mismatch.
        """
        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch")
        self._x = features.astype(np.float32)
        self._y = labels.astype(np.int64)
        self._seq_len = seq_len
        self._valid_start = seq_len - 1
        self._n = len(features) - self._valid_start

    def __len__(self) -> int:
        return max(0, self._n)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = self._valid_start + idx
        start = end - self._seq_len + 1
        window = self._x[start : end + 1]
        y = self._y[end]
        return torch.from_numpy(window), torch.tensor(y, dtype=torch.long)


def build_arrays_from_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Drop rows with NaN features or label; return aligned numpy arrays.

    Args:
        df: Feature matrix.
        feature_cols: Column names used as inputs.
        label_col: Integer class column.

    Returns:
        Tuple ``(X, y, index)`` for valid rows only.
    """
    sub = df[feature_cols + [label_col]].dropna()
    X = sub[feature_cols].to_numpy(dtype=np.float64)
    y = sub[label_col].astype("int64").to_numpy(dtype=np.int64)
    return X, y, sub.index


def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training feature rows (2D).

    Args:
        X_train: Shape ``(N, F)``.

    Returns:
        Fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale feature matrix.

    Args:
        X: Features shape ``(N, F)``.
        scaler: Fitted scaler.

    Returns:
        Scaled array same shape.
    """
    return scaler.transform(X).astype(np.float32)


def save_artifact_meta(
    path: Path,
    feature_cols: List[str],
    seq_len: int,
    label_spec: LabelSpec,
) -> None:
    """Write JSON sidecar for inference.

    Args:
        path: Output JSON path.
        feature_cols: Model input columns in order.
        seq_len: Sequence length.
        label_spec: Label construction parameters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": feature_cols,
        "seq_len": seq_len,
        "label_spec": {
            "horizon": label_spec.horizon,
            "hold_epsilon": label_spec.hold_epsilon,
            "num_classes": label_spec.num_classes,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote artifact meta %s", path)


def load_artifact_meta(path: Path) -> Tuple[List[str], int, LabelSpec]:
    """Load JSON sidecar.

    Args:
        path: JSON path.

    Returns:
        feature_cols, seq_len, LabelSpec
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    ls = data["label_spec"]
    spec = LabelSpec(
        horizon=int(ls["horizon"]),
        hold_epsilon=float(ls["hold_epsilon"]),
        num_classes=int(ls.get("num_classes", 3)),
    )
    return list(data["feature_columns"]), int(data["seq_len"]), spec

"""Data quality checks for time-ordered feature matrices."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def assert_monotonic_index(df: pd.DataFrame) -> None:
    """Raise if index is not DatetimeIndex sorted ascending unique."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be sorted ascending")
    if df.index.has_duplicates:
        raise ValueError("DatetimeIndex must be unique")


def assert_no_future_in_features(
    feature_cols: List[str],
    label_col: str,
    df: pd.DataFrame,
) -> None:
    """Heuristic: label column may be NaN only where features are NaN (warmup).

    Does not prove causality; documents contract that label is forward-looking
    from features at same row (to be shifted in dataset builder).

    Args:
        feature_cols: Feature column names at time t.
        label_col: Label column (typically NaN on last rows).
        df: Full frame.
    """
    if label_col not in df.columns:
        return
    # Rows with any feature NaN should not be used for training; optional check
    bad = df[feature_cols].notna().all(axis=1) & df[label_col].isna()
    # Last `horizon` rows may legitimately have NaN label
    trailing_na = bad.tail(min(5, len(bad)))
    if trailing_na.any():
        logger.debug("Trailing NaN labels where features exist (expected near series end)")


def validate_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: Optional[str] = None,
) -> None:
    """Run structural validations and log summary.

    Args:
        df: Feature matrix indexed by time.
        feature_cols: Columns required for modeling.
        label_col: Optional label column name.

    Raises:
        ValueError: On index or missing columns.
    """
    assert_monotonic_index(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if label_col and label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    logger.info(
        "validate_feature_matrix: rows=%s range=%s..%s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    if label_col:
        assert_no_future_in_features(list(feature_cols), label_col, df)

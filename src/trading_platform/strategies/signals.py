from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def apply_confidence_threshold(
    signal: pd.Series,
    confidence: pd.Series,
    min_confidence: float,
) -> pd.Series:
    out = signal.astype(float).copy()
    mask = confidence.astype(float) < min_confidence
    out.loc[mask] = 0.0
    logger.debug("Confidence filter removed %s bars", int(mask.sum()))
    return out


def signals_to_entries_exits(signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = signal.astype(float).fillna(0.0)
    prev = s.shift(1).fillna(0.0)
    entries = (s > 0) & (prev <= 0)
    exits = (s <= 0) & (prev > 0)
    exits = exits | ((s < 0) & (prev >= 0))
    return entries, exits

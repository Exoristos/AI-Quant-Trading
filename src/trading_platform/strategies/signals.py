"""Map model outputs to discrete entries/exits for the backtester."""

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
    """Zero out signals where confidence is below threshold.

    Args:
        signal: Values in ``{-1, 0, 1}`` (or float equivalents).
        confidence: Probabilities in ``[0, 1]``.
        min_confidence: Minimum confidence to keep non-zero signal.

    Returns:
        Adjusted signal series (HOLD where filtered).
    """
    out = signal.astype(float).copy()
    mask = confidence.astype(float) < min_confidence
    out.loc[mask] = 0.0
    logger.debug("Confidence filter removed %s bars", int(mask.sum()))
    return out


def signals_to_entries_exits(signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Convert signed signals into long-only entry/exit booleans.

    Entry when signal transitions to +1 from not +1; exit when transitions
    to non-+1 from +1. SELL (-1) forces exit on that bar.

    Args:
        signal: Discrete signal per bar.

    Returns:
        Tuple ``(entries, exits)`` boolean Series aligned to ``signal`` index.
    """
    s = signal.astype(float).fillna(0.0)
    prev = s.shift(1).fillna(0.0)
    entries = (s > 0) & (prev <= 0)
    exits = (s <= 0) & (prev > 0)
    exits = exits | ((s < 0) & (prev >= 0))
    return entries, exits

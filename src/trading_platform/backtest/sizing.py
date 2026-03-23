"""Position sizing utilities."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def clamp_position_pct(pct: float) -> float:
    """Clamp fraction-of-equity size to (0, 1].

    Args:
        pct: Desired fraction (e.g. 0.1 for 10%).

    Returns:
        Clamped positive fraction, max 1.0.
    """
    if pct <= 0:
        logger.warning("position_size_pct <= 0; using 1%%")
        return 0.01
    return min(1.0, float(pct))

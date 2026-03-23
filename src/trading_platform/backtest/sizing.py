from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def clamp_position_pct(pct: float) -> float:
    if pct <= 0:
        logger.warning("position_size_pct <= 0; using 1%%")
        return 0.01
    return min(1.0, float(pct))

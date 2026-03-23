"""Commission and slippage helpers."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

Side = Literal["buy", "sell"]


def slippage_fraction(bps: float) -> float:
    """Convert basis points to fractional slippage (per unit price).

    Args:
        bps: Basis points (1 bp = 0.01%).

    Returns:
        Slippage as fraction (e.g. 5 bps -> 0.0005).
    """
    return bps / 10_000.0


def apply_slippage(price: float, side: Side, bps: float) -> float:
    """Adjust price for a conservative slippage model (fixed bps per side).

    Args:
        price: Execution reference price.
        side: ``buy`` pays more; ``sell`` receives less.
        bps: Slippage in basis points.

    Returns:
        Adjusted execution price.
    """
    frac = slippage_fraction(bps)
    if side == "buy":
        adj = price * (1.0 + frac)
    else:
        adj = price * (1.0 - frac)
    logger.debug("slippage %s bps=%s price %s -> %s", side, bps, price, adj)
    return adj

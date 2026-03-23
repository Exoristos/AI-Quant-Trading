from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

Side = Literal["buy", "sell"]


def slippage_fraction(bps: float) -> float:
    return bps / 10_000.0


def apply_slippage(price: float, side: Side, bps: float) -> float:
    frac = slippage_fraction(bps)
    if side == "buy":
        adj = price * (1.0 + frac)
    else:
        adj = price * (1.0 - frac)
    logger.debug("slippage %s bps=%s price %s -> %s", side, bps, price, adj)
    return adj

"""BIST-oriented symbol lists for UI presets (not official index membership).

Official BIST 30 membership changes over time; using a fixed list without
historical reconstruction introduces **survivorship bias** in multi-name studies.
For single-ticker backtests, pick one symbol you intend to trade.
"""

from __future__ import annotations

from typing import Final, Tuple

# Liquid names often used in research; suffix ``.IS`` for EODHD / yfinance.
# Update from https://www.borsaistanbul.com/ or your data vendor when needed.
BIST_QUICK_PICK: Final[Tuple[str, ...]] = (
    "THYAO.IS",
    "GARAN.IS",
    "AKBNK.IS",
    "BIMAS.IS",
    "EREGL.IS",
    "ASELS.IS",
    "ISCTR.IS",
    "KCHOL.IS",
    "SAHOL.IS",
    "TUPRS.IS",
    "FROTO.IS",
    "SISE.IS",
    "TCELL.IS",
    "YKBNK.IS",
    "VAKBN.IS",
)

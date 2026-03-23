from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, TypeAlias

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

OHLCVFrame: TypeAlias = pd.DataFrame


class MarketProvider(ABC):
    @abstractmethod
    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        raise NotImplementedError


def normalize_ohlcv(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    rename_map = {
        "adj close": "adj_close",
        "adj_close": "adj_close",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    if "close" not in out.columns:
        raise ValueError("Normalized frame must contain a 'close' column.")
    idx = pd.to_datetime(out.index, utc=True).tz_convert(None)
    out.index = idx.normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if ticker is not None:
        out["ticker"] = ticker
    return out

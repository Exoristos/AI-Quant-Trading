from __future__ import annotations

import logging
from typing import List

import pandas as pd
import yfinance as yf

from trading_platform.data.providers.base import MarketProvider, normalize_ohlcv

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketProvider):
    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for sym in tickers:
            logger.info("yfinance: fetching %s from %s to %s", sym, start, end)
            t = yf.Ticker(sym)
            raw = t.history(start=start, end=end, auto_adjust=False)
            if raw.empty:
                logger.warning("yfinance: no rows for %s", sym)
                continue
            norm = normalize_ohlcv(raw, ticker=sym)
            frames.append(norm)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).sort_index()

"""US (and generic) equity OHLCV via yfinance."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
import yfinance as yf

from trading_platform.data.providers.base import MarketProvider, normalize_ohlcv

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketProvider):
    """Downloads OHLCV using yfinance (no API key)."""

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch one or more US-style tickers and concatenate.

        Args:
            tickers: e.g. ``["AAPL", "MSFT"]``.
            start: Inclusive start ``YYYY-MM-DD``.
            end: Inclusive end ``YYYY-MM-DD``.

        Returns:
            Panel DataFrame with columns OHLCV + ``ticker``.
        """
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

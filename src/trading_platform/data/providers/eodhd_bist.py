from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
import requests

from trading_platform.data.providers.base import MarketProvider, normalize_ohlcv

logger = logging.getLogger(__name__)

EODHD_EOD_URL = "https://eodhistoricaldata.com/api/eod/{symbol}"


class EodhdBistProvider(MarketProvider):
    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        if not self._api_key:
            logger.error("EODHD_API_KEY not set; cannot fetch BIST EOD data")
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for sym in tickers:
            url = EODHD_EOD_URL.format(symbol=sym)
            params = {
                "api_token": self._api_key,
                "from": start,
                "to": end,
                "fmt": "json",
                "period": "d",
            }
            logger.info("EODHD: fetching %s from %s to %s", sym, start, end)
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                logger.exception("EODHD request failed for %s: %s", sym, exc)
                continue
            if not data:
                logger.warning("EODHD: empty payload for %s", sym)
                continue
            df = pd.DataFrame(data)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep]
            norm = normalize_ohlcv(df, ticker=sym)
            frames.append(norm)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).sort_index()

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ema(series: pd.Series, span: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods if min_periods is not None else span
    return series.ewm(span=span, adjust=False, min_periods=mp).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": hist},
        index=close.index,
    )


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower}, index=close.index)


def add_all_indicators(ohlcv: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    df = ohlcv.copy()
    c = df[price_col]
    df["ema_12"] = ema(c, 12)
    df["ema_26"] = ema(c, 26)
    df["sma_50"] = sma(c, 50)
    df["rsi_14"] = rsi(c, 14)
    macd_df = macd(c)
    for col in macd_df.columns:
        df[col] = macd_df[col]
    bb = bollinger_bands(c)
    for col in bb.columns:
        df[col] = bb[col]
    logger.debug("Indicators added: %s", [x for x in df.columns if x not in ohlcv.columns])
    return df


def feature_columns_default() -> List[str]:
    return [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_12",
        "ema_26",
        "sma_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
    ]

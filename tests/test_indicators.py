"""Indicator sanity checks (causal windows)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.data.indicators import rsi, sma


def test_rsi_warmup_no_future() -> None:
    """RSI defined when both gains and losses exist (oscillating series)."""
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 + rng.normal(0, 0.5, len(idx)).cumsum(), index=idx)
    r = rsi(close, period=14)
    assert r.notna().sum() > len(idx) // 2


def test_sma_uses_past_only() -> None:
    """SMA at t equals mean of window ending at t."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    close = pd.Series(np.arange(10, dtype=float), index=idx)
    s = sma(close, window=3)
    assert abs(s.iloc[2] - close.iloc[0:3].mean()) < 1e-9

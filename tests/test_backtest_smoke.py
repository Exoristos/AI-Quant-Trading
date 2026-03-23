"""Backtest engine smoke test (requires vectorbt)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from trading_platform.backtest.engine import run_backtest


def test_run_backtest_minimal() -> None:
    """Minimal long round-trip."""
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    close = pd.Series(100.0 + pd.RangeIndex(len(idx)), index=idx, dtype=float)
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    entries.iloc[2] = True
    exits.iloc[10] = True
    _, trades, eq = run_backtest(
        close,
        entries,
        exits,
        initial_cash=10_000.0,
        commission=0.0,
        slippage_bps=0.0,
        position_size_pct=0.2,
        lag_signals=True,
    )
    assert len(eq) == len(close)
    assert eq.iloc[-1] > 0

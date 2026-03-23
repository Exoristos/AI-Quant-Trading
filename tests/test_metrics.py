"""Tests for annualized risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.metrics.performance import (
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate_from_trades,
)


def test_sharpe_positive_on_upward_trend() -> None:
    """Sharpe should be positive for steady positive daily returns."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    equity = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    rets = equity.pct_change().dropna()
    s = sharpe_ratio(rets, risk_free_daily=0.0)
    assert s > 0.5


def test_max_drawdown_simple() -> None:
    """Known peak-to-trough drawdown."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    equity = pd.Series([100.0, 110.0, 90.0, 95.0, 100.0], index=idx)
    mdd = max_drawdown(equity)
    assert mdd < -0.15


def test_sortino_with_only_downside() -> None:
    """Sortino uses downside deviation; mixed series should be finite."""
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.0005, 0.01, 200))
    s = sortino_ratio(rets, risk_free_daily=0.0)
    assert not np.isnan(s)


def test_win_rate() -> None:
    """Win rate counts positive PnL trades."""
    pnl = pd.Series([1.0, -2.0, 3.0, 0.0])
    wr = win_rate_from_trades(pnl)
    assert abs(wr - 0.5) < 1e-9

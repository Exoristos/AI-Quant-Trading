from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class PerformanceReport:
    cumulative_pnl: float
    cumulative_return: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    n_trades: int


def equity_to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_daily: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    excess = daily_returns - risk_free_daily
    mu = excess.mean()
    sig = excess.std(ddof=1)
    if sig == 0 or np.isnan(sig):
        return float("nan")
    return float((mu / sig) * np.sqrt(trading_days))


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_daily: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    excess = daily_returns - risk_free_daily
    downside = excess.copy()
    downside[downside > 0] = 0.0
    ddev = downside.std(ddof=1)
    if ddev == 0 or np.isnan(ddev):
        return float("nan")
    return float((excess.mean() / ddev) * np.sqrt(trading_days))


def win_rate_from_trades(trade_pnl: pd.Series) -> float:
    if trade_pnl.empty:
        return 0.0
    return float((trade_pnl > 0).mean())


def compute_performance(
    equity: pd.Series,
    trade_pnl: Optional[pd.Series],
    initial_cash: float,
    risk_free_daily: float = 0.0,
) -> PerformanceReport:
    eq = equity.dropna()
    final = float(eq.iloc[-1])
    cum_pnl = final - initial_cash
    cum_ret = final / initial_cash - 1.0 if initial_cash else float("nan")
    rets = equity_to_returns(eq)
    sharpe = sharpe_ratio(rets, risk_free_daily=risk_free_daily)
    sortino = sortino_ratio(rets, risk_free_daily=risk_free_daily)
    mdd = max_drawdown(eq)
    wr = win_rate_from_trades(trade_pnl) if trade_pnl is not None else 0.0
    n_trades = int(trade_pnl.shape[0]) if trade_pnl is not None else 0
    logger.info(
        "Performance: pnl=%.2f cum_ret=%.4f sharpe=%.3f sortino=%.3f mdd=%.4f win_rate=%.3f trades=%s",
        cum_pnl,
        cum_ret,
        sharpe,
        sortino,
        mdd,
        wr,
        n_trades,
    )
    return PerformanceReport(
        cumulative_pnl=cum_pnl,
        cumulative_return=cum_ret,
        win_rate=wr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=mdd,
        n_trades=n_trades,
    )

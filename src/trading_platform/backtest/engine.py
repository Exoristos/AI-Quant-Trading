from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import pandas as pd
import vectorbt as vbt

from trading_platform.backtest.costs import slippage_fraction
from trading_platform.backtest.sizing import clamp_position_pct

logger = logging.getLogger(__name__)


def _shift_signals_no_lookahead(
    entries: pd.Series,
    exits: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    e = entries.shift(1).astype("boolean").fillna(False).astype(bool)
    x = exits.shift(1).astype("boolean").fillna(False).astype(bool)
    return e, x


def run_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage_bps: float = 5.0,
    position_size_pct: float = 0.10,
    lag_signals: bool = True,
) -> Tuple[Any, pd.DataFrame, pd.Series]:
    close = close.sort_index().astype(float)
    entries = entries.reindex(close.index).astype("boolean").fillna(False).astype(bool)
    exits = exits.reindex(close.index).astype("boolean").fillna(False).astype(bool)
    if lag_signals:
        entries, exits = _shift_signals_no_lookahead(entries, exits)

    pct = clamp_position_pct(position_size_pct)
    slip = slippage_fraction(slippage_bps)

    try:
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            direction="longonly",
            init_cash=float(initial_cash),
            fees=float(commission),
            slippage=float(slip),
            size=float(pct),
            size_type="targetpercent",
            freq="1D",
        )
    except Exception as exc:
        logger.warning("targetpercent failed (%s); falling back to percent-of-cash sizing", exc)
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            direction="longonly",
            init_cash=float(initial_cash),
            fees=float(commission),
            slippage=float(slip),
            size=float(pct),
            size_type="percent",
            freq="1D",
        )

    equity = portfolio.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]

    trades_df = _trades_to_dataframe(portfolio)
    trade_pnl = trades_df["pnl"] if "pnl" in trades_df.columns else pd.Series(dtype=float)
    logger.info("Backtest done: trades=%s final_value=%s", len(trades_df), float(equity.iloc[-1]))
    return portfolio, trades_df, equity


def _trades_to_dataframe(portfolio: Any) -> pd.DataFrame:
    try:
        tr = portfolio.trades.records_readable
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not read trades: %s", exc)
        return pd.DataFrame()
    if tr is None or len(tr) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(tr)
    rename_map = {
        "Entry Index": "entry_idx",
        "Exit Index": "exit_idx",
        "Entry Time": "entry_time",
        "Exit Time": "exit_time",
        "Entry Price": "entry_price",
        "Exit Price": "exit_price",
        "Size": "size",
        "PnL": "pnl",
        "Return": "return_pct",
    }
    cols = {c: rename_map[c] for c in df.columns if c in rename_map}
    df = df.rename(columns=cols)
    return df

"""VectorBT backtesting engine and trade log assembly."""

from __future__ import annotations

from typing import Any

__all__ = ["run_backtest"]


def __getattr__(name: str) -> Any:
    """Defer ``vectorbt`` import until backtest is used."""
    if name == "run_backtest":
        from trading_platform.backtest.engine import run_backtest

        return run_backtest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

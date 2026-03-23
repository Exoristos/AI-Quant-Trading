from __future__ import annotations

from typing import Any

__all__ = ["run_backtest"]


def __getattr__(name: str) -> Any:
    if name == "run_backtest":
        from trading_platform.backtest.engine import run_backtest

        return run_backtest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

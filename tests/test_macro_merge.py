"""Macro merge_asof alignment."""

from __future__ import annotations

import pandas as pd

from trading_platform.data.macro import merge_macro_asof


def test_merge_asof_backward() -> None:
    """Equity bar receives last macro on or before that date."""
    eq_idx = pd.to_datetime(["2020-01-10", "2020-01-15", "2020-01-20"])
    macro = pd.DataFrame({"cpi": [100.0, 101.0]}, index=pd.to_datetime(["2020-01-05", "2020-01-12"]))
    out = merge_macro_asof(eq_idx, macro, tolerance_days=60)
    assert out.loc[eq_idx[0], "cpi"] == 100.0
    assert out.loc[eq_idx[1], "cpi"] == 101.0

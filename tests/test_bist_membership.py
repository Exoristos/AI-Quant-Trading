"""BIST membership table helpers."""

from __future__ import annotations

import io

import pandas as pd

from trading_platform.data.bist_membership import tickers_as_of


def test_tickers_as_of_selects_latest_bucket() -> None:
    """Tickers match the most recent rebalance on or before as_of."""
    csv = io.StringIO(
        "rebalance_date,ticker\n"
        "2020-01-01,AAA.IS\n"
        "2020-01-01,BBB.IS\n"
        "2022-06-01,AAA.IS\n"
        "2022-06-01,CCC.IS\n"
    )
    df = pd.read_csv(csv, parse_dates=["rebalance_date"])
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df = df.sort_values("rebalance_date").reset_index(drop=True)
    t2021 = tickers_as_of("2021-01-01", df)
    assert set(t2021) == {"AAA.IS", "BBB.IS"}
    t2023 = tickers_as_of("2023-01-01", df)
    assert set(t2023) == {"AAA.IS", "CCC.IS"}

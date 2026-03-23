"""Load BIST index membership tables (user-maintained; reduces survivorship bias when accurate).

CSV format (long form)::

    rebalance_date,ticker
    2023-01-02,THYAO.IS
    2023-01-02,GARAN.IS
    2024-06-03,THYAO.IS

Each ``rebalance_date`` lists all index constituents valid from that date until the
next rebalance row (by date). Tickers on the same date form one basket.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_membership_csv(path: str | Path) -> pd.DataFrame:
    """Load membership table from CSV.

    Args:
        path: CSV path with columns ``rebalance_date`` and ``ticker``.

    Returns:
        DataFrame sorted by ``rebalance_date`` with normalized tickers.
    """
    p = Path(path)
    df = pd.read_csv(p, parse_dates=["rebalance_date"])
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], utc=True).dt.tz_convert(None).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values("rebalance_date").reset_index(drop=True)
    logger.info("Loaded membership %s rows=%s rebalances=%s", p.name, len(df), df["rebalance_date"].nunique())
    return df


def tickers_as_of(as_of: Union[str, pd.Timestamp], membership: pd.DataFrame) -> List[str]:
    """Return constituent tickers active on ``as_of`` (last rebalance on or before).

    Args:
        as_of: Calendar date or timestamp.
        membership: Output of :func:`load_membership_csv`.

    Returns:
        Unique tickers for the applicable rebalance bucket; empty if none apply.
    """
    d = pd.Timestamp(as_of).normalize()
    sub = membership[membership["rebalance_date"] <= d]
    if sub.empty:
        logger.warning("No membership rows on or before %s", d.date())
        return []
    last = sub["rebalance_date"].max()
    tickers = sub.loc[sub["rebalance_date"] == last, "ticker"].drop_duplicates().tolist()
    logger.debug("as_of=%s -> rebalance=%s n=%s", d.date(), last.date(), len(tickers))
    return tickers

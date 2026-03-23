# CSV: rebalance_date,ticker — long form; tickers on same date are one basket until next date.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_membership_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, parse_dates=["rebalance_date"])
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], utc=True).dt.tz_convert(None).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values("rebalance_date").reset_index(drop=True)
    logger.info("Loaded membership %s rows=%s rebalances=%s", p.name, len(df), df["rebalance_date"].nunique())
    return df


def tickers_as_of(as_of: Union[str, pd.Timestamp], membership: pd.DataFrame) -> List[str]:
    d = pd.Timestamp(as_of).normalize()
    sub = membership[membership["rebalance_date"] <= d]
    if sub.empty:
        logger.warning("No membership rows on or before %s", d.date())
        return []
    last = sub["rebalance_date"].max()
    tickers = sub.loc[sub["rebalance_date"] == last, "ticker"].drop_duplicates().tolist()
    logger.debug("as_of=%s -> rebalance=%s n=%s", d.date(), last.date(), len(tickers))
    return tickers

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def load_macro_from_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, parse_dates=[date_col])
    df = df.set_index(date_col)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info("Loaded macro CSV %s rows=%s cols=%s", p.name, len(df), list(df.columns))
    return df


def fetch_fred_series(
    series_id: str,
    api_key: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    params: Dict[str, str] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    logger.info("FRED: fetching series %s", series_id)
    resp = requests.get(FRED_OBS_URL, params=params, timeout=60)
    resp.raise_for_status()
    ctype = (resp.headers.get("content-type") or "").lower()
    body = resp.text or ""
    head = body.lstrip()[:1]
    if "html" in ctype or head not in ("{", "["):
        logger.warning(
            "FRED non-JSON response (ctype=%s, len=%s); skipping series %s",
            ctype,
            len(body),
            series_id,
        )
        return pd.DataFrame(columns=[series_id])
    try:
        payload = resp.json()
    except ValueError as exc:
        logger.warning("FRED JSON parse failed for %s: %s", series_id, exc)
        return pd.DataFrame(columns=[series_id])
    obs = payload.get("observations", [])
    rows = []
    for row in obs:
        val = row.get("value")
        if val in (".", "", None):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        rows.append({"date": pd.to_datetime(row["date"]), series_id: v})
    if not rows:
        return pd.DataFrame(columns=[series_id])
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    df = df.sort_index()
    return df


def merge_macro_asof(
    equity_index: pd.DatetimeIndex,
    macro: pd.DataFrame,
    tolerance_days: int = 31,
) -> pd.DataFrame:
    if macro.empty:
        return pd.DataFrame(index=equity_index)
    macro = macro.sort_index()
    macro = macro[~macro.index.duplicated(keep="last")]
    idx_norm = pd.to_datetime(equity_index, utc=True).tz_convert(None).normalize()
    left = pd.DataFrame({"date": idx_norm})
    right = macro.reset_index()
    first_col = right.columns[0]
    right = right.rename(columns={first_col: "date"})
    right["date"] = pd.to_datetime(right["date"], utc=True).dt.tz_convert(None).dt.normalize()
    value_cols = [c for c in right.columns if c != "date"]
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    merged = merged.set_index("date")
    merged = merged.reindex(idx_norm)
    out = merged[value_cols].copy()
    m_index = macro.index.values
    for col in value_cols:
        staleness: List[Optional[int]] = []
        for d in idx_norm:
            if pd.isna(out.loc[d, col]):
                staleness.append(None)
                continue
            pos = macro.index.searchsorted(d, side="right") - 1
            if pos < 0:
                staleness.append(None)
            else:
                last_m = pd.Timestamp(macro.index[pos]).normalize()
                staleness.append(int((d - last_m).days))
        out[f"{col}_staleness_days"] = staleness
    logger.info(
        "merge_asof macro: equity=%s non_null=%s",
        len(equity_index),
        {c: int(out[c].notna().sum()) for c in value_cols},
    )
    return out


def build_default_macro_panel(
    start: str,
    end: str,
    fred_api_key: Optional[str],
    extra_series: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not fred_api_key:
        logger.warning("FRED_API_KEY not set; macro panel empty")
        return pd.DataFrame()
    ids = ["CPIAUCSL", "DEXUSEU"]
    if extra_series:
        ids = list(dict.fromkeys(ids + extra_series))
    frames = []
    for sid in ids:
        try:
            f = fetch_fred_series(sid, fred_api_key, start=start, end=end)
            if not f.empty:
                frames.append(f)
        except requests.RequestException as exc:
            logger.warning("FRED series %s failed: %s", sid, exc)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.sort_index()
    out = out.ffill()
    return out

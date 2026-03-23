"""TCMB EVDS macro series (USD/TRY, rates, etc.) via public API."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# EVDS 3 platform: https://evds3.tcmb.gov.tr — legacy REST path kept for compatibility.
EVDS_DEFAULT_BASE_URL = "https://evds3.tcmb.gov.tr/service/evds/"


def _iso_to_evds_date(iso: str) -> str:
    """Convert ``YYYY-MM-DD`` to EVDS ``DD-MM-YYYY``."""
    ts = pd.Timestamp(iso)
    return ts.strftime("%d-%m-%Y")


def fetch_evds_series(
    series_codes: List[str],
    start: str,
    end: str,
    api_key: str,
    timeout: int = 120,
    base_url: Optional[str] = None,
) -> pd.DataFrame:
    """Download one or more EVDS series and return a tidy daily DataFrame.

    Args:
        series_codes: EVDS codes, e.g. ``["TP.DK.USD.A.YTL"]``. Multiple codes
            are joined with ``-`` per TCMB convention.
        start: Inclusive start ``YYYY-MM-DD``.
        end: Inclusive end ``YYYY-MM-DD``.
        api_key: EVDS API key from https://evds3.tcmb.gov.tr (profile / API).
        timeout: HTTP timeout seconds.
        base_url: REST base (trailing slash optional). Defaults to EVDS 3
            ``.../service/evds/``. Set env ``EVDS_BASE_URL`` via :class:`AppSettings`
            if TCMB documents a different path.

    Returns:
        DataFrame indexed by normalized date; one float column per series
        (sanitized snake_case names). Empty if no rows.

    Raises:
        requests.HTTPError: On non-success HTTP status.

    Note:
        Invalid keys or wrong ``base_url`` may yield HTTP 200 with an HTML page;
        those cases return an empty DataFrame and log a warning.
    """
    if not api_key:
        logger.error("EVDS_API_KEY missing")
        return pd.DataFrame()
    codes = "-".join(c.strip() for c in series_codes if c.strip())
    if not codes:
        return pd.DataFrame()
    endpoint = (base_url or EVDS_DEFAULT_BASE_URL).strip().rstrip("/") + "/"
    params = {
        "series": codes,
        "startDate": _iso_to_evds_date(start),
        "endDate": _iso_to_evds_date(end),
        "type": "json",
        "formulas": "0",
        "frequency": "1",
        "key": api_key,
    }
    logger.info("EVDS request base=%s series=%s", endpoint, codes)
    resp = requests.get(endpoint, params=params, timeout=timeout)
    resp.raise_for_status()
    ctype = (resp.headers.get("content-type") or "").lower()
    body = resp.text or ""
    head = body.lstrip()[:1]
    if "html" in ctype or head not in ("{", "["):
        logger.warning(
            "EVDS returned non-JSON (ctype=%s, body_len=%s); skipping macro merge",
            ctype,
            len(body),
        )
        return pd.DataFrame()
    try:
        payload = resp.json()
    except ValueError as exc:
        logger.warning("EVDS JSON parse failed: %s", exc)
        return pd.DataFrame()
    items = payload.get("items")
    if not items:
        logger.warning("EVDS empty items for series=%s", codes)
        return pd.DataFrame()
    rows = []
    for row in items:
        if not isinstance(row, dict):
            continue
        date_key = next((k for k in row if str(k).lower() in ("tarih", "date")), None)
        if date_key is None:
            continue
        raw_d = row[date_key]
        try:
            dt = pd.to_datetime(raw_d, dayfirst=True)
        except (ValueError, TypeError):
            continue
        rec = {"date": dt.normalize()}
        for k, v in row.items():
            if k == date_key or str(k).upper() == "UNIXTIME":
                continue
            if isinstance(v, dict):
                continue
            try:
                val = float(str(v).replace(",", "."))
            except (TypeError, ValueError):
                continue
            col = _sanitize_col(k)
            rec[col] = val
        if len(rec) > 1:
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info("EVDS rows=%s cols=%s", len(df), list(df.columns))
    return df


def _sanitize_col(name: str) -> str:
    """Make a safe column name from EVDS field keys."""
    s = str(name).strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = s.strip("_").lower()
    return s or "value"


def fetch_evds_to_frame(
    series_codes: List[str],
    start: str,
    end: str,
    api_key: Optional[str],
    base_url: Optional[str] = None,
) -> pd.DataFrame:
    """Wrapper that returns empty DataFrame when ``api_key`` is missing."""
    if not api_key:
        return pd.DataFrame()
    return fetch_evds_series(series_codes, start, end, api_key, base_url=base_url)

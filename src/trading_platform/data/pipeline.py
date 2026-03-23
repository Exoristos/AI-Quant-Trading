from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

import pandas as pd

from trading_platform.config.settings import AppSettings
from trading_platform.data.indicators import add_all_indicators, feature_columns_default
from trading_platform.data.macro import build_default_macro_panel, merge_macro_asof
from trading_platform.data.providers import EodhdBistProvider, YFinanceProvider
from trading_platform.data.validation import validate_feature_matrix

logger = logging.getLogger(__name__)

MarketChoice = Literal["us", "bist"]


def fetch_ohlcv_panel(
    tickers: List[str],
    start: str,
    end: str,
    market: MarketChoice,
    settings: Optional[AppSettings] = None,
) -> pd.DataFrame:
    settings = settings or AppSettings()
    if not tickers:
        raise ValueError("tickers must be non-empty")
    sym = tickers[0]
    if market == "us":
        prov = YFinanceProvider()
        df = prov.fetch([sym], start, end)
    else:
        prov = EodhdBistProvider(settings.eodhd_api_key)
        df = prov.fetch([sym], start, end)
        if df.empty:
            logger.warning("BIST EODHD empty; falling back to yfinance for %s", sym)
            df = YFinanceProvider().fetch([sym], start, end)
    if df.empty:
        return df
    if "ticker" in df.columns:
        df = df[df["ticker"] == sym]
    return df


def forward_return_label(
    close: pd.Series,
    horizon: int = 1,
    hold_epsilon: float = 0.002,
) -> pd.Series:
    """3-class labels from forward return: 0 short, 1 flat, 2 long (within ±hold_epsilon → flat)."""
    fwd = close.shift(-horizon) / close - 1.0
    out = pd.Series(pd.NA, index=close.index, dtype="Int64")
    out.loc[fwd > hold_epsilon] = 2
    out.loc[fwd < -hold_epsilon] = 0
    mask = fwd.notna() & (fwd.abs() <= hold_epsilon)
    out.loc[mask] = 1
    return out


def build_feature_matrix(
    tickers: List[str],
    start: str,
    end: str,
    market: MarketChoice,
    horizon: int = 1,
    hold_epsilon: float = 0.002,
    settings: Optional[AppSettings] = None,
    macro_csv_path: Optional[str] = None,
    use_fred_macro: bool = False,
    use_evds_macro: bool = False,
    evds_series_codes: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    settings = settings or AppSettings()
    ohlcv = fetch_ohlcv_panel(tickers, start, end, market, settings)
    if ohlcv.empty:
        logger.error("No OHLCV returned for %s", tickers[0])
        return pd.DataFrame(), []
    feats = add_all_indicators(ohlcv)
    idx = feats.index
    macro_parts = []
    if macro_csv_path:
        from trading_platform.data.macro import load_macro_from_csv

        m = load_macro_from_csv(macro_csv_path)
        aligned = merge_macro_asof(idx, m)
        macro_parts.append(aligned)
    if use_fred_macro and settings.fred_api_key:
        m2 = build_default_macro_panel(start, end, settings.fred_api_key)
        if not m2.empty:
            macro_parts.append(merge_macro_asof(idx, m2))
    if use_evds_macro and settings.evds_api_key and evds_series_codes:
        from trading_platform.data.evds import fetch_evds_to_frame

        codes = [c.strip() for c in evds_series_codes.split(",") if c.strip()]
        if codes:
            ev = fetch_evds_to_frame(
                codes,
                start,
                end,
                settings.evds_api_key,
                base_url=settings.evds_base_url,
            )

            if not ev.empty:
                macro_parts.append(merge_macro_asof(idx, ev))
    if macro_parts:
        macro_df = pd.concat(macro_parts, axis=1)
        macro_df = macro_df.loc[:, ~macro_df.columns.duplicated()]
        feats = feats.join(macro_df, how="left")
    feats["y_class"] = forward_return_label(feats["close"], horizon=horizon, hold_epsilon=hold_epsilon)
    base_feats = feature_columns_default()
    macro_extra = [c for c in feats.columns if c not in base_feats + ["y_class", "ticker"] and not str(c).endswith("_staleness_days")]
    feature_cols = base_feats + macro_extra
    feature_cols = [c for c in feature_cols if c in feats.columns]
    validate_feature_matrix(feats, feature_cols, label_col="y_class")
    return feats, feature_cols

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from trading_platform.backtest.engine import run_backtest
from trading_platform.config.settings import AppSettings, BacktestSettings
from trading_platform.data.pipeline import MarketChoice, build_feature_matrix
from trading_platform.metrics.performance import compute_performance
from trading_platform.models.dataset import LabelSpec
from trading_platform.models.inference import predict_signals
from trading_platform.models.train import train_lstm_classifier
from trading_platform.strategies.signals import apply_confidence_threshold, signals_to_entries_exits

logger = logging.getLogger(__name__)


def _safe_artifact_name(ticker: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", ticker.strip())


def scan_equities(
    tickers: List[str],
    start: str,
    end: str,
    market: MarketChoice,
    settings: AppSettings,
    artifacts_root: Path,
    *,
    label_horizon: int = 1,
    hold_epsilon: float = 0.002,
    macro_csv_path: Optional[str] = None,
    use_fred_macro: bool = False,
    use_evds_macro: bool = False,
    evds_series_codes: Optional[str] = None,
    seq_len: int = 20,
    epochs: int = 10,
    conf_threshold: float = 0.35,
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage_bps: float = 5.0,
    position_size_pct: float = 0.10,
    risk_free_daily: float = 0.0,
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    if max_symbols is not None:
        tickers = tickers[: int(max_symbols)]
    rows: List[dict] = []
    bt = BacktestSettings()
    rf = risk_free_daily if risk_free_daily is not None else bt.risk_free_daily

    for sym in tickers:
        row: dict = {"ticker": sym, "status": "ok"}
        try:
            feats, feature_cols = build_feature_matrix(
                [sym],
                start,
                end,
                market=market,
                horizon=label_horizon,
                hold_epsilon=hold_epsilon,
                settings=settings,
                macro_csv_path=macro_csv_path,
                use_fred_macro=use_fred_macro,
                use_evds_macro=use_evds_macro,
                evds_series_codes=evds_series_codes,
            )
            if feats.empty or not feature_cols:
                row["status"] = "no_data"
                rows.append(row)
                continue
            art_dir = Path(artifacts_root) / _safe_artifact_name(sym)
            train_lstm_classifier(
                feats,
                feature_cols,
                "y_class",
                seq_len=seq_len,
                label_spec=LabelSpec(horizon=label_horizon, hold_epsilon=hold_epsilon),
                epochs=epochs,
                artifacts_dir=art_dir,
            )
            preds = predict_signals(feats, art_dir)
            sig = apply_confidence_threshold(
                preds["signal"].fillna(0.0),
                preds["confidence"].fillna(0.0),
                min_confidence=float(conf_threshold),
            )
            entries, exits = signals_to_entries_exits(sig)
            close = feats["close"].astype(float)
            _, trades_df, equity = run_backtest(
                close,
                entries,
                exits,
                initial_cash=float(initial_cash),
                commission=float(commission),
                slippage_bps=float(slippage_bps),
                position_size_pct=float(position_size_pct),
            )
            trade_pnl = trades_df["pnl"] if "pnl" in trades_df.columns else pd.Series(dtype=float)
            rep = compute_performance(
                equity,
                trade_pnl,
                initial_cash=float(initial_cash),
                risk_free_daily=float(rf),
            )
            row.update(
                {
                    "cumulative_return": rep.cumulative_return,
                    "sharpe_ratio": rep.sharpe_ratio,
                    "max_drawdown": rep.max_drawdown,
                    "win_rate": rep.win_rate,
                    "n_trades": rep.n_trades,
                }
            )
        except Exception as exc:
            logger.exception("Scan failed for %s", sym)
            row["status"] = f"error: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)

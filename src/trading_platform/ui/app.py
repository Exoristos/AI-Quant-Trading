"""Streamlit dashboard: data, LSTM signals, backtest, and risk metrics."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_platform.backtest.engine import run_backtest
from trading_platform.config.settings import AppSettings, BacktestSettings
from trading_platform.data.bist_membership import load_membership_csv, tickers_as_of
from trading_platform.data.bist_universe import BIST_QUICK_PICK
from trading_platform.data.pipeline import build_feature_matrix
from trading_platform.data.scan import scan_equities
from trading_platform.logging_config import setup_logging
from trading_platform.metrics.performance import compute_performance, max_drawdown
from trading_platform.models.dataset import LabelSpec
from trading_platform.models.inference import predict_signals
from trading_platform.models.train import train_lstm_classifier
from trading_platform.models.walk_forward import walk_forward_lstm_metrics
from trading_platform.strategies.signals import apply_confidence_threshold, signals_to_entries_exits

logger = logging.getLogger(__name__)


def _ensure_bist_suffix(ticker: str) -> str:
    """Append ``.IS`` for Istanbul symbols when missing."""
    t = ticker.strip().upper()
    if not t.endswith(".IS"):
        return f"{t}.IS"
    return t


def _resolve_macro_path(raw: str) -> str | None:
    """Return usable macro CSV path or None."""
    s = raw.strip()
    if not s:
        return None
    p = Path(s)
    if p.is_file():
        return str(p.resolve())
    return None


def main() -> None:
    """Streamlit entrypoint."""
    setup_logging()
    st.set_page_config(page_title="LSTM Trading Platform", layout="wide")
    st.title("LSTM trading and backtesting")

    settings = AppSettings()
    bt_defaults = BacktestSettings()

    with st.sidebar:
        st.header("Parameters")
        run_mode = st.selectbox(
            "Run mode",
            (
                "Single symbol",
                "Scan: membership CSV (BIST)",
                "Scan: US tickers (comma list)",
            ),
        )
        market = st.selectbox("Market", ["us", "bist"], index=0)
        if run_mode == "Single symbol":
            if market == "bist":
                bist_mode = st.radio("BIST symbol", ("Quick pick", "Custom"), horizontal=True)
                if bist_mode == "Quick pick":
                    ticker = st.selectbox("Symbol", list(BIST_QUICK_PICK), index=0)
                else:
                    ticker = st.text_input("Ticker (e.g. THYAO or THYAO.IS)", value="THYAO.IS")
                ticker = _ensure_bist_suffix(ticker)
            else:
                ticker = st.text_input("Ticker", value="AAPL")
        else:
            ticker = "SCAN"
        end_d = st.date_input("End date", value=date.today())
        start_d = st.date_input(
            "Start date",
            value=end_d - timedelta(days=365 * 3),
        )
        membership_csv = st.text_input(
            "BIST membership CSV (scan / template)",
            value="examples/bist30_membership_template.csv",
            help="rebalance_date,ticker format — see examples/",
        )
        us_scan_tickers = st.text_input("US scan tickers (comma)", "AAPL,MSFT,GOOG")
        max_scan = st.number_input("Max symbols per scan", min_value=1, max_value=50, value=5)
        initial_cash = st.number_input("Initial capital", min_value=1000.0, value=bt_defaults.initial_cash)
        commission = st.number_input("Commission (fraction)", min_value=0.0, value=float(bt_defaults.commission), format="%.4f")
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=float(bt_defaults.slippage_bps))
        conf_threshold = st.slider("LSTM confidence threshold", 0.0, 1.0, 0.35, 0.01)
        position_pct = st.slider("Position size (% of portfolio)", 1, 100, 10) / 100.0
        seq_len = st.number_input("Sequence length", min_value=5, max_value=120, value=20)
        epochs = st.number_input("Training epochs", min_value=1, max_value=200, value=15)
        scan_epochs = st.number_input("Scan: epochs per symbol", min_value=1, max_value=100, value=8)
        use_fred = st.checkbox("Use FRED macro (needs FRED_API_KEY)", value=False)
        use_evds = st.checkbox("Use EVDS macro (needs EVDS_API_KEY)", value=False)
        evds_series = st.text_input("EVDS series codes (comma)", "TP.DK.USD.A.YTL")
        macro_csv = st.text_input(
            "Macro CSV path (optional; columns + date)",
            value="",
            help="Yerel CSV; tarih sütunu: date",
        )
        label_horizon = st.number_input("Label horizon (bars)", min_value=1, max_value=20, value=1)
        hold_epsilon = st.number_input("HOLD band ε (abs. forward return)", min_value=0.0001, value=0.002, format="%.4f")
        train_new = st.checkbox("Train new model on this run", value=True)
        run_walk_forward = st.checkbox("Walk-forward analysis (single mode, after run)", value=False)
        wf_splits = st.number_input("Walk-forward splits", min_value=2, max_value=8, value=4)
        wf_epochs = st.number_input("Walk-forward epochs / fold", min_value=1, max_value=30, value=5)
        artifacts_path = st.text_input("Artifacts directory", value="artifacts")
        scan_artifacts_root = st.text_input("Scan artifacts root", value="artifacts/scan")
        run_btn = st.button("Run pipeline")

    if not run_btn:
        st.info("Configure the sidebar and click **Run pipeline**.")
        return

    start_s = start_d.isoformat()
    end_s = end_d.isoformat()
    macro_csv_path = _resolve_macro_path(macro_csv)
    if macro_csv.strip() and macro_csv_path is None:
        st.warning(f"Macro CSV not found (skipping): {macro_csv.strip()}")

    evds_codes = evds_series.strip() if use_evds else None

    if run_mode == "Scan: membership CSV (BIST)":
        mp = Path(membership_csv.strip())
        if not mp.is_file():
            st.error(f"Membership file not found: {mp}")
            return
        tickers = tickers_as_of(pd.Timestamp(end_d), load_membership_csv(mp))
        if not tickers:
            st.error("No tickers for this end date; check membership dates.")
            return
        st.info(f"Scanning {len(tickers)} names (capped at {int(max_scan)}).")
        with st.spinner("Scanning universe (may take several minutes)..."):
            table = scan_equities(
                tickers,
                start_s,
                end_s,
                "bist",
                settings,
                Path(scan_artifacts_root),
                label_horizon=int(label_horizon),
                hold_epsilon=float(hold_epsilon),
                macro_csv_path=macro_csv_path,
                use_fred_macro=use_fred,
                use_evds_macro=bool(use_evds),
                evds_series_codes=evds_codes,
                seq_len=int(seq_len),
                epochs=int(scan_epochs),
                conf_threshold=float(conf_threshold),
                initial_cash=float(initial_cash),
                commission=float(commission),
                slippage_bps=float(slippage_bps),
                position_size_pct=float(position_pct),
                risk_free_daily=float(bt_defaults.risk_free_daily),
                max_symbols=int(max_scan),
            )
        st.subheader("Scan results")
        st.dataframe(table, use_container_width=True)
        return

    if run_mode == "Scan: US tickers (comma list)":
        tickers = [x.strip().upper() for x in us_scan_tickers.split(",") if x.strip()]
        if not tickers:
            st.error("Enter at least one US ticker.")
            return
        st.info(f"Scanning {len(tickers)} US names (capped at {int(max_scan)}).")
        with st.spinner("Scanning US tickers..."):
            table = scan_equities(
                tickers,
                start_s,
                end_s,
                "us",
                settings,
                Path(scan_artifacts_root),
                label_horizon=int(label_horizon),
                hold_epsilon=float(hold_epsilon),
                macro_csv_path=macro_csv_path,
                use_fred_macro=use_fred,
                use_evds_macro=bool(use_evds),
                evds_series_codes=evds_codes,
                seq_len=int(seq_len),
                epochs=int(scan_epochs),
                conf_threshold=float(conf_threshold),
                initial_cash=float(initial_cash),
                commission=float(commission),
                slippage_bps=float(slippage_bps),
                position_size_pct=float(position_pct),
                risk_free_daily=float(bt_defaults.risk_free_daily),
                max_symbols=int(max_scan),
            )
        st.subheader("Scan results")
        st.dataframe(table, use_container_width=True)
        return

    with st.spinner("Building features..."):
        feats, feature_cols = build_feature_matrix(
            [ticker],
            start_s,
            end_s,
            market=market,
            horizon=int(label_horizon),
            hold_epsilon=float(hold_epsilon),
            settings=settings,
            macro_csv_path=macro_csv_path,
            use_fred_macro=use_fred,
            use_evds_macro=bool(use_evds),
            evds_series_codes=evds_codes,
        )

    if feats.empty or not feature_cols:
        st.error("No data returned. Check ticker, dates, and EODHD_API_KEY for BIST.")
        return

    art_dir = Path(artifacts_path)
    if train_new:
        with st.spinner("Training LSTM (may take a minute)..."):
            try:
                train_lstm_classifier(
                    feats,
                    feature_cols,
                    "y_class",
                    seq_len=int(seq_len),
                    label_spec=LabelSpec(
                        horizon=int(label_horizon),
                        hold_epsilon=float(hold_epsilon),
                    ),
                    epochs=int(epochs),
                    artifacts_dir=art_dir,
                )
            except Exception as exc:
                logger.exception("Training failed")
                st.error(f"Training failed: {exc}")
                return
    else:
        if not (art_dir / "meta.json").exists():
            st.error("Artifacts not found; enable **Train new model** or set artifacts path.")
            return

    with st.spinner("Inference + backtest..."):
        try:
            preds = predict_signals(feats, art_dir)
        except Exception as exc:
            logger.exception("Inference failed")
            st.error(f"Inference failed: {exc}")
            return

        sig = apply_confidence_threshold(
            preds["signal"].fillna(0.0),
            preds["confidence"].fillna(0.0),
            min_confidence=float(conf_threshold),
        )
        entries, exits = signals_to_entries_exits(sig)
        close = feats["close"].reindex(feats.index).astype(float)
        _, trades_df, equity = run_backtest(
            close,
            entries,
            exits,
            initial_cash=float(initial_cash),
            commission=float(commission),
            slippage_bps=float(slippage_bps),
            position_size_pct=float(position_pct),
        )
        trade_pnl = trades_df["pnl"] if "pnl" in trades_df.columns else pd.Series(dtype=float)
        report = compute_performance(
            equity,
            trade_pnl,
            initial_cash=float(initial_cash),
            risk_free_daily=float(bt_defaults.risk_free_daily),
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative PnL", f"{report.cumulative_pnl:,.2f}")
    c2.metric("Cumulative return", f"{report.cumulative_return:.2%}")
    c3.metric("Sharpe (252)", f"{report.sharpe_ratio:.3f}")
    c4.metric("Max drawdown", f"{report.max_drawdown:.2%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Sortino (252)", f"{report.sortino_ratio:.3f}")
    c6.metric("Win rate", f"{report.win_rate:.2%}")
    c7.metric("Trades", f"{report.n_trades}")

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity"))
    eq_fig.update_layout(title="Equity curve", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(eq_fig, use_container_width=True)

    dd_series = equity / equity.cummax() - 1.0
    uw_fig = go.Figure()
    uw_fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, name="Drawdown", fill="tozeroy"))
    uw_fig.update_layout(title="Underwater (drawdown)", xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(uw_fig, use_container_width=True)

    st.subheader("Trade log")
    if trades_df.empty:
        st.write("No trades in this window.")
    else:
        st.dataframe(trades_df, use_container_width=True)

    st.caption(
        f"Max drawdown (alt check): {max_drawdown(equity):.2%}. "
        "Signals are lagged one bar vs. decision time to reduce same-bar lookahead."
    )

    if run_walk_forward:
        st.subheader("Walk-forward (chronological LSTM)")
        try:
            with st.spinner("Walk-forward evaluation..."):
                wf_df = walk_forward_lstm_metrics(
                    feats,
                    feature_cols,
                    "y_class",
                    seq_len=int(seq_len),
                    n_splits=int(wf_splits),
                    epochs_per_fold=int(wf_epochs),
                )
            st.dataframe(wf_df, use_container_width=True)
        except Exception as exc:
            st.warning(f"Walk-forward skipped: {exc}")


if __name__ == "__main__":
    main()

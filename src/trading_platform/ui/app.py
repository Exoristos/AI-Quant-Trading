from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# streamlit run .../app.py (editable kurulum yokken): paket src/trading_platform altında
_src_root = Path(__file__).resolve().parents[2]
if _src_root.name == "src" and str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

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


def _inject_streamlit_secrets_into_environ() -> None:
    """Flat ``st.secrets`` (TOML) → ``os.environ`` so :class:`AppSettings` picks them up."""
    try:
        sec = st.secrets
    except Exception:
        return
    for key, value in sec.items():
        if isinstance(value, dict):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            env_val = "true" if value else "false"
        elif isinstance(value, str):
            env_val = value.strip()
            if not env_val:
                continue
        else:
            env_val = str(value)
        os.environ.setdefault(str(key), env_val)


def _stale_attr(settings: AppSettings, name: str, default):  # type: ignore[no-untyped-def]
    """Support older AppSettings when venv has an outdated editable install."""
    return getattr(settings, name, default)


@dataclass
class SidebarState:
    run_mode: str
    market: str
    ticker: str
    end_d: date
    start_d: date
    membership_csv: str
    us_scan_tickers: str
    max_scan: int
    initial_cash: float
    commission: float
    slippage_bps: float
    conf_threshold: float
    position_pct: float
    seq_len: int
    epochs: int
    scan_epochs: int
    use_fred: bool
    use_evds: bool
    evds_series: str
    label_horizon: int
    hold_epsilon: float
    train_new: bool
    run_walk_forward: bool
    wf_splits: int
    wf_epochs: int
    artifacts_path: str
    scan_artifacts_root: str
    run_btn: bool
    macro_server_path: str | None = None


def _ensure_bist_suffix(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(".IS"):
        return f"{t}.IS"
    return t


def _resolve_server_macro_path(settings: AppSettings) -> str | None:
    raw = _stale_attr(settings, "macro_csv_path", None)
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    if p.is_file():
        return str(p.resolve())
    return None


def _sidebar_public(settings: AppSettings, bt_defaults: BacktestSettings) -> SidebarState:
    market = st.selectbox("Market", ["us", "bist"], index=0)
    if market == "bist":
        bist_mode = st.radio("BIST symbol", ("Quick pick", "Custom"), horizontal=True)
        if bist_mode == "Quick pick":
            ticker = st.selectbox("Symbol", list(BIST_QUICK_PICK), index=0)
        else:
            ticker = st.text_input("Ticker (e.g. THYAO or THYAO.IS)", value="THYAO.IS")
        ticker = _ensure_bist_suffix(ticker)
    else:
        ticker = st.text_input("Ticker", value="AAPL")
    end_d = st.date_input("End date", value=date.today())
    start_d = st.date_input("Start date", value=end_d - timedelta(days=365 * 3))
    conf_threshold = st.slider("LSTM confidence threshold", 0.0, 1.0, 0.35, 0.01)
    position_pct = st.slider("Position size (% of portfolio)", 1, 100, 10) / 100.0
    run_btn = st.button("Run pipeline")
    macro_server = _resolve_server_macro_path(settings)
    return SidebarState(
        run_mode="Single symbol",
        market=market,
        ticker=ticker,
        end_d=end_d,
        start_d=start_d,
        membership_csv="",
        us_scan_tickers="",
        max_scan=5,
        initial_cash=float(bt_defaults.initial_cash),
        commission=float(bt_defaults.commission),
        slippage_bps=float(bt_defaults.slippage_bps),
        conf_threshold=float(conf_threshold),
        position_pct=float(position_pct),
        seq_len=int(_stale_attr(settings, "lstm_seq_len", 20)),
        epochs=int(_stale_attr(settings, "lstm_epochs", 15)),
        scan_epochs=int(_stale_attr(settings, "lstm_epochs", 15)),
        use_fred=bool(settings.fred_api_key),
        use_evds=bool(settings.evds_api_key),
        evds_series=str(_stale_attr(settings, "evds_series_codes", "TP.DK.USD.A.YTL")).strip(),
        label_horizon=int(_stale_attr(settings, "label_horizon", 1)),
        hold_epsilon=float(_stale_attr(settings, "hold_epsilon", 0.002)),
        train_new=bool(_stale_attr(settings, "public_train_each_run", True)),
        run_walk_forward=False,
        wf_splits=4,
        wf_epochs=5,
        artifacts_path=str(settings.artifacts_dir),
        scan_artifacts_root=str(settings.artifacts_dir / "scan"),
        run_btn=run_btn,
        macro_server_path=macro_server,
    )


def _sidebar_full(settings: AppSettings, bt_defaults: BacktestSettings) -> SidebarState:
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
    membership_default = str(_stale_attr(settings, "bist_membership_csv", ""))
    if run_mode == "Scan: membership CSV (BIST)":
        membership_csv = st.text_input(
            "BIST membership CSV path",
            value=membership_default,
            help="Sütunlar: rebalance_date, ticker. İsterseniz varsayılanı BIST_MEMBERSHIP_CSV ile .env üzerinden verin.",
        )
    else:
        membership_csv = membership_default
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
    label_horizon = st.number_input("Label horizon (bars)", min_value=1, max_value=20, value=1)
    hold_epsilon = st.number_input("HOLD band ε (abs. forward return)", min_value=0.0001, value=0.002, format="%.4f")
    train_new = st.checkbox("Train new model on this run", value=True)
    run_walk_forward = st.checkbox("Walk-forward analysis (single mode, after run)", value=False)
    wf_splits = st.number_input("Walk-forward splits", min_value=2, max_value=8, value=4)
    wf_epochs = st.number_input("Walk-forward epochs / fold", min_value=1, max_value=30, value=5)
    artifacts_path = str(settings.artifacts_dir)
    scan_artifacts_root = str(settings.artifacts_dir / "scan")
    run_btn = st.button("Run pipeline")
    return SidebarState(
        run_mode=run_mode,
        market=market,
        ticker=ticker,
        end_d=end_d,
        start_d=start_d,
        membership_csv=membership_csv,
        us_scan_tickers=us_scan_tickers,
        max_scan=int(max_scan),
        initial_cash=float(initial_cash),
        commission=float(commission),
        slippage_bps=float(slippage_bps),
        conf_threshold=float(conf_threshold),
        position_pct=float(position_pct),
        seq_len=int(seq_len),
        epochs=int(epochs),
        scan_epochs=int(scan_epochs),
        use_fred=use_fred,
        use_evds=use_evds,
        evds_series=evds_series.strip(),
        label_horizon=int(label_horizon),
        hold_epsilon=float(hold_epsilon),
        train_new=train_new,
        run_walk_forward=run_walk_forward,
        wf_splits=int(wf_splits),
        wf_epochs=int(wf_epochs),
        artifacts_path=artifacts_path.strip() or "artifacts",
        scan_artifacts_root=scan_artifacts_root.strip() or "artifacts/scan",
        run_btn=run_btn,
        macro_server_path=None,
    )


def main() -> None:
    setup_logging()
    _inject_streamlit_secrets_into_environ()
    settings = AppSettings()
    bt_defaults = BacktestSettings()
    st.set_page_config(
        page_title=str(_stale_attr(settings, "ui_page_title", "AI-Quant-Trading")),
        layout="wide",
    )
    st.title("AI-Quant-Trading")

    with st.sidebar:
        st.header("Parameters")
        if _stale_attr(settings, "public_ui", False):
            ui = _sidebar_public(settings, bt_defaults)
        else:
            ui = _sidebar_full(settings, bt_defaults)

    if not ui.run_btn:
        st.info("Configure the sidebar and click **Run pipeline**.")
        return

    start_s = ui.start_d.isoformat()
    end_s = ui.end_d.isoformat()

    if _stale_attr(settings, "public_ui", False):
        macro_csv_path = ui.macro_server_path
    else:
        raw_macro = _stale_attr(settings, "macro_csv_path", None)
        macro_csv_path = _resolve_server_macro_path(settings)
        if raw_macro and str(raw_macro).strip() and macro_csv_path is None:
            st.warning(f"Macro CSV not found (skipping): {str(raw_macro).strip()}")

    evds_codes = ui.evds_series if ui.use_evds else None

    if ui.run_mode == "Scan: membership CSV (BIST)":
        mp = Path(ui.membership_csv.strip())
        if not mp.is_file():
            st.error(f"Membership file not found: {mp}")
            return
        tickers = tickers_as_of(pd.Timestamp(ui.end_d), load_membership_csv(mp))
        if not tickers:
            st.error("No tickers for this end date; check membership dates.")
            return
        st.info(f"Scanning {len(tickers)} names (capped at {int(ui.max_scan)}).")
        with st.spinner("Scanning universe (may take several minutes)..."):
            table = scan_equities(
                tickers,
                start_s,
                end_s,
                "bist",
                settings,
                Path(ui.scan_artifacts_root),
                label_horizon=int(ui.label_horizon),
                hold_epsilon=float(ui.hold_epsilon),
                macro_csv_path=macro_csv_path,
                use_fred_macro=ui.use_fred,
                use_evds_macro=bool(ui.use_evds),
                evds_series_codes=evds_codes,
                seq_len=int(ui.seq_len),
                epochs=int(ui.scan_epochs),
                conf_threshold=float(ui.conf_threshold),
                initial_cash=float(ui.initial_cash),
                commission=float(ui.commission),
                slippage_bps=float(ui.slippage_bps),
                position_size_pct=float(ui.position_pct),
                risk_free_daily=float(bt_defaults.risk_free_daily),
                max_symbols=int(ui.max_scan),
            )
        st.subheader("Scan results")
        st.dataframe(table, use_container_width=True)
        return

    if ui.run_mode == "Scan: US tickers (comma list)":
        tickers = [x.strip().upper() for x in ui.us_scan_tickers.split(",") if x.strip()]
        if not tickers:
            st.error("Enter at least one US ticker.")
            return
        st.info(f"Scanning {len(tickers)} US names (capped at {int(ui.max_scan)}).")
        with st.spinner("Scanning US tickers..."):
            table = scan_equities(
                tickers,
                start_s,
                end_s,
                "us",
                settings,
                Path(ui.scan_artifacts_root),
                label_horizon=int(ui.label_horizon),
                hold_epsilon=float(ui.hold_epsilon),
                macro_csv_path=macro_csv_path,
                use_fred_macro=ui.use_fred,
                use_evds_macro=bool(ui.use_evds),
                evds_series_codes=evds_codes,
                seq_len=int(ui.seq_len),
                epochs=int(ui.scan_epochs),
                conf_threshold=float(ui.conf_threshold),
                initial_cash=float(ui.initial_cash),
                commission=float(ui.commission),
                slippage_bps=float(ui.slippage_bps),
                position_size_pct=float(ui.position_pct),
                risk_free_daily=float(bt_defaults.risk_free_daily),
                max_symbols=int(ui.max_scan),
            )
        st.subheader("Scan results")
        st.dataframe(table, use_container_width=True)
        return

    with st.spinner("Building features..."):
        feats, feature_cols = build_feature_matrix(
            [ui.ticker],
            start_s,
            end_s,
            market=ui.market,
            horizon=int(ui.label_horizon),
            hold_epsilon=float(ui.hold_epsilon),
            settings=settings,
            macro_csv_path=macro_csv_path,
            use_fred_macro=ui.use_fred,
            use_evds_macro=bool(ui.use_evds),
            evds_series_codes=evds_codes,
        )

    if feats.empty or not feature_cols:
        st.error("No data returned. Check ticker, dates, and EODHD_API_KEY for BIST.")
        return

    art_dir = Path(ui.artifacts_path)
    if ui.train_new:
        with st.spinner("Training LSTM (may take a minute)..."):
            try:
                train_lstm_classifier(
                    feats,
                    feature_cols,
                    "y_class",
                    seq_len=int(ui.seq_len),
                    label_spec=LabelSpec(
                        horizon=int(ui.label_horizon),
                        hold_epsilon=float(ui.hold_epsilon),
                    ),
                    epochs=int(ui.epochs),
                    artifacts_dir=art_dir,
                )
            except Exception as exc:
                logger.exception("Training failed")
                st.error(f"Training failed: {exc}")
                return
    else:
        if not (art_dir / "meta.json").exists():
            if _stale_attr(settings, "public_ui", False):
                st.error("Pre-trained model artifacts are not available.")
            else:
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
            min_confidence=float(ui.conf_threshold),
        )
        entries, exits = signals_to_entries_exits(sig)
        close = feats["close"].reindex(feats.index).astype(float)
        _, trades_df, equity = run_backtest(
            close,
            entries,
            exits,
            initial_cash=float(ui.initial_cash),
            commission=float(ui.commission),
            slippage_bps=float(ui.slippage_bps),
            position_size_pct=float(ui.position_pct),
        )
        trade_pnl = trades_df["pnl"] if "pnl" in trades_df.columns else pd.Series(dtype=float)
        report = compute_performance(
            equity,
            trade_pnl,
            initial_cash=float(ui.initial_cash),
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
        "Signals use a one-bar execution lag vs. the decision bar."
    )

    if ui.run_walk_forward:
        st.subheader("Walk-forward (chronological LSTM)")
        try:
            with st.spinner("Walk-forward evaluation..."):
                wf_df = walk_forward_lstm_metrics(
                    feats,
                    feature_cols,
                    "y_class",
                    seq_len=int(ui.seq_len),
                    n_splits=int(ui.wf_splits),
                    epochs_per_fold=int(ui.wf_epochs),
                )
            st.dataframe(wf_df, use_container_width=True)
        except Exception as exc:
            st.warning(f"Walk-forward skipped: {exc}")


if __name__ == "__main__":
    main()

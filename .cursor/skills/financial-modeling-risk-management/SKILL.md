---
name: financial-modeling-risk-management
description: >-
  Guides valuation-style modeling, portfolio risk metrics, stress testing, and
  reporting conventions for trading and research. Covers returns math,
  annualization (252 trading days), drawdowns, VaR/CVaR, factor and beta
  interpretation, position sizing, and cost-aware performance. Use when building
  or reviewing DCF or scenario models, risk reports, Sharpe/Sortino/VaR,
  portfolio optimization, stress tests, capital allocation, or when the user
  mentions financial modeling, risk management, or portfolio metrics.
---

# Financial Modeling & Risk Management

## When this skill applies

Use when **quantifying value, risk, or portfolio behavior**: simple valuation worksheets, risk dashboards, backtest analytics, position limits, or strategy capacity. Pair with the **time-series / leakage** skill for anything involving train-test splits or causal features.

## Core conventions

1. **Returns**: Prefer **log returns** \( \ln(P_t/P_{t-1}) \) for additive aggregation and theory; **simple returns** \( (P_t-P_{t-1})/P_{t-1} \) when linking to PnL and linear exposure. State which is used in outputs.
2. **Annualization (equities)**: Use **252** trading days per year for daily series unless the user specifies a different calendar (e.g. crypto 365). For **monthly** data, use **12**; **weekly** typically **52**. Document the choice.
3. **Risk-free rate**: Subtract only when computing **excess** returns (Sharpe, Jensen’s alpha). Use a rate consistent with the asset’s currency and horizon; state the series and assumption.
4. **Currencies**: Do not mix nominal amounts without **FX** rules; for multi-currency portfolios, define **hedging** or **translation** assumptions explicitly.

## Return and risk metrics (definitions)

Report definitions next to numbers so results are auditable.

| Metric | Typical definition | Notes |
|--------|-------------------|--------|
| **Volatility** | Std dev of returns, annualized | Use **sample** vs **population** consistently; prefer **sample** for empirical series. |
| **Sharpe** | Mean excess return / vol (×√252 for daily) | Sensitive to **non-normality**; report alongside drawdown. |
| **Sortino** | Mean excess return / downside deviation | Downside dev uses returns below a **threshold** (often 0 or MAR); state threshold. |
| **Max drawdown** | Peak-to-trough decline on **cumulative** return index | Path-dependent; **not** captured by vol alone. |
| **Calmar** | Annualized return / \|max drawdown\| | Undefined if max DD is 0; handle edge cases. |
| **VaR (α)** | Loss not exceeded with probability (1−α) over horizon | **Parametric** (often normal) vs **historical**; state method and horizon. |
| **CVaR / ES** | Expected loss **beyond** VaR | Coherent risk measure; preferred when tails matter. |

## Portfolio and factor concepts

- **Covariance / correlation**: Require **sufficient history**; warn on **estimation error** in small samples. **Shrinkage** or **factor models** may be appropriate for large universes—say when simplifying assumptions are used.
- **Beta**: Regression of asset excess returns on benchmark excess returns; **rolling beta** needs **enough observations** per window and **aligned** dates.
- **Optimization** (mean-variance, etc.): Flag **concentration**, **corner solutions**, and **input uncertainty**; consider **constraints** (long-only, sector caps) as part of the model, not an afterthought.

## Valuation and scenario modeling (lightweight)

- **DCF / multiples**: Separate **forecast** assumptions from **terminal** assumptions; document **WACC** or exit multiple sources. **Sensitivity tables**: vary one or two drivers at a time; avoid implying precision (no false significant digits).
- **Scenarios**: Name **base / bull / bear** with explicit drivers (growth, margins, multiples). **Stress tests**: shock prices, vol, correlations, or liquidity—state whether shocks are **instantaneous** or **paths**.

## Trading and backtesting alignment

- Include **commissions**, **fees**, and **slippage** in **net** performance and risk of strategies; gross vs net should be labeled.
- **Survivorship bias**: Warn if universes omit delisted names or use only current index constituents historically.
- **Capacity**: When discussing size, tie to **liquidity** (ADV, impact)—qualitative is fine if data is missing, but flag the gap.

## Implementation defaults (Python)

- **NumPy / pandas** for returns and rolling stats; **explicit** `min_periods` on rolling windows.
- **scipy.stats** or **empirical quantiles** for historical VaR/CVaR as appropriate.
- Use **`logging`** for pipelines; avoid `print` in production code.

## Anti-patterns to flag

- Annualizing with **365** for US equity daily returns without justification.
- **Sharpe** on **gross** returns while claiming **risk-adjusted** skill without a defined risk-free rate.
- **VaR** from a normal model on **fat-tailed** returns without caveat or alternative (e.g. historical / EVT).
- **In-sample** optimization metrics presented as **out-of-sample** performance.
- Ignoring **correlation breakdown** in crisis when showing “diversified” portfolio risk.

## Relationship to other skills

- **Causality, leakage, walk-forward evaluation**: follow the **time-series analysis & data engineering** skill for sequential data and backtests.

## Progressive disclosure

For instrument-specific rules (BIST vs US session, shorting, margin), read project modules or data README when present.

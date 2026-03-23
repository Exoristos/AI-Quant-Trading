---
name: time-series-analysis-data-engineering
description: >-
  Applies rigorous time-indexed data engineering, exploratory analysis, and
  modeling hygiene for sequential data. Covers datetime indexing, resampling,
  alignment, leakage prevention, train/validation splits, and evaluation metrics
  for forecasts and signals. Use when working with time series, panel data,
  rolling windows, calendars, seasonality, lag features, backtests, pipelines,
  or when the user mentions timestamps, causality, or look-ahead bias.
---

# Time-Series Analysis & Data Engineering

## When this skill applies

Use for any workflow where **order in time matters**: ingestion, cleaning, feature generation, visualization, forecasting, or backtesting. Default to **causal** operations only: at time \(t\), only information available at or before \(t\) may influence labels or features.

## Core invariants

1. **Monotonic time index**: Sort ascending; enforce unique timestamps (or explicit multi-index keys for duplicates). Document timezone (`UTC` preferred for storage; localize for display).
2. **No look-ahead**: No `shift(-k)` for features feeding same-row labels; no global stats (mean/std of full series) applied before split; no future rows in rolling windows that include the prediction row incorrectly.
3. **Alignment**: After joins/merges, verify row counts and index; `merge_asof` for irregular series when appropriate; forward-fill only when business logic allows and document staleness risk.
4. **Gaps and irregular spacing**: Detect missing bars; decide explicit vs calendar frequency; avoid silently interpolating across large gaps in price/volume without a rule.

## Data engineering checklist

- [ ] Parse datetimes explicitly; avoid mixed formats; set `utc=True` when sources are ambiguous.
- [ ] Choose a **business calendar** (trading days vs calendar) and stick to it for annualization and resampling.
- [ ] **Resample** with clear rules: OHLC aggregation for bars (`first`/`max`/`min`/`last`/`sum` for volume); label/closed conventions documented.
- [ ] **Normalize** per series or panel with **expanding** or **rolling** stats only inside training folds, fit **per split** for ML.
- [ ] **Log** pipeline steps and row drops (NaN removal, outlier rules); make drops attributable to a rule name.

## Exploratory analysis (lightweight)

- Plot series and **differences/returns**; mark structural breaks if known.
- For univariate structure: ACF/PACF on **stationarized** series (e.g. returns) when classical models matter; note seasonality length if intraday/daily/weekly.
- Decomposition (STL/classical) only for **EDA**, not as a leak-free feature unless implemented causally (e.g. fit on past window only per row).

## Feature engineering (causal defaults)

- **Lags** `shift(k)`, \(k \ge 1\), for inputs; label horizon defines minimum lag between last used feature time and label time.
- **Rolling/expanding** features: window ends **at or before** current bar; no centering that pulls future into the window unless explicitly shifted afterward to restore causality.
- **Technical indicators**: implement or use libraries that compute **sequential** values; verify first `warmup` rows are dropped or masked consistently in train and inference.

## Train / validation / test for time series

- Prefer **purged** or **walk-forward** splits for overlapping labels or long horizons.
- **Never** shuffle time order for evaluation of sequential models or strategies.
- For hyperparameter tuning, use **nested** walk-forward or blocked CV; outer test remains strictly future.

## Forecast and signal evaluation

- Report **scale-aware** metrics (e.g. MAE, RMSE on errors) and, when relevant, **relative** metrics (MAPE only if zeros handled).
- For directional or trading-like outputs, separate **hit rate** from **economic** metrics; always state **transaction costs and slippage** assumptions when tied to PnL-style metrics.
- **Baselines**: naive (last value), seasonal naive, or simple linear model—compare against them before complex models.

## Implementation defaults (Python)

- **pandas** for indexing, resampling, alignment, `rolling`, `expanding`, `merge_asof`.
- **NumPy** for vectorized math; keep dtypes explicit (`float64` for features unless memory-bound).
- **statsmodels** when classical decomposition/unit root tests are needed; interpret tests with small-sample caution.
- Use **`logging`**, not `print`, in durable pipelines.

## Anti-patterns to flag

- Random train/test split on a single ordered series for forecasting.
- `fillna(method='bfill')` or any backward fill that imports future into past.
- StandardScaler fit on **entire** dataset before split.
- Joining features on **calendar** date without verifying **as-of** availability in production.

## Progressive disclosure

For project-specific schemas, bar rules, or indicator definitions, read the repo’s data README or module docstrings when present.

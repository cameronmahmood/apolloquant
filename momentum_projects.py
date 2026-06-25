# momentum_projects.py
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


# =========================
# Cross-Sectional Momentum
# =========================

def run_cross_sectional():
    st.subheader("🔁 Cross-Sectional Momentum Across Sectors")
    st.markdown(
        "Ranks selected tickers by past performance and rotates capital monthly into top performers. "
        "This strategy ranks assets by trailing momentum and rotates monthly into the strongest names."
    )

    start_date = "2019-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    lookback_months = 3
    holding_period_months = 1
    top_n = 5
    transaction_cost = 0.002
    risk_free_rate = 0.02 / 12

    # ============================================================
#  Drop this STOCKS list into run_cross_sectional()
#  Replaces the original 20-ticker list with 80+ tickers
#  across all major sectors
# ============================================================

    STOCKS = [
        # ── Mega Cap Tech ──────────────────────────────────────
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
        "NFLX", "ADBE", "CRM", "ORCL", "NOW", "SNOW", "PLTR",

        # ── Semiconductors ─────────────────────────────────────
        "AMD", "INTC", "QCOM", "AVGO", "MU", "AMAT", "LRCX", "KLAC",

        # ── Financials ─────────────────────────────────────────
        "JPM", "BAC", "GS", "MS", "WFC", "BLK", "V", "MA",
        "AXP", "SCHW", "C", "USB",

        # ── Healthcare ─────────────────────────────────────────
        "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT",
        "DHR", "ISRG", "REGN", "VRTX",

        # ── Consumer ───────────────────────────────────────────
        "HD", "WMT", "COST", "TGT", "MCD", "SBUX", "NKE",
        "PEP", "KO", "PG", "CL", "MDLZ",

        # ── Energy ─────────────────────────────────────────────
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO",

        # ── Industrials ────────────────────────────────────────
        "CAT", "DE", "BA", "HON", "UPS", "LMT", "RTX", "GE",

        # ── Real Estate / Utilities ────────────────────────────
        "AMT", "PLD", "EQIX", "NEE", "DUK", "SO",

        # ── Communication ──────────────────────────────────────
        "T", "VZ", "DIS", "CMCSA", "TMUS",

        # ── ETFs as stock proxies ──────────────────────────────
        "SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV",
        "XLI", "XLP", "XLU", "ARKK",
    ]
    COMMODITIES = {
        "📊 Broad Commodity Indexes": ["DBC", "PDBC", "COMT", "BCI"],
        "⚡ Energy": ["USO", "BNO", "UNG", "UGA", "XLE"],
        "🥇 Metals — Precious": ["GLD", "SLV", "PLTM", "PALL"],
        "🔩 Metals — Industrial": ["CPER", "JJN", "SLX"],
        "⛏️ Miners": ["SGDM"],
        "🌾 Agriculture": ["CORN", "SOYB", "WEAT", "JO", "NIB", "CANE"],
        "🐄 Livestock": ["COW"],
        "🚀 Thematic / Special": ["URA", "LIT", "KRBN"],
    }

    st.markdown("### Universe Selection")
    with st.expander("Stocks", expanded=True):
        sel_all_stocks = st.checkbox("Select all stocks", key="all_stocks", value=False)
        default_stocks = STOCKS if sel_all_stocks else ["AAPL", "MSFT", "NVDA"]
        selected_stocks = st.multiselect("Choose stocks", options=STOCKS, default=default_stocks, key="stocks_multiselect")

    with st.expander("Commodities", expanded=True):
        sel_all_comms = st.checkbox("Select ALL commodities", key="all_comms", value=False)
        selected_commodities = []
        if sel_all_comms:
            for _, tickers in COMMODITIES.items():
                selected_commodities.extend(tickers)
        else:
            for section_name, tickers in COMMODITIES.items():
                with st.expander(section_name, expanded=False):
                    sec_key = section_name.replace(" ", "_")
                    sec_all = st.checkbox(f"Select all in {section_name}", key=f"all_{sec_key}", value=False)
                    chosen = st.multiselect("Tickers", options=tickers, default=tickers if sec_all else [], key=f"multi_{sec_key}")
                    selected_commodities.extend(chosen)

    main_tickers = list(dict.fromkeys(selected_stocks + selected_commodities))
    if not main_tickers:
        st.info("No tickers selected — defaulting to ['AAPL','MSFT','NVDA','GLD','DBC'].")
        main_tickers = ["AAPL", "MSFT", "NVDA", "GLD", "DBC"]

    short_bottom = st.checkbox("🔻 Short Bottom 20% (Long-Only recommended for paper trading)", value=False)

    with st.spinner("Downloading data..."):
        price_data = yf.download(main_tickers, start=start_date, end=end_date, auto_adjust=True)[["Close"]]
        price_data.columns = price_data.columns.droplevel(0)
        spy_data = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True)["Close"]
        price_data["SPY"] = spy_data

    monthly_prices = price_data.resample("ME").last()
    momentum = monthly_prices.pct_change(periods=lookback_months)

    # ---- Current Top 5 Picks ----
    st.markdown("### 🎯 Current Top 5 Momentum Picks")
    last_rebalance = momentum.index[-1]
    latest_scores = momentum.loc[last_rebalance].drop("SPY").dropna().sort_values(ascending=False)
    next_rebalance = (last_rebalance + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    if not latest_scores.empty:
        st.info(f"**Signal Date:** {last_rebalance.strftime('%Y-%m-%d')} | **Next Rebalance:** {next_rebalance}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🟢 Top 5 — Consider Buying**")
            for i, (ticker, score) in enumerate(latest_scores.head(5).items(), 1):
                st.markdown(f"{i}. **{ticker}** — {score*100:+.1f}% momentum")
        with col2:
            st.markdown("**🔴 Bottom 5 — Avoid**")
            for i, (ticker, score) in enumerate(latest_scores.tail(5).iloc[::-1].items(), 1):
                st.markdown(f"{i}. **{ticker}** — {score*100:+.1f}% momentum")

    st.subheader("Momentum Scores (Last 10 Rows)")
    st.dataframe(momentum.tail(10).round(4))

    # ===== Backtest =====
    portfolio_value = 1000.0
    portfolio_values, spy_values, monthly_returns, rebalance_log = [], [], [], []
    selection_matrix = pd.DataFrame(0, index=momentum.index, columns=monthly_prices.columns)
    rebalance_dates = momentum.dropna(how="all").index

    for date in rebalance_dates:
        try:
            past_returns = momentum.loc[date].drop("SPY").dropna()
            actual_top_n = min(top_n, len(past_returns))
            if actual_top_n == 0:
                continue
            top_names = past_returns.nlargest(actual_top_n)
            entry_prices = monthly_prices.loc[date, top_names.index]
            exit_idx = monthly_prices.index.get_loc(date) + holding_period_months
            if exit_idx >= len(monthly_prices.index):
                break
            exit_date = monthly_prices.index[exit_idx]
            exit_prices = monthly_prices.loc[exit_date, top_names.index]
            long_returns = (exit_prices - entry_prices) / entry_prices
            position_marks = pd.Series(0, index=monthly_prices.columns)

            if short_bottom:
                bottom_names = past_returns.nsmallest(actual_top_n)
                short_entry = monthly_prices.loc[date, bottom_names.index]
                short_exit = monthly_prices.loc[exit_date, bottom_names.index]
                short_returns = (short_entry - short_exit) / short_entry
                net_returns = pd.concat([long_returns, short_returns]) - transaction_cost
                position_marks[top_names.index] = 1
                position_marks[bottom_names.index] = -1
            else:
                net_returns = long_returns - transaction_cost
                position_marks[top_names.index] = 1

            avg_return = net_returns.mean()
            portfolio_value *= (1 + avg_return)
            spy_entry = monthly_prices.loc[date, "SPY"]
            spy_exit = monthly_prices.loc[exit_date, "SPY"]
            spy_return = (spy_exit - spy_entry) / spy_entry
            spy_value = spy_values[-1] * (1 + spy_return) if spy_values else 1000.0

            portfolio_values.append(portfolio_value)
            spy_values.append(spy_value)
            monthly_returns.append(avg_return)
            selection_matrix.loc[date] = position_marks
            rebalance_log.append({
                "Rebalance Date": date.strftime("%Y-%m-%d"),
                "Exit Date": exit_date.strftime("%Y-%m-%d"),
                "Top Tickers": ", ".join(top_names.index),
                "Monthly Return (%)": round(avg_return * 100, 2),
            })
        except Exception:
            continue

    if not portfolio_values:
        st.warning("Not enough data to run backtest.")
        return

    result_df = pd.DataFrame({
        "Portfolio Value": portfolio_values,
        "SPY Value": spy_values,
        "Monthly Return": monthly_returns,
    }, index=rebalance_dates[:len(portfolio_values)])

    returns = pd.Series(monthly_returns, index=result_df.index)
    excess_returns = returns - risk_free_rate
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(12)
    volatility = returns.std() * np.sqrt(12)
    cumulative = pd.Series(portfolio_values).pct_change().add(1).cumprod()
    drawdown = 1 - cumulative / cumulative.cummax()
    max_drawdown = drawdown.max()
    total_years = (result_df.index[-1] - result_df.index[0]).days / 365.25
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / total_years) - 1
    downside_returns = returns[returns < risk_free_rate]
    downside_std = downside_returns.std()
    sortino = (returns.mean() - risk_free_rate) / downside_std if downside_std > 0 else np.nan
    spy_cagr = (spy_values[-1] / spy_values[0]) ** (1 / total_years) - 1 if spy_values[0] > 0 else 0

    st.subheader("📊 Strategy Performance vs SPY")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c2.metric("Sortino Ratio", f"{sortino:.2f}" if pd.notna(sortino) else "N/A")
    c3.metric("CAGR", f"{cagr:.2%}")
    c4.metric("Ann. Volatility", f"{volatility:.2%}")
    c5.metric("Max Drawdown", f"{max_drawdown:.2%}")
    st.metric("SPY CAGR (benchmark)", f"{spy_cagr:.2%}",
              delta=f"Outperforms by {(cagr-spy_cagr)*100:.1f}%" if cagr > spy_cagr else f"Underperforms by {(spy_cagr-cagr)*100:.1f}%")

    st.subheader("Last 5 Rebalances")
    st.dataframe(pd.DataFrame(rebalance_log).tail())

    st.subheader("📈 Strategy vs SPY")
    st.line_chart(result_df[["Portfolio Value", "SPY Value"]])

    st.subheader("📊 Selection Heatmap")
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = selection_matrix.loc[rebalance_dates[:len(portfolio_values)]].T
    heatmap_data.columns = heatmap_data.columns.strftime("%Y-%m")
    sns.heatmap(heatmap_data, cmap=sns.color_palette(["red", "white", "blue"], as_cmap=True),
                center=0, cbar=True, ax=ax)
    st.pyplot(fig)
    st.caption(f"Last updated: {datetime.today().strftime('%Y-%m-%d')} | Data via Yahoo Finance")


# =========================
# Dual Momentum Helpers
# =========================

def _fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x*100:.2f}%"

def _fmt_num(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.2f}"

def _ensure_month_end_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = None
    return df.resample("ME").last()

def _parse_csv(file):
    df = pd.read_csv(file)
    df.rename(columns={c: c.upper().strip() for c in df.columns}, inplace=True)
    if not all(c in df.columns for c in ["DATE", "US", "INTL", "CASH"]):
        raise ValueError("CSV must include columns: Date, US, INTL, CASH (TBILL optional)")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)
    df = df[[c for c in ["US", "INTL", "CASH", "TBILL"] if c in df.columns]].astype(float)
    return _ensure_month_end_index(df).sort_index()

def _gen_demo_data(start="2010-01-31", periods=180, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="ME")
    US, INTL, CASH, TBILL = [100.0], [100.0], [100.0], [100.0]
    for _ in range(periods - 1):
        US.append(US[-1] * (1 + (rng.random() - 0.45) * 0.08))
        INTL.append(INTL[-1] * (1 + (rng.random() - 0.48) * 0.09))
        CASH.append(CASH[-1] * (1 + 0.001 + (rng.random() - 0.5) * 0.002))
        TBILL.append(TBILL[-1] * (1 + 0.0015 + (rng.random() - 0.5) * 0.0015))
    return pd.DataFrame({"US": US, "INTL": INTL, "CASH": CASH, "TBILL": TBILL}, index=dates)

# _today parameter forces cache to refresh daily
@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_monthly_from_yf(tickers: dict, start: str = "2005-01-01", _today: str = ""):
    end = datetime.today().strftime("%Y-%m-%d")
    frames = {}
    for col, tkr in tickers.items():
        try:
            raw = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download {tkr}: {e}")
        if raw is None or raw.empty:
            raise ValueError(f"No data returned for ticker: {tkr}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [
                '_'.join([str(c) for c in col_tuple if str(c) != tkr]).strip('_') or col_tuple[0]
                for col_tuple in raw.columns
            ]
        price_col = None
        for candidate in ["Close", "Adj Close", f"Close_{tkr}", f"Adj Close_{tkr}"]:
            if candidate in raw.columns:
                price_col = candidate
                break
        if price_col is None:
            numeric_cols = raw.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise KeyError(f"No usable price column for {tkr}. Columns: {list(raw.columns)}")
            price_col = numeric_cols[0]
        s = raw[price_col].dropna()
        if s.empty:
            raise ValueError(f"Price series empty for {tkr}")
        s.index = pd.to_datetime(s.index)
        s.index.name = None
        frames[col] = s.rename(col)

    df = pd.concat(frames.values(), axis=1)
    df.columns = list(frames.keys())
    df.index = pd.to_datetime(df.index)
    df.index.name = None
    monthly = df.resample("ME").last().dropna(how="any")
    if monthly.empty:
        raise ValueError("Monthly data is empty after resampling.")
    return monthly / monthly.iloc[0] * 100.0

def _lb_ret(series, i, L):
    if i - L < 0:
        return None
    return series.iat[i] / series.iat[i - L] - 1.0

@dataclass
class _BTResults:
    equity: pd.Series
    drawdown: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict

def _backtest_dual(levels, lookback=12, abs_mode="ZERO"):
    if not {"US", "INTL", "CASH"}.issubset(levels.columns):
        raise ValueError("levels must include US, INTL, CASH")
    df = levels.copy().sort_index()
    eq = pd.Series(index=df.index, dtype=float)
    dd = pd.Series(index=df.index, dtype=float)
    w  = pd.DataFrame(index=df.index, columns=["US", "INTL", "CASH"], dtype=float)
    trades = []
    equity = 100.0; peak = 100.0; pos = None

    for i, date in enumerate(df.index):
        if i < lookback:
            eq.iloc[i] = equity; dd.iloc[i] = 0.0; w.iloc[i] = [0.0, 0.0, 1.0]
            continue
        r_us  = _lb_ret(df["US"],   i, lookback)
        r_int = _lb_ret(df["INTL"], i, lookback)
        winner = "US" if (r_us or -1) > (r_int or -1) else "INTL"
        winner_ret = r_us if winner == "US" else r_int
        threshold = 0.0
        if abs_mode == "TBILL" and "TBILL" in df.columns:
            r_tb = _lb_ret(df["TBILL"], i, lookback)
            threshold = r_tb if r_tb is not None else 0.0
        target = winner if (winner_ret is not None and winner_ret > threshold) else "CASH"
        mret = df[target].iat[i] / df[target].iat[i - 1] - 1.0
        equity *= (1 + mret); peak = max(peak, equity)
        eq.iloc[i] = equity; dd.iloc[i] = equity / peak - 1.0
        row = {"US": 0.0, "INTL": 0.0, "CASH": 0.0}; row[target] = 1.0; w.iloc[i] = row
        if pos != target:
            trades.append({
                "Date": date, "Position": target,
                "Reason": "Absolute momentum below threshold — switched to Cash/Bonds" if target == "CASH"
                          else f"Relative momentum favored {target} over {'INTL' if target == 'US' else 'US'}"
            })
            pos = target

    rets = eq.dropna().pct_change().dropna()
    if len(rets):
        avg = rets.mean(); vol = rets.std(ddof=1)
        ann_ret = (1 + avg) ** 12 - 1; ann_vol = vol * np.sqrt(12)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        downside = rets[rets < 0]; dstd = downside.std(ddof=1) if len(downside) else np.nan
        sortino  = ((1 + avg) ** 12 - 1) / (dstd * np.sqrt(12)) if (isinstance(dstd, float) and dstd > 0) else np.nan
        maxdd    = dd.min()
        calmar   = ann_ret / abs(maxdd) if (maxdd is not None and maxdd < 0) else np.nan
        winrate  = (rets > 0).mean()
    else:
        ann_ret = ann_vol = sharpe = sortino = maxdd = calmar = winrate = np.nan

    return _BTResults(
        equity=eq, drawdown=dd, weights=w,
        trades=(pd.DataFrame(trades).set_index("Date") if trades else pd.DataFrame(columns=["Position", "Reason"])),
        metrics={"CAGR": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
                 "MaxDD": maxdd, "Calmar": calmar, "WinRate": winrate, "Samples": len(rets)},
    )


# =========================
# Dual Momentum UI
# =========================

def run_dual_momentum():
    st.subheader("🧮 Dual Momentum Strategy")
    st.markdown(
        "Combines **relative momentum** (US vs International equities) and "
        "**absolute momentum** (trend filter) to decide whether to hold US equities, "
        "international equities, or cash/bonds. A simple, evidence-based monthly rotation strategy."
    )

    with st.expander("📖 How Dual Momentum Works", expanded=False):
        st.markdown("""
**Two-step process each month:**

**Step 1 — Relative Momentum:** Compare US equities vs International equities.
Hold whichever has stronger trailing momentum over the lookback period.

**Step 2 — Absolute Momentum:** If the relative winner has positive momentum vs the threshold, hold it.
If NOT → switch to Cash/Bonds (defensive).

**ETF Mapping:**
| Allocation | Default ETF | Description |
|-----------|-------------|-------------|
| 🟢 US Equities | SPY | S&P 500 |
| 🔵 International | ACWX | All-World ex-US |
| 🔴 Cash/Bonds | IEF | 7-10Y Treasuries |

**Why it works:** Momentum is one of the most robust factors in academic finance. By combining relative and absolute momentum, this strategy avoids both underperforming assets AND bear markets.
""")

    st.markdown("### ⚙️ Settings")
    c0, c1, c2 = st.columns(3)
    with c0:
        lookback = st.selectbox("Lookback (months)", [3, 6, 9, 10, 11, 12, 18], index=4, key="dm_lookback")
    with c1:
        abs_mode = st.selectbox("Absolute Filter", ["ZERO", "TBILL"], index=0, key="dm_absmode",
                                 help="0% threshold (ZERO) or beat T-Bill return (TBILL)")
    with c2:
        src = st.radio("Data Source", ["Fetch with yfinance", "Upload CSV", "Demo (synthetic)"],
                       index=0, key="dm_src", horizontal=True)

    # Ticker inputs
    us_tkr = "SPY"; intl_tkr = "ACWX"; cash_tkr = "IEF"; tbill_tkr = "BIL"
    start = "2005-01-01"
    uploaded = None

    if src == "Upload CSV":
        uploaded = st.file_uploader("Upload monthly CSV", type=["csv"], key="dm_csv")
    elif src == "Fetch with yfinance":
        st.markdown("**ETF Tickers** (customize as needed):")
        cA, cB = st.columns(2)
        with cA:
            us_tkr   = st.text_input("🟢 US Equities",   value="SPY",  key="dm_us")
            cash_tkr = st.text_input("🔴 Cash/Bonds",    value="IEF",  key="dm_cash")
        with cB:
            intl_tkr  = st.text_input("🔵 International", value="ACWX", key="dm_intl")
            tbill_tkr = st.text_input("T-Bill proxy",     value="BIL",  key="dm_tbill")
        start = st.date_input("Start date", value=pd.Timestamp("2005-01-01"), key="dm_start").strftime("%Y-%m-%d")
        st.caption(f"Data will be fetched through today: **{datetime.today().strftime('%Y-%m-%d')}**")

    # Load data
    today_str = datetime.today().strftime("%Y-%m-%d")

    if src == "Upload CSV" and uploaded is not None:
        try:
            levels = _parse_csv(uploaded)
            st.success(f"Loaded CSV: {levels.index.min().date()} → {levels.index.max().date()}")
        except Exception as e:
            st.error(f"CSV error: {e}")
            levels = _gen_demo_data()
    elif src == "Fetch with yfinance":
        with st.spinner("Fetching live data from Yahoo Finance..."):
            try:
                levels = _fetch_monthly_from_yf(
                    {"US": us_tkr, "INTL": intl_tkr, "CASH": cash_tkr, "TBILL": tbill_tkr},
                    start=start,
                    _today=today_str   # forces cache refresh daily
                )
                data_end = levels.index.max().strftime("%Y-%m-%d")
                st.success(f"✅ Live data loaded: {levels.index.min().date()} → {data_end}")
                if data_end < today_str[:7]:
                    st.warning(f"⚠️ Data ends at {data_end} — this may be stale. Try refreshing the page.")
            except Exception as e:
                st.error(f"yfinance error: {e}")
                st.info("Falling back to demo data.")
                levels = _gen_demo_data()
    else:
        levels = _gen_demo_data()
        st.info("Using synthetic demo data. Switch to 'Fetch with yfinance' for live signals.")

    # Backtest
    try:
        res = _backtest_dual(levels, lookback=lookback, abs_mode=abs_mode)
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return

    # ---- Current Signal ----
    st.markdown("### 📡 Current Signal")
    if not res.trades.empty:
        current_position = res.trades["Position"].iloc[-1]
        signal_date      = res.trades.index[-1]
        signal_date_str  = signal_date.strftime("%Y-%m-%d")
        next_rebalance   = (signal_date + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

        color_map = {"US": "#2ecc71", "INTL": "#3498db", "CASH": "#e74c3c"}
        icon_map  = {"US": "🟢", "INTL": "🔵", "CASH": "🔴"}
        etf_map   = {"US": us_tkr, "INTL": intl_tkr, "CASH": cash_tkr}
        color = color_map.get(current_position, "#3498db")
        icon  = icon_map.get(current_position, "⚪")
        etf   = etf_map.get(current_position, current_position)

        # Check if signal is current month
        signal_month   = signal_date.strftime("%Y-%m")
        current_month  = datetime.today().strftime("%Y-%m")
        is_current     = signal_month >= current_month

        st.markdown(
            f"""<div style="background:{color}22;border-left:5px solid {color};
            border-radius:8px;padding:18px 22px;margin-bottom:12px;">
            <div style="font-size:1.8rem;font-weight:700;color:{color};">
                {icon} Hold {current_position} — {etf}
            </div>
            <div style="color:#ccc;font-size:0.9rem;margin-top:6px;">
                Signal date: <b>{signal_date_str}</b> &nbsp;|&nbsp;
                Next rebalance: <b>{next_rebalance}</b> &nbsp;|&nbsp;
                Lookback: <b>{lookback} months</b>
            </div>
            <div style="margin-top:8px;color:{'#2ecc71' if is_current else '#e74c3c'};font-size:0.85rem;">
                {'✅ Signal is current — valid for this month' if is_current else '⚠️ Signal may be stale — data ends before current month. Refresh the page.'}
            </div>
            </div>""",
            unsafe_allow_html=True
        )

        # ETF status cards
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🟢 US — {us_tkr}",    "← ACTIVE" if current_position == "US"   else "Inactive")
        c2.metric(f"🔵 INTL — {intl_tkr}", "← ACTIVE" if current_position == "INTL" else "Inactive")
        c3.metric(f"🔴 Cash — {cash_tkr}", "← ACTIVE" if current_position == "CASH" else "Inactive")

        # Paper trade action box
        st.markdown("### 📋 Paper Trade Action")
        reason = res.trades["Reason"].iloc[-1]
        st.markdown(
            f"""<div style="border:1px solid #f0b42966;border-radius:8px;
            padding:16px 20px;background:#f0b42908;">
            <div style="font-size:1rem;font-weight:700;color:#f0b429;margin-bottom:10px;">
                📋 Investopedia Paper Trade Action
            </div>
            <table style="width:100%;color:#ccc;font-size:0.88rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 0;width:160px;"><b>Action</b></td>
                <td>Buy / Hold <b>{etf}</b> ({current_position})</td>
            </tr>
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 0;"><b>Model</b></td>
                <td>Dual Momentum Strategy</td>
            </tr>
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 0;"><b>Reason</b></td>
                <td>{reason}</td>
            </tr>
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 0;"><b>Signal Date</b></td>
                <td>{signal_date_str}</td>
            </tr>
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 0;"><b>Next Rebalance</b></td>
                <td>{next_rebalance} — check signal on the first trading day</td>
            </tr>
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 0;"><b>Risk Check</b></td>
                <td>Run VaR Calculator + Correlation Matrix before entering</td>
            </tr>
            <tr>
                <td style="padding:6px 0;"><b>Status</b></td>
                <td style="color:{'#2ecc71' if is_current else '#e74c3c'};">
                    {'✅ Valid — signal is current month' if is_current else '⚠️ Stale — refresh page to update'}
                </td>
            </tr>
            </table></div>""",
            unsafe_allow_html=True
        )
    else:
        st.info("No signal yet — insufficient history for the selected lookback.")

    # ---- Performance Metrics ----
    st.markdown("### 📊 Performance Metrics")
    m = res.metrics
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("CAGR",         _fmt_pct(m.get("CAGR")))
    c2.metric("Ann. Vol",     _fmt_pct(m.get("AnnVol")))
    c3.metric("Sharpe",       _fmt_num(m.get("Sharpe")))
    c4.metric("Sortino",      _fmt_num(m.get("Sortino")))
    c5.metric("Max Drawdown", _fmt_pct(m.get("MaxDD")))
    c6.metric("Calmar",       _fmt_num(m.get("Calmar")))
    c7.metric("Win Rate",     _fmt_pct(m.get("WinRate")))
    c8.metric("# Months",     _fmt_num(m.get("Samples")))

    # ---- Charts ----
    st.subheader("📈 Equity Curve & Drawdown")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Equity Curve (Start = 100)**")
        st.line_chart(res.equity.dropna().rename("Equity").to_frame())
    with colB:
        st.markdown("**Drawdown**")
        st.area_chart(res.drawdown.dropna().rename("Drawdown").to_frame())

    st.subheader("📊 Allocation Timeline")
    st.area_chart(res.weights.fillna(0.0))

    # ---- Trade Log ----
    st.subheader("📋 Full Trade Log")
    if not res.trades.empty:
        st.dataframe(res.trades, use_container_width=True)
    else:
        st.info("No trades yet.")

    # ---- Export ----
    st.subheader("📥 Export")
    exp1 = pd.concat([res.equity.rename("Equity"), res.drawdown.rename("Drawdown"), res.weights], axis=1)
    buf1 = io.StringIO()
    exp1.to_csv(buf1, index_label="Date")
    st.download_button("⬇️ Download Results (CSV)", buf1.getvalue(),
                       file_name="dual_momentum_results.csv", mime="text/csv")
    if not res.trades.empty:
        buf2 = io.StringIO()
        res.trades.to_csv(buf2, index=True)
        st.download_button("⬇️ Download Trades (CSV)", buf2.getvalue(),
                           file_name="dual_momentum_trades.csv", mime="text/csv")

    st.caption(f"Last updated: {datetime.today().strftime('%Y-%m-%d')} | Data via Yahoo Finance")
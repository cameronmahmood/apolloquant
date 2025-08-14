# momentum_projects.py

from __future__ import annotations
import io, math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# yfinance may not exist locally unless installed
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


# ============================================================
# Simple SMA demo (placeholder)
# ============================================================
def run_sma():
    st.subheader("\U0001F4C8 Time-Series Momentum with SMA Crossover")
    st.write("Tests whether short-term trends in price predict future returns using SMA crossovers.")
    # You can add a chart or code sample here


# ============================================================
# Cross-Sectional Momentum (with stocks + commodities UI)
# ============================================================
def run_cross_sectional():
    st.subheader("\U0001F501 Cross-Sectional Momentum Across Sectors")
    st.write("Ranks selected tickers by past performance and rotates capital monthly into top performers.")

    # --- Parameters ---
    start_date = "2019-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    lookback_months = 3
    holding_period_months = 1
    top_n = 5
    transaction_cost = 0.002
    risk_free_rate = 0.02 / 12

    # ===== Universe selection =====
    STOCKS = [
        "AAPL","MSFT","TSLA","GOOGL","NVDA","META","AMZN","NFLX","JPM","UNH",
        "V","MA","HD","BAC","XOM","WMT","PEP","KO","CSCO","INTC"
    ]
    COMMODITIES = {
        "📊 Broad Commodity Indexes": ["DBC", "PDBC", "COMT", "BCI"],
        "⚡ Energy": ["USO", "BNO", "UNG", "UGA", "XLE"],   # XLE = equities proxy
        "🥇 Metals — Precious": ["GLD", "SLV", "PLTM", "PALL"],
        "🔩 Metals — Industrial": ["CPER", "JJN", "SLX"],
        "⛏️ Miners": ["SGDM"],  # (or SGDJ)
        "🌾 Agriculture": ["CORN", "SOYB", "WEAT", "JO", "NIB", "CANE"],
        "🐄 Livestock": ["COW"],
        "🚀 Thematic / Special Commodities": ["URA", "LIT", "KRBN"],
    }

    st.markdown("### Universe Selection")

    # ---- Stocks block ----
    with st.expander("Stocks (separate from commodities)", expanded=True):
        sel_all_stocks = st.checkbox("Select **all stocks**", key="all_stocks", value=False)
        default_stocks = STOCKS if sel_all_stocks else ["AAPL", "MSFT", "NVDA"]
        selected_stocks = st.multiselect(
            "Choose stocks", options=STOCKS, default=default_stocks, key="stocks_multiselect"
        )

    # ---- Commodities block ----
    with st.expander("Commodities (grouped by section)", expanded=True):
        sel_all_comms = st.checkbox("Select **ALL commodities** (every section)", key="all_comms", value=False)

        selected_commodities: list[str] = []
        if sel_all_comms:
            for _section, tickers in COMMODITIES.items():
                selected_commodities.extend(tickers)
        else:
            for section_name, tickers in COMMODITIES.items():
                with st.expander(section_name, expanded=False):
                    sec_key = section_name.replace(" ", "_")
                    sec_all = st.checkbox(f"Select all in {section_name}", key=f"all_{sec_key}", value=False)
                    default_sec = tickers if sec_all else []
                    chosen = st.multiselect(
                        "Tickers", options=tickers, default=default_sec, key=f"multi_{sec_key}"
                    )
                    selected_commodities.extend(chosen)

    # Final combined universe (unique & order-preserving)
    main_tickers = list(dict.fromkeys(selected_stocks + selected_commodities))

    if not main_tickers:
        st.info("No tickers selected — defaulting to ['AAPL','MSFT','NVDA','GLD','DBC'].")
        main_tickers = ["AAPL", "MSFT", "NVDA", "GLD", "DBC"]

    short_bottom = st.checkbox("\U0001F53B Short Bottom 20% of selected assets?", value=False)

    # ===== Data =====
    with st.spinner("Downloading data..."):
        price_data = yf.download(main_tickers, start=start_date, end=end_date, auto_adjust=True)[["Close"]]
        price_data.columns = price_data.columns.droplevel(0)  # drop 'Close' level
        spy_data = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True)["Close"]
        price_data["SPY"] = spy_data

    monthly_prices = price_data.resample("ME").last()
    momentum = monthly_prices.pct_change(periods=lookback_months)

    st.subheader("Momentum Scores (Last 10 Rows)")
    st.dataframe(momentum.tail(10).round(4))

    last_rebalance = momentum.dropna().index[-1]
    st.subheader(f"Latest Momentum Ranking on {last_rebalance.strftime('%Y-%m-%d')}")
    st.dataframe(momentum.loc[last_rebalance].drop("SPY").sort_values(ascending=False).round(4))

    # ===== Backtest =====
    portfolio_value = 1000.0
    portfolio_values, spy_values, monthly_returns, rebalance_log = [], [], [], []
    selection_matrix = pd.DataFrame(0, index=momentum.index, columns=monthly_prices.columns)
    rebalance_dates = momentum.dropna().index

    for date in rebalance_dates:
        try:
            past_returns = momentum.loc[date].drop("SPY")
            top_names = past_returns.nlargest(top_n)
            entry_prices = monthly_prices.loc[date, top_names.index]

            exit_idx = monthly_prices.index.get_loc(date) + holding_period_months
            if exit_idx >= len(monthly_prices.index):
                break
            exit_date = monthly_prices.index[exit_idx]
            exit_prices = monthly_prices.loc[exit_date, top_names.index]

            long_returns = (exit_prices - entry_prices) / entry_prices
            position_marks = pd.Series(0, index=monthly_prices.columns)

            if short_bottom:
                bottom_names = past_returns.nsmallest(top_n)
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

    result_df = pd.DataFrame({
        "Portfolio Value": portfolio_values,
        "SPY Value": spy_values,
        "Monthly Return": monthly_returns,
    }, index=rebalance_dates[:len(portfolio_values)])

    # ===== Metrics & Charts =====
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
    downside_std = downside_returns.std() * np.sqrt(12)
    sortino = (returns.mean() - risk_free_rate) / downside_std if downside_std > 0 else np.nan

    st.subheader("Strategy Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Sortino Ratio", f"{sortino:.2f}")
    col3.metric("CAGR", f"{cagr:.2%}")
    st.metric("Annualized Volatility", f"{volatility:.2%}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")

    st.subheader("Last 5 Rebalances")
    st.dataframe(pd.DataFrame(rebalance_log).tail())

    st.subheader("\U0001F4C8 Strategy vs SPY")
    st.line_chart(result_df[["Portfolio Value", "SPY Value"]])

    st.subheader("\U0001F4CA Selection Heatmap (Blue = Long, Red = Short)")
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = selection_matrix.loc[rebalance_dates[:len(portfolio_values)]].T
    heatmap_data.columns = heatmap_data.columns.strftime("%Y-%m")
    sns.heatmap(
        heatmap_data,
        cmap=sns.color_palette(["red", "white", "blue"], as_cmap=True),
        center=0,
        cbar=True,
        ax=ax,
    )
    st.pyplot(fig)

    st.subheader("\U0001F4C9 Rolling 12-Month Sharpe Ratio")
    rolling_sharpe = ((returns.rolling(window=12).mean() - risk_free_rate) / returns.rolling(window=12).std()) * np.sqrt(12)
    st.line_chart(rolling_sharpe)


# ============================================================
# Dual Momentum (embedded app)
# ============================================================

def _fmt_pct(x):  # small helpers for the dual-momentum UI
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x*100:.2f}%"

def _fmt_num(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.2f}"

def _ensure_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.index = pd.to_datetime(df.index); return df.resample("M").last()

def _parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.rename(columns={c: c.upper().strip() for c in df.columns}, inplace=True)
    need = ["DATE", "US", "INTL", "CASH"]
    if not all(c in df.columns for c in need):
        raise ValueError("CSV must include columns: Date, US, INTL, CASH (TBILL optional)")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)
    df = df[[c for c in ["US", "INTL", "CASH", "TBILL"] if c in df.columns]].astype(float)
    return _ensure_month_end_index(df).sort_index()

def _gen_demo_data(start="2010-01-31", periods=180, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="M")
    US, INTL, CASH, TBILL = [100.0], [100.0], [100.0], [100.0]
    for _ in range(periods - 1):
        US.append(US[-1] * (1 + (rng.random() - 0.45) * 0.08))
        INTL.append(INTL[-1] * (1 + (rng.random() - 0.48) * 0.09))
        CASH.append(CASH[-1] * (1 + 0.001 + (rng.random() - 0.5) * 0.002))
        TBILL.append(TBILL[-1] * (1 + 0.0015 + (rng.random() - 0.5) * 0.0015))
    return pd.DataFrame({"US": US, "INTL": INTL, "CASH": CASH, "TBILL": TBILL}, index=dates)

@st.cache_data(show_spinner=False)
def _fetch_monthly_from_yf(tickers: dict[str, str], start="2005-01-01") -> pd.DataFrame:
    if not HAS_YF:
        raise RuntimeError("yfinance not installed. Add it to requirements.txt")
    data = {col: yf.download(tkr, start=start, progress=False)["Adj Close"].rename(col)
            for col, tkr in tickers.items()}
    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())
    df = _ensure_month_end_index(df)
    df = df / df.iloc[0] * 100.0  # normalize to index level 100
    return df.dropna(how="any")

def _lb_ret(series: pd.Series, i: int, L: int):
    if i - L < 0: return None
    return series.iat[i] / series.iat[i - 1] - 1.0 if L == 1 else series.iat[i] / series.iat[i - L] - 1.0

@dataclass
class _BTResults:
    equity: pd.Series
    drawdown: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict

def _backtest_dual(levels: pd.DataFrame, lookback: int = 12, abs_mode: str = "ZERO") -> _BTResults:
    if not {"US", "INTL", "CASH"}.issubset(levels.columns):
        raise ValueError("levels must include US, INTL, CASH (TBILL optional)")
    df = levels.copy().sort_index()

    eq = pd.Series(index=df.index, dtype=float)
    dd = pd.Series(index=df.index, dtype=float)
    w = pd.DataFrame(index=df.index, columns=["US", "INTL", "CASH"], dtype=float)

    trades = []
    equity = 100.0
    peak = 100.0
    pos = None

    for i, date in enumerate(df.index):
        if i < lookback:
            eq.iloc[i] = equity
            dd.iloc[i] = 0.0
            w.iloc[i] = [0.0, 0.0, 1.0]
            continue

        r_us = _lb_ret(df["US"], i, lookback)
        r_int = _lb_ret(df["INTL"], i, lookback)
        winner = "US" if (r_us or -1) > (r_int or -1) else "INTL"
        winner_ret = r_us if winner == "US" else r_int

        threshold = 0.0
        if abs_mode == "TBILL" and "TBILL" in df.columns:
            r_tb = _lb_ret(df["TBILL"], i, lookback)
            threshold = r_tb if r_tb is not None else 0.0

        target = winner if (winner_ret is not None and winner_ret > threshold) else "CASH"

        series = df[target]
        mret = series.iat[i] / series.iat[i - 1] - 1.0
        equity *= (1 + mret)
        peak = max(peak, equity)
        eq.iloc[i] = equity
        dd.iloc[i] = equity / peak - 1.0

        row = {"US": 0.0, "INTL": 0.0, "CASH": 0.0}
        row[target] = 1.0
        w.iloc[i] = row

        if pos != target:
            trades.append({
                "Date": date,
                "Position": target,
                "Reason": "Absolute filter" if target == "CASH" else f"Relative favored {target}",
            })
            pos = target

    rets = eq.dropna().pct_change().dropna()
    if len(rets):
        avg = rets.mean()
        vol = rets.std(ddof=1)
        ann_ret = (1 + avg) ** 12 - 1
        ann_vol = vol * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        downside = rets[rets < 0]
        dstd = downside.std(ddof=1) if len(downside) else np.nan
        sortino = ((1 + avg) ** 12 - 1) / (dstd * np.sqrt(12)) if dstd and dstd > 0 else np.nan
        maxdd = dd.min()
        calmar = ann_ret / abs(maxdd) if (maxdd is not None and maxdd < 0) else np.nan
        winrate = (rets > 0).mean()
    else:
        ann_ret = ann_vol = sharpe = sortino = maxdd = calmar = winrate = np.nan

    return _BTResults(
        equity=eq,
        drawdown=dd,
        weights=w,
        trades=(pd.DataFrame(trades).set_index("Date") if trades else pd.DataFrame(columns=["Position", "Reason"])),
        metrics={"CAGR": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
                 "MaxDD": maxdd, "Calmar": calmar, "WinRate": winrate, "Samples": len(rets)},
    )

def run_dual_momentum():
    st.subheader("\U0001F9EE Dual Momentum Strategy")
    st.write("Combines time-series and cross-sectional filters to build a robust long-only ETF strategy.")

    with st.sidebar:
        st.header("Dual Momentum Inputs")
        lookback = st.selectbox("Lookback (months)", [3, 6, 9, 10, 11, 12, 18], index=4, key="dm_lookback")
        abs_mode = st.selectbox("Absolute Momentum Filter", ["ZERO", "TBILL"], index=0, key="dm_absmode",
                                help="0% threshold or T-Bill lookback if TBILL column present")
        st.subheader("Data Source")
        src = st.radio("Choose data", ["Upload CSV", "Demo (synthetic)", "Fetch with yfinance"], index=1, key="dm_src")

        uploaded = None
        yf_block = None
        if src == "Upload CSV":
            uploaded = st.file_uploader("Upload monthly CSV", type=["csv"], key="dm_csv",
                                        help="Columns: Date, US, INTL, CASH, (TBILL optional)")
        elif src == "Fetch with yfinance":
            if not HAS_YF:
                st.warning("yfinance not available. Add it to requirements.txt")
            else:
                st.markdown("**Example tickers** (change as desired):")
                c1, c2 = st.columns(2)
                with c1:
                    us_tkr = st.text_input("US (e.g., SPY)", value="SPY", key="dm_us")
                    cash_tkr = st.text_input("Cash/Bonds (e.g., IEF/SHY)", value="IEF", key="dm_cash")
                with c2:
                    intl_tkr = st.text_input("INTL (e.g., ACWX/EFA)", value="ACWX", key="dm_intl")
                    tbill_tkr = st.text_input("T-Bill (e.g., BIL)", value="BIL", key="dm_tbill")
                start = st.date_input("Start date", value=pd.Timestamp("2005-01-01"),
                                      key="dm_start").strftime("%Y-%m-%d")
                yf_block = (us_tkr, intl_tkr, cash_tkr, tbill_tkr, start)

    # Load data
    if src == "Upload CSV" and uploaded is not None:
        try:
            levels = _parse_csv(uploaded)
            st.success(f"Loaded CSV with {len(levels):,} rows from {levels.index.min().date()} to {levels.index.max().date()}")
        except Exception as e:
            st.error(f"CSV parse error: {e}")
            levels = _gen_demo_data()
    elif src == "Fetch with yfinance" and HAS_YF and yf_block is not None:
        us_tkr, intl_tkr, cash_tkr, tbill_tkr, start = yf_block
        try:
            levels = _fetch_monthly_from_yf({"US": us_tkr, "INTL": intl_tkr, "CASH": cash_tkr, "TBILL": tbill_tkr}, start=start)
            st.success(f"Fetched yfinance data from {levels.index.min().date()} to {levels.index.max().date()}")
        except Exception as e:
            st.error(f"yfinance error: {e}")
            levels = _gen_demo_data()
    else:
        levels = _gen_demo_data()

    # Run backtest
    try:
        res = _backtest_dual(levels, lookback=lookback, abs_mode=abs_mode)
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return

    # Metrics
    st.subheader("Performance (since first full signal)")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    m = res.metrics
    c1.metric("CAGR", _fmt_pct(m.get("CAGR")))
    c2.metric("Ann. Vol", _fmt_pct(m.get("AnnVol")))
    c3.metric("Sharpe", _fmt_num(m.get("Sharpe")))
    c4.metric("Sortino", _fmt_num(m.get("Sortino")))
    c5.metric("Max Drawdown", _fmt_pct(m.get("MaxDD")))
    c6.metric("Calmar", _fmt_num(m.get("Calmar")))
    c7.metric("Win Rate", _fmt_pct(m.get("WinRate")))
    c8.metric("# Months", _fmt_num(m.get("Samples")))

    # Charts
    st.subheader("Equity Curve & Drawdown")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Equity Curve (Start = 100)**")
        st.line_chart(res.equity.dropna().rename("Equity").to_frame())
    with colB:
        st.markdown("**Drawdown**")
        st.area_chart(res.drawdown.dropna().rename("Drawdown").to_frame())

    st.subheader("Allocation Timeline (weights)")
    st.area_chart(res.weights.fillna(0.0))

    # Trades
    st.subheader("Trade Log")
    if not res.trades.empty:
        st.dataframe(res.trades, use_container_width=True)
    else:
        st.info("No switches yet (insufficient history for selected lookback or constant allocation).")

    # Downloads
    st.subheader("Export Results")
    exp1 = pd.concat([res.equity.rename("Equity"), res.drawdown.rename("Drawdown"), res.weights], axis=1)
    buf1 = io.StringIO(); exp1.to_csv(buf1, index_label="Date")
    st.download_button("⬇️ Download Equity/Drawdown/Weights (CSV)", buf1.getvalue(),
                       file_name="dual_momentum_results.csv", mime="text/csv")

    if not res.trades.empty:
        buf2 = io.StringIO(); res.trades.to_csv(buf2, index=True)
        st.download_button("⬇️ Download Trades (CSV)", buf2.getvalue(),
                           file_name="dual_momentum_trades.csv", mime="text/csv")

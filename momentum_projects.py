import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_sma():
    st.subheader("\U0001F4C8 Time-Series Momentum with SMA Crossover")
    st.write("Tests whether short-term trends in price predict future returns using SMA crossovers.")
    # You can add a chart or code sample here

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
    # Stocks kept separate from commodities
    STOCKS = [
        'AAPL','MSFT','TSLA','GOOGL','NVDA','META','AMZN','NFLX','JPM','UNH',
        'V','MA','HD','BAC','XOM','WMT','PEP','KO','CSCO','INTC'
    ]

    COMMODITIES = {
        "📊 Broad Commodity Indexes": ["DBC", "PDBC", "COMT", "BCI"],
        "⚡ Energy": ["USO", "BNO", "UNG", "UGA", "XLE"],  # XLE = equities proxy
        "🥇 Metals — Precious": ["GLD", "SLV", "PLTM", "PALL"],
        "🔩 Metals — Industrial": ["CPER", "JJN", "SLX"],
        "⛏️ Miners": ["SGDM"],  # (or SGDJ if you prefer)
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
            "Choose stocks",
            options=STOCKS,
            default=default_stocks,
            key="stocks_multiselect"
        )

    # ---- Commodities block ----
    with st.expander("Commodities (grouped by section)", expanded=True):
        sel_all_comms = st.checkbox("Select **ALL commodities** (every section)", key="all_comms", value=False)

        selected_commodities = []
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
                        "Tickers",
                        options=tickers,
                        default=default_sec,
                        key=f"multi_{sec_key}"
                    )
                    selected_commodities.extend(chosen)

    # Final combined universe (unique order-preserving)
    main_tickers = list(dict.fromkeys(selected_stocks + selected_commodities))

    # Safety default so the app still runs
    if not main_tickers:
        st.info("No tickers selected — defaulting to ['AAPL','MSFT','NVDA','GLD','DBC'].")
        main_tickers = ['AAPL', 'MSFT', 'NVDA', 'GLD', 'DBC']

    # Strategy option
    short_bottom = st.checkbox("\U0001F53B Short Bottom 20% of selected assets?", value=False)

    # ===== Data =====
    with st.spinner("Downloading data..."):
        price_data = yf.download(main_tickers, start=start_date, end=end_date, auto_adjust=True)[['Close']]
        price_data.columns = price_data.columns.droplevel(0)  # drop 'Close' multiindex level
        spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
        price_data['SPY'] = spy_data

    monthly_prices = price_data.resample('ME').last()
    momentum = monthly_prices.pct_change(periods=lookback_months)

    st.subheader("Momentum Scores (Last 10 Rows)")
    st.dataframe(momentum.tail(10).round(4))

    last_rebalance = momentum.dropna().index[-1]
    st.subheader(f"Latest Momentum Ranking on {last_rebalance.strftime('%Y-%m-%d')}")
    st.dataframe(momentum.loc[last_rebalance].drop('SPY').sort_values(ascending=False).round(4))

    # ===== Backtest =====
    portfolio_value = 1000.0
    portfolio_values, spy_values, monthly_returns, rebalance_log = [], [], [], []
    selection_matrix = pd.DataFrame(0, index=momentum.index, columns=monthly_prices.columns)
    rebalance_dates = momentum.dropna().index

    for date in rebalance_dates:
        try:
            past_returns = momentum.loc[date].drop('SPY')
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

            # SPY benchmark
            spy_entry = monthly_prices.loc[date, 'SPY']
            spy_exit = monthly_prices.loc[exit_date, 'SPY']
            spy_return = (spy_exit - spy_entry) / spy_entry
            spy_value = spy_values[-1] * (1 + spy_return) if spy_values else 1000.0

            portfolio_values.append(portfolio_value)
            spy_values.append(spy_value)
            monthly_returns.append(avg_return)
            selection_matrix.loc[date] = position_marks

            rebalance_log.append({
                'Rebalance Date': date.strftime('%Y-%m-%d'),
                'Exit Date': exit_date.strftime('%Y-%m-%d'),
                'Top Tickers': ', '.join(top_names.index),
                'Monthly Return (%)': round(avg_return * 100, 2)
            })

        except Exception:
            continue

    result_df = pd.DataFrame({
        'Portfolio Value': portfolio_values,
        'SPY Value': spy_values,
        'Monthly Return': monthly_returns
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
    st.line_chart(result_df[['Portfolio Value', 'SPY Value']])

    st.subheader("\U0001F4CA Selection Heatmap (Blue = Long, Red = Short)")
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = selection_matrix.loc[rebalance_dates[:len(portfolio_values)]].T
    heatmap_data.columns = heatmap_data.columns.strftime('%Y-%m')
    sns.heatmap(
        heatmap_data,
        cmap=sns.color_palette(["red", "white", "blue"], as_cmap=True),
        center=0,
        cbar=True,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("\U0001F4C9 Rolling 12-Month Sharpe Ratio")
    rolling_sharpe = (
        (returns.rolling(window=12).mean() - risk_free_rate) /
        returns.rolling(window=12).std()
    ) * np.sqrt(12)
    st.line_chart(rolling_sharpe)


def run_dual_momentum():
    st.subheader("\U0001F9EE Dual Momentum Strategy")
    st.write("Combines time-series and cross-sectional filters to build a robust long-only ETF strategy.")
    # Add your logic or visuals
"""
Modular Dual Momentum component for Streamlit multi-page apps.
- No st.set_page_config here (keep it in your main app).
- Import `render_dual_momentum` and call it from your Projects page (inside an expander/section).

Usage in momentum_projects.py (example):
    from dual_momentum import render_dual_momentum
    with st.expander("📊 Dual Momentum Strategy", expanded=False):
        render_dual_momentum()
"""

import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# Optional yfinance
try:
    import yfinance as yf
    HAS_YF = True
except Exception:  # pragma: no cover
    HAS_YF = False

# -------------------- Helpers --------------------
def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.2f}%"


def _fmt_num(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.2f}"


def _ensure_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("M").last()


def _parse_csv(file: io.BytesIO | io.StringIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c: c.upper().strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    need = ["DATE", "US", "INTL", "CASH"]
    if not all(c in df.columns for c in need):
        raise ValueError("CSV must include: Date, US, INTL, CASH (TBILL optional)")
    df["DATE"] = pd.to_datetime(df["DATE"])  # coerce
    df.set_index("DATE", inplace=True)
    df = df[[c for c in ["US", "INTL", "CASH", "TBILL"] if c in df.columns]].astype(float)
    df = _ensure_month_end_index(df)
    return df.sort_index()


def _gen_demo_data(start="2010-01-31", periods=180, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="M")
    US = [100.0]
    INTL = [100.0]
    CASH = [100.0]
    TBILL = [100.0]
    for _ in range(periods - 1):
        r_us = (rng.random() - 0.45) * 0.08
        r_int = (rng.random() - 0.48) * 0.09
        r_cash = 0.001 + (rng.random() - 0.5) * 0.002
        r_tb = 0.0015 + (rng.random() - 0.5) * 0.0015
        US.append(US[-1] * (1 + r_us))
        INTL.append(INTL[-1] * (1 + r_int))
        CASH.append(CASH[-1] * (1 + r_cash))
        TBILL.append(TBILL[-1] * (1 + r_tb))
    return pd.DataFrame({"US": US, "INTL": INTL, "CASH": CASH, "TBILL": TBILL}, index=dates)


def _lookback_return(series: pd.Series, i: int, L: int) -> float | None:
    if i - L < 0:
        return None
    now, then = series.iat[i], series.iat[i - L]
    return (now / then) - 1.0


@dataclass
class _BTResults:
    equity: pd.Series
    drawdown: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _backtest_dual_momentum(levels: pd.DataFrame, lookback: int = 12, abs_mode: str = "ZERO") -> _BTResults:
    cols = [c for c in ["US", "INTL", "CASH", "TBILL"] if c in levels.columns]
    if not set(["US", "INTL", "CASH"]).issubset(cols):
        raise ValueError("levels must include US, INTL, CASH (TBILL optional)")
    df = levels.copy().sort_index()

    eq = pd.Series(index=df.index, dtype=float)
    dd = pd.Series(index=df.index, dtype=float)
    w = pd.DataFrame(index=df.index, columns=["US", "INTL", "CASH"], dtype=float)

    trades = []
    equity = 100.0
    peak = 100.0
    pos = None

    for i in range(len(df)):
        date = df.index[i]
        if i < lookback:
            eq.iloc[i] = equity
            dd.iloc[i] = 0.0
            w.iloc[i] = [0.0, 0.0, 1.0]
            continue

        r_us = _lookback_return(df["US"], i, lookback)
        r_int = _lookback_return(df["INTL"], i, lookback)
        winner = "US" if (r_us or -1) > (r_int or -1) else "INTL"
        winner_ret = r_us if winner == "US" else r_int

        threshold = 0.0
        if abs_mode == "TBILL" and "TBILL" in df.columns:
            r_tb = _lookback_return(df["TBILL"], i, lookback)
            threshold = r_tb if r_tb is not None else 0.0

        target = winner if (winner_ret is not None and winner_ret > threshold) else "CASH"

        # this month's return of the target
        if target == "US":
            mret = df["US"].iat[i] / df["US"].iat[i - 1] - 1
        elif target == "INTL":
            mret = df["INTL"].iat[i] / df["INTL"].iat[i - 1] - 1
        else:
            mret = df["CASH"].iat[i] / df["CASH"].iat[i - 1] - 1

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
                "Reason": (
                    f"Absolute filter triggered (L={lookback}m)" if target == "CASH" else
                    f"Relative momentum favored {target} (L={lookback}m)"
                )
            })
            pos = target

    # Metrics
    eq_filled = eq.dropna()
    rets = eq_filled.pct_change().dropna()
    if len(rets) > 0:
        avg_m = rets.mean()
        vol_m = rets.std(ddof=1)
        ann_ret = (1 + avg_m) ** 12 - 1
        ann_vol = vol_m * math.sqrt(12)
        sharpe = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
        downside = rets[rets < 0]
        if len(downside) > 0:
            down_stdev_m = downside.std(ddof=1)
            sortino = ((1 + avg_m) ** 12 - 1) / (down_stdev_m * math.sqrt(12)) if down_stdev_m > 0 else np.nan
        else:
            sortino = np.nan
        maxdd = dd.min()
        calmar = (ann_ret / abs(maxdd)) if (maxdd is not None and maxdd < 0) else np.nan
        winrate = (rets > 0).mean()
    else:
        ann_ret = ann_vol = sharpe = sortino = maxdd = calmar = winrate = np.nan

    return _BTResults(
        equity=eq,
        drawdown=dd,
        weights=w,
        trades=pd.DataFrame(trades).set_index("Date") if trades else pd.DataFrame(columns=["Position", "Reason"]),
        metrics={
            "CAGR": ann_ret,
            "AnnVol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": maxdd,
            "Calmar": calmar,
            "WinRate": winrate,
            "Samples": len(rets),
        },
    )


# -------------------- Public UI Renderer --------------------
def render_dual_momentum():
    """Drop-in Streamlit UI to render under your existing Projects page/expander."""
    st.markdown("### 📊 Dual Momentum Strategy")
    st.caption("Relative momentum (US vs INTL) + absolute momentum (0% or T‑Bills). Monthly rebalancing.")

    # Controls (use columns to fit in your layout)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        lookback = st.selectbox("Lookback (months)", [3, 6, 9, 10, 11, 12, 18], index=4, key="dm_lb")
    with c2:
        abs_mode = st.selectbox("Abs Filter", ["ZERO", "TBILL"], index=0, key="dm_abs")
    with c3:
        src = st.radio("Data", ["Upload CSV", "Demo", "yfinance"], index=0, horizontal=True, key="dm_src")

    # Load data
    if src == "Upload CSV":
        up = st.file_uploader("Monthly CSV: Date, US, INTL, CASH, (TBILL)", type=["csv"], key="dm_csv")
        if up is not None:
            try:
                levels = _parse_csv(up)
                st.success(f"Loaded {len(levels):,} rows • {levels.index.min().date()} → {levels.index.max().date()}")
            except Exception as e:
                st.error(f"CSV parse error: {e}")
                levels = _gen_demo_data()
        else:
            levels = _gen_demo_data()
    elif src == "yfinance":
        if not HAS_YF:
            st.warning("yfinance not installed in this environment.")
            levels = _gen_demo_data()
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                us_tkr = st.text_input("US", value="SPY", key="dm_us")
            with c2:
                intl_tkr = st.text_input("INTL", value="ACWX", key="dm_intl")
            with c3:
                cash_tkr = st.text_input("Cash/Bonds", value="IEF", key="dm_cash")
            with c4:
                tbill_tkr = st.text_input("T‑Bill", value="BIL", key="dm_tbill")
            start = st.date_input("Start date", value=pd.Timestamp("2005-01-01"), key="dm_start").strftime("%Y-%m-%d")
            try:
                data = {}
                for col, tkr in {"US": us_tkr, "INTL": intl_tkr, "CASH": cash_tkr, "TBILL": tbill_tkr}.items():
                    s = yf.download(tkr, start=start, progress=False)["Adj Close"].rename(col)
                    data[col] = s
                levels = pd.concat(data.values(), axis=1)
                levels.columns = list(data.keys())
                levels = _ensure_month_end_index(levels)
                levels = levels / levels.iloc[0] * 100.0
                levels = levels.dropna(how="any")
                st.success(f"Fetched {levels.index.min().date()} → {levels.index.max().date()}")
            except Exception as e:
                st.error(f"yfinance error: {e}")
                levels = _gen_demo_data()
    else:
        levels = _gen_demo_data()

    # Backtest
    try:
        res = _backtest_dual_momentum(levels, lookback=lookback, abs_mode=abs_mode)
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return

    # Metrics row
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("CAGR", _fmt_pct(res.metrics.get("CAGR")))
    m2.metric("Ann. Vol", _fmt_pct(res.metrics.get("AnnVol")))
    m3.metric("Sharpe", _fmt_num(res.metrics.get("Sharpe")))
    m4.metric("Sortino", _fmt_num(res.metrics.get("Sortino")))
    m5.metric("Max DD", _fmt_pct(res.metrics.get("MaxDD")))
    m6.metric("Calmar", _fmt_num(res.metrics.get("Calmar")))
    m7.metric("Win Rate", _fmt_pct(res.metrics.get("WinRate")))
    m8.metric("# Months", _fmt_num(res.metrics.get("Samples")))

    # Charts
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Equity Curve (Start=100)**")
        st.line_chart(res.equity.dropna().rename("Equity").to_frame())
    with cb:
        st.markdown("**Drawdown**")
        st.area_chart(res.drawdown.dropna().rename("Drawdown").to_frame())

    st.markdown("**Allocation Timeline**")
    st.area_chart(res.weights.fillna(0.0))

    # Trades
    st.markdown("**Trade Log**")
    if not res.trades.empty:
        st.dataframe(res.trades, use_container_width=True)
    else:
        st.info("No switches yet for the current lookback / data window.")

    # Downloads
    exp = pd.concat([
        res.equity.rename("Equity"),
        res.drawdown.rename("Drawdown"),
        res.weights
    ], axis=1)
    buf = io.StringIO(); exp.to_csv(buf, index_label="Date")
    st.download_button("⬇️ Download Equity/Drawdown/Weights (CSV)", buf.getvalue(), file_name="dual_momentum_results.csv", mime="text/csv")

    if not res.trades.empty:
        buf2 = io.StringIO(); res.trades.to_csv(buf2, index=True)
        st.download_button("⬇️ Download Trades (CSV)", buf2.getvalue(), file_name="dual_momentum_trades.csv", mime="text/csv")

    st.caption("For education only. Backtests are hypothetical.")

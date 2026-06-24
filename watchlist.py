# watchlist.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

WATCHLIST_UNIVERSE = {
    "📊 Broad Market ETFs":  ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "💻 Technology":         ["XLK", "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL"],
    "🏦 Financials":         ["XLF", "JPM", "BAC", "GS", "MS", "V", "MA"],
    "⚡ Energy":              ["XLE", "USO", "XOM", "CVX"],
    "🏥 Healthcare":         ["XLV", "UNH", "JNJ", "ABBV", "MRK"],
    "🛒 Consumer":           ["XLY", "XLP", "AMZN", "WMT", "KO", "PEP"],
    "🏭 Industrials":        ["XLI", "CAT", "UPS", "HON"],
    "💰 Fixed Income / Safe Haven": ["TLT", "IEF", "GLD", "SLV", "UUP"],
    "🌍 International":      ["EEM", "EFA", "FXI"],
}

ALL_TICKERS = list(dict.fromkeys([t for tickers in WATCHLIST_UNIVERSE.values() for t in tickers]))

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_watchlist_data():
    frames = {}
    for ticker in ALL_TICKERS:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="6mo", auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[ticker] = s
        except Exception:
            pass
    if not frames:
        return None
    return pd.DataFrame(frames).sort_index()

def _compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _score_ticker(series, spy_series):
    if len(series) < 22:
        return None

    price   = series.iloc[-1]
    ret_1d  = (series.iloc[-1] / series.iloc[-2]  - 1) * 100 if len(series) > 1  else 0
    ret_1w  = (series.iloc[-1] / series.iloc[-5]  - 1) * 100 if len(series) > 5  else 0
    ret_1m  = (series.iloc[-1] / series.iloc[-22] - 1) * 100 if len(series) > 22 else 0
    ret_3m  = (series.iloc[-1] / series.iloc[-66] - 1) * 100 if len(series) > 66 else ret_1m

    # vs SPY
    spy_1m = (spy_series.iloc[-1] / spy_series.iloc[-22] - 1) * 100 if spy_series is not None and len(spy_series) > 22 else 0
    rs_1m  = ret_1m - spy_1m

    # RSI
    rsi = _compute_rsi(series).iloc[-1]

    # Z-Score
    sma20 = series.rolling(20).mean().iloc[-1]
    std20 = series.rolling(20).std().iloc[-1]
    zscore = (price - sma20) / std20 if std20 > 0 else 0

    # Trend
    sma50  = series.rolling(50).mean().iloc[-1]  if len(series) >= 50  else None
    sma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else None
    trend = sum([
        1 if pd.notna(sma20)  and price > sma20  else 0,
        1 if sma50  and pd.notna(sma50)  and price > sma50  else 0,
        1 if sma200 and pd.notna(sma200) and price > sma200 else 0,
    ])

    # Momentum score
    mom_score = ret_1w * 1 + ret_1m * 2 + ret_3m * 3

    # Category
    if rs_1m > 5 and trend >= 2 and rsi < 70:
        category = "🟢 Leader"
    elif rs_1m > 2 and trend >= 1:
        category = "🟡 Watch"
    elif rsi < 30 and zscore < -2:
        category = "💡 Mean Reversion"
    elif rsi > 70 and zscore > 2:
        category = "⚠️ Overbought"
    elif rs_1m < -5 and trend <= 1:
        category = "🔴 Laggard"
    else:
        category = "⚪ Neutral"

    return {
        "Price":    round(price, 2),
        "1D %":     round(ret_1d, 2),
        "1W %":     round(ret_1w, 2),
        "1M %":     round(ret_1m, 2),
        "vs SPY":   round(rs_1m, 2),
        "RSI":      round(rsi, 1)  if pd.notna(rsi)    else None,
        "Z-Score":  round(zscore, 2),
        "Trend":    f"{trend}/3",
        "Mom Score":round(mom_score, 1),
        "Category": category,
    }

def run_watchlist():
    st.subheader("📋 Watchlist")
    st.markdown(
        "Auto-generated from live market data. Updated every 5 minutes. "
        "Shows **leaders, laggards, mean reversion candidates, and overbought warnings** "
        "across your entire universe."
    )

    with st.spinner("Fetching live market data..."):
        df = _fetch_watchlist_data()

    if df is None or df.empty:
        st.error("Unable to fetch market data.")
        return

    spy = df["SPY"] if "SPY" in df.columns else None

    # Compute scores
    results = {}
    for ticker in df.columns:
        score = _score_ticker(df[ticker].dropna(), spy)
        if score:
            results[ticker] = score

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Ticker"

    # ---- Summary Counts ----
    st.markdown("### 📊 Market Snapshot")
    leaders    = results_df[results_df["Category"] == "🟢 Leader"]
    watch      = results_df[results_df["Category"] == "🟡 Watch"]
    mr         = results_df[results_df["Category"] == "💡 Mean Reversion"]
    overbought = results_df[results_df["Category"] == "⚠️ Overbought"]
    laggards   = results_df[results_df["Category"] == "🔴 Laggard"]
    neutral    = results_df[results_df["Category"] == "⚪ Neutral"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🟢 Leaders",          len(leaders))
    c2.metric("🟡 Watch",            len(watch))
    c3.metric("💡 Mean Reversion",   len(mr))
    c4.metric("⚠️ Overbought",       len(overbought))
    c5.metric("🔴 Laggards",         len(laggards))

    breadth = (len(leaders) + len(watch)) / len(results_df) * 100 if len(results_df) > 0 else 0
    if breadth > 60:
        st.success(f"✅ Market breadth: {breadth:.0f}% of assets are leaders or watch — broadly bullish")
    elif breadth > 40:
        st.info(f"⚪ Market breadth: {breadth:.0f}% — mixed market, be selective")
    else:
        st.warning(f"⚠️ Market breadth: {breadth:.0f}% — most assets are lagging — defensive positioning")

    # ---- Leaders ----
    st.markdown("### 🟢 Leaders — Consider Buying")
    st.caption("Assets outperforming SPY with strong momentum and uptrend. Best candidates for long entries.")
    if not leaders.empty:
        leaders_sorted = leaders.sort_values("Mom Score", ascending=False)
        for ticker, row in leaders_sorted.head(8).iterrows():
            st.markdown(
                f"**{ticker}** — ${row['Price']:.2f} | 1M: {row['1M %']:+.1f}% | "
                f"vs SPY: {row['vs SPY']:+.1f}% | RSI: {row['RSI']} | Trend: {row['Trend']}"
            )
    else:
        st.info("No strong leaders currently. Market may be in Risk-Off or transitioning.")

    # ---- Watch ----
    st.markdown("### 🟡 Watch — Building Momentum")
    if not watch.empty:
        for ticker, row in watch.sort_values("Mom Score", ascending=False).head(6).iterrows():
            st.markdown(
                f"**{ticker}** — ${row['Price']:.2f} | 1M: {row['1M %']:+.1f}% | "
                f"vs SPY: {row['vs SPY']:+.1f}% | RSI: {row['RSI']}"
            )
    else:
        st.info("No watch candidates currently.")

    # ---- Mean Reversion ----
    st.markdown("### 💡 Mean Reversion Candidates — Oversold")
    st.caption("RSI < 30 and Z-Score < -2. High-risk contrarian setups — confirm with Market Regime before buying.")
    if not mr.empty:
        for ticker, row in mr.sort_values("Z-Score").head(5).iterrows():
            st.markdown(
                f"**{ticker}** — ${row['Price']:.2f} | RSI: {row['RSI']} | "
                f"Z-Score: {row['Z-Score']:.2f} | 1M: {row['1M %']:+.1f}%"
            )
    else:
        st.info("No extreme oversold conditions currently.")

    # ---- Overbought ----
    st.markdown("### ⚠️ Overbought — Take Profit / Avoid Chasing")
    if not overbought.empty:
        for ticker, row in overbought.sort_values("Z-Score", ascending=False).head(5).iterrows():
            st.markdown(
                f"**{ticker}** — ${row['Price']:.2f} | RSI: {row['RSI']} | "
                f"Z-Score: {row['Z-Score']:.2f} | 1M: {row['1M %']:+.1f}%"
            )
    else:
        st.info("No overbought conditions currently.")

    # ---- Laggards ----
    st.markdown("### 🔴 Laggards — Avoid")
    st.caption("Underperforming SPY with weak momentum. Do not buy laggards expecting a bounce unless Mean Reversion confirms.")
    if not laggards.empty:
        for ticker, row in laggards.sort_values("vs SPY").head(6).iterrows():
            st.markdown(
                f"**{ticker}** — ${row['Price']:.2f} | 1M: {row['1M %']:+.1f}% | "
                f"vs SPY: {row['vs SPY']:+.1f}% | Trend: {row['Trend']}"
            )
    else:
        st.info("No clear laggards currently.")

    # ---- Full Table ----
    st.markdown("### 📋 Full Watchlist Table")
    sort_col = st.selectbox("Sort by", ["Mom Score", "1M %", "vs SPY", "RSI", "Z-Score"], key="wl_sort")
    asc = st.checkbox("Ascending", value=False, key="wl_asc")
    full_sorted = results_df.sort_values(sort_col, ascending=asc)
    st.dataframe(full_sorted, use_container_width=True)

    # ---- Group view ----
    st.markdown("### 🗂️ By Sector")
    for group_name, tickers in WATCHLIST_UNIVERSE.items():
        available = [t for t in tickers if t in results_df.index]
        if not available:
            continue
        group_df = results_df.loc[available].sort_values("Mom Score", ascending=False)
        with st.expander(group_name, expanded=False):
            st.dataframe(group_df[["Price","1D %","1M %","vs SPY","RSI","Z-Score","Trend","Category"]],
                         use_container_width=True)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | Refreshes every 5 minutes"
    )

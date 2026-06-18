# mean_reversion.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timezone

# =========================
# Default Tickers
# =========================

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "TLT", "GLD", "USO",
    "UUP", "XLE", "XLF", "XLK", "XLV", "XLP",
    "XLI", "XLU", "HYG", "EEM"
]

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data(tickers: list, period: str = "6mo"):
    frames = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, auto_adjust=True)
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
    df = pd.DataFrame(frames).sort_index()
    return df

# =========================
# Indicator Calculations
# =========================

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _compute_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    pct_b = (series - lower) / (upper - lower)
    return sma, upper, lower, pct_b

def _compute_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)

def _signal_label(rsi, pct_b, zscore):
    signals = []
    if pd.notna(rsi):
        if rsi < 30:
            signals.append("RSI Oversold")
        elif rsi > 70:
            signals.append("RSI Overbought")
    if pd.notna(pct_b):
        if pct_b < 0:
            signals.append("Below BB Lower")
        elif pct_b > 1:
            signals.append("Above BB Upper")
    if pd.notna(zscore):
        if zscore < -2:
            signals.append("Z-Score Oversold")
        elif zscore > 2:
            signals.append("Z-Score Overbought")
    if not signals:
        return "Neutral"
    return " | ".join(signals)

def _overall_signal(rsi, pct_b, zscore):
    score = 0
    if pd.notna(rsi):
        if rsi < 30: score -= 1
        elif rsi > 70: score += 1
    if pd.notna(pct_b):
        if pct_b < 0: score -= 1
        elif pct_b > 1: score += 1
    if pd.notna(zscore):
        if zscore < -2: score -= 1
        elif zscore > 2: score += 1
    if score <= -2: return "🟢 Strong Buy"
    elif score == -1: return "🟡 Buy"
    elif score == 0: return "⚪ Neutral"
    elif score == 1: return "🟡 Sell"
    else: return "🔴 Strong Sell"

# =========================
# Main Function
# =========================

def run_mean_reversion():
    st.subheader("📉 Mean Reversion Scanner")
    st.markdown(
        "Scans a customizable list of tickers for **overbought and oversold conditions** "
        "using RSI, Bollinger Bands, and Z-Score. "
        "Green signals suggest mean reversion buy opportunities. Red signals suggest potential pullbacks."
    )

    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
**Three complementary mean reversion indicators:**

| Indicator | Overbought | Oversold | What it measures |
|-----------|-----------|---------|-----------------|
| **RSI (14)** | > 70 | < 30 | Momentum — how fast price is moving |
| **Bollinger Bands (20, 2σ)** | %B > 1 | %B < 0 | Volatility — how far price is from its average |
| **Z-Score (20)** | > +2 | < -2 | Statistical — standard deviations from mean |

**Signal strength:**
- 1 indicator triggered = Weak signal
- 2 indicators triggered = Moderate signal
- All 3 triggered = Strong signal — highest conviction

**How to use with your other tools:**
- Mean reversion signals work best when they **align with your Market Regime**
- In Risk-Off: oversold signals in defensive assets (TLT, GLD) are most reliable
- In Risk-On: oversold signals in growth assets (QQQ, XLK) are most reliable
- Never trade against the regime based on mean reversion alone
""")

    # ---- Controls ----
    st.markdown("### ⚙️ Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value=", ".join(DEFAULT_TICKERS),
            key="mr_tickers"
        )
    with col2:
        period = st.selectbox("Lookback Period", ["3mo", "6mo", "1y"], index=1, key="mr_period")
    with col3:
        rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=30, key="mr_rsi")

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
        return

    # ---- Fetch Data ----
    with st.spinner("Fetching data and computing indicators..."):
        df = _fetch_data(tickers, period=period)

    if df is None or df.empty:
        st.error("Unable to fetch data. Please try again.")
        return

    # ---- Compute Indicators ----
    rows = []
    for ticker in df.columns:
        series = df[ticker].dropna()
        if len(series) < 30:
            continue

        rsi = _compute_rsi(series, period=int(rsi_period)).iloc[-1]
        sma, upper, lower, pct_b = _compute_bollinger(series)
        zscore = _compute_zscore(series).iloc[-1]
        pct_b_val = pct_b.iloc[-1]
        price = series.iloc[-1]
        chg_1d = (series.iloc[-1] / series.iloc[-2] - 1) * 100 if len(series) > 1 else np.nan
        chg_1m = (series.iloc[-1] / series.iloc[-22] - 1) * 100 if len(series) > 22 else np.nan

        signal = _signal_label(rsi, pct_b_val, zscore)
        overall = _overall_signal(rsi, pct_b_val, zscore)

        rows.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "1D %": round(chg_1d, 2) if pd.notna(chg_1d) else None,
            "1M %": round(chg_1m, 2) if pd.notna(chg_1m) else None,
            "RSI (14)": round(rsi, 1) if pd.notna(rsi) else None,
            "%B": round(pct_b_val, 3) if pd.notna(pct_b_val) else None,
            "Z-Score": round(zscore, 2) if pd.notna(zscore) else None,
            "Signals": signal,
            "Overall": overall,
        })

    results = pd.DataFrame(rows)

    # ---- Summary Metrics ----
    st.markdown("### 📡 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    strong_buy = len(results[results["Overall"] == "🟢 Strong Buy"])
    buy = len(results[results["Overall"] == "🟡 Buy"])
    neutral = len(results[results["Overall"] == "⚪ Neutral"])
    sell = len(results[results["Overall"].isin(["🟡 Sell", "🔴 Strong Sell"])])
    col1.metric("🟢 Strong Buy", strong_buy)
    col2.metric("🟡 Buy", buy)
    col3.metric("⚪ Neutral", neutral)
    col4.metric("🔴 Sell / Strong Sell", sell)

    # ---- Scanner Table ----
    st.markdown("### 📊 Scanner Results")
    st.caption("Sorted by Z-Score — most oversold at top, most overbought at bottom.")
    results_sorted = results.sort_values("Z-Score", ascending=True)
    st.dataframe(results_sorted, use_container_width=True)

    # ---- Top Opportunities ----
    st.markdown("### 🎯 Top Mean Reversion Opportunities")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Most Oversold (Potential Buys)")
        oversold = results[results["Overall"].isin(["🟢 Strong Buy", "🟡 Buy"])].sort_values("Z-Score")
        if oversold.empty:
            st.info("No oversold signals currently.")
        else:
            for _, row in oversold.head(5).iterrows():
                st.markdown(f"**{row['Ticker']}** — {row['Overall']} | RSI: {row['RSI (14)']} | Z: {row['Z-Score']}")

    with col2:
        st.markdown("#### 🔴 Most Overbought (Potential Sells)")
        overbought = results[results["Overall"].isin(["🔴 Strong Sell", "🟡 Sell"])].sort_values("Z-Score", ascending=False)
        if overbought.empty:
            st.info("No overbought signals currently.")
        else:
            for _, row in overbought.head(5).iterrows():
                st.markdown(f"**{row['Ticker']}** — {row['Overall']} | RSI: {row['RSI (14)']} | Z: {row['Z-Score']}")

    # ---- Individual Ticker Deep Dive ----
    st.markdown("### 🔍 Individual Ticker Analysis")
    selected = st.selectbox("Select a ticker for detailed chart:", df.columns.tolist(), key="mr_selected")

    if selected and selected in df.columns:
        series = df[selected].dropna()
        rsi_series = _compute_rsi(series, period=int(rsi_period))
        sma, upper, lower, pct_b = _compute_bollinger(series)
        zscore_series = _compute_zscore(series)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                              gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.patch.set_facecolor("#0e1117")
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white", labelsize=8)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        # Price + Bollinger Bands
        ax1.plot(series.index, series.values, color="#1f77b4", linewidth=1.5, label="Price")
        ax1.plot(sma.index, sma.values, color="#f0b429", linewidth=1, linestyle="--", label="SMA 20", alpha=0.8)
        ax1.plot(upper.index, upper.values, color="#e74c3c", linewidth=0.8, linestyle=":", label="Upper BB", alpha=0.7)
        ax1.plot(lower.index, lower.values, color="#2ecc71", linewidth=0.8, linestyle=":", label="Lower BB", alpha=0.7)
        ax1.fill_between(upper.index, upper.values, lower.values, alpha=0.05, color="#f0b429")
        ax1.set_ylabel("Price", color="white", fontsize=9)
        ax1.set_title(f"{selected} — Bollinger Bands, RSI & Z-Score", color="white", fontsize=11)
        ax1.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white", loc="upper left")

        # RSI
        ax2.plot(rsi_series.index, rsi_series.values, color="#9b59b6", linewidth=1.2)
        ax2.axhline(70, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.axhline(30, color="#2ecc71", linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.fill_between(rsi_series.index, rsi_series.values, 70,
                          where=(rsi_series > 70), alpha=0.2, color="#e74c3c")
        ax2.fill_between(rsi_series.index, rsi_series.values, 30,
                          where=(rsi_series < 30), alpha=0.2, color="#2ecc71")
        ax2.set_ylabel("RSI", color="white", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.text(rsi_series.index[-1], 72, "Overbought", color="#e74c3c", fontsize=7, ha="right")
        ax2.text(rsi_series.index[-1], 25, "Oversold", color="#2ecc71", fontsize=7, ha="right")

        # Z-Score
        ax3.plot(zscore_series.index, zscore_series.values, color="#1abc9c", linewidth=1.2)
        ax3.axhline(2, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.axhline(-2, color="#2ecc71", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        ax3.fill_between(zscore_series.index, zscore_series.values, 2,
                          where=(zscore_series > 2), alpha=0.2, color="#e74c3c")
        ax3.fill_between(zscore_series.index, zscore_series.values, -2,
                          where=(zscore_series < -2), alpha=0.2, color="#2ecc71")
        ax3.set_ylabel("Z-Score", color="white", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # Current readings
        latest_rsi = rsi_series.iloc[-1]
        latest_z = zscore_series.iloc[-1]
        latest_pctb = pct_b.iloc[-1]

        st.markdown("#### 📋 Current Readings")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${series.iloc[-1]:.2f}")
        c2.metric("RSI (14)", f"{latest_rsi:.1f}",
                  delta="Oversold" if latest_rsi < 30 else ("Overbought" if latest_rsi > 70 else "Neutral"))
        c3.metric("Z-Score", f"{latest_z:.2f}",
                  delta="Oversold" if latest_z < -2 else ("Overbought" if latest_z > 2 else "Neutral"))
        c4.metric("%B", f"{latest_pctb:.3f}",
                  delta="Below Lower BB" if latest_pctb < 0 else ("Above Upper BB" if latest_pctb > 1 else "Inside BB"))

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | Refreshes every 5 minutes"
    )

# technical_analysis.py
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

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "TLT", "GLD", "USO"]

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data(ticker: str, period: str = "1y"):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist.dropna()
    except Exception:
        return None

# =========================
# Indicator Calculations
# =========================

def _compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def _compute_sma(close: pd.Series, windows=[20, 50, 200]):
    return {w: close.rolling(w).mean() for w in windows}

def _compute_ema(close: pd.Series, windows=[12, 26]):
    return {w: close.ewm(span=w, adjust=False).mean() for w in windows}

def _compute_rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _compute_bollinger(close: pd.Series, period=20, std=2.0):
    sma = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    return sma, sma + std * std_dev, sma - std * std_dev

def _compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _find_support_resistance(close: pd.Series, window=20, n_levels=5):
    levels = []
    for i in range(window, len(close) - window):
        local_max = close.iloc[i-window:i+window].max()
        local_min = close.iloc[i-window:i+window].min()
        if close.iloc[i] == local_max:
            levels.append(("resistance", float(close.iloc[i]), close.index[i]))
        elif close.iloc[i] == local_min:
            levels.append(("support", float(close.iloc[i]), close.index[i]))

    support = sorted([l for l in levels if l[0] == "support"], key=lambda x: -x[2])
    resistance = sorted([l for l in levels if l[0] == "resistance"], key=lambda x: -x[2])

    seen_support = []
    for s in support:
        if not any(abs(s[1] - x[1]) / x[1] < 0.02 for x in seen_support):
            seen_support.append(s)
        if len(seen_support) >= n_levels:
            break

    seen_resistance = []
    for r in resistance:
        if not any(abs(r[1] - x[1]) / x[1] < 0.02 for x in seen_resistance):
            seen_resistance.append(r)
        if len(seen_resistance) >= n_levels:
            break

    return seen_support, seen_resistance

def _compute_signal_score(close, macd, signal_line, histogram, smas, rsi, volume):
    score = 0
    signals = []

    # MACD signals
    macd_now = macd.iloc[-1]
    sig_now = signal_line.iloc[-1]
    hist_now = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

    if macd_now > sig_now and hist_now > 0:
        score += 2; signals.append("✅ MACD bullish crossover")
    elif macd_now < sig_now and hist_now < 0:
        score -= 2; signals.append("❌ MACD bearish crossover")
    if hist_now > hist_prev:
        score += 1; signals.append("✅ MACD histogram expanding bullish")
    elif hist_now < hist_prev:
        score -= 1; signals.append("❌ MACD histogram expanding bearish")

    # Moving average signals
    price = close.iloc[-1]
    if 20 in smas and 50 in smas:
        sma20 = smas[20].iloc[-1]
        sma50 = smas[50].iloc[-1]
        sma20_prev = smas[20].iloc[-2] if len(smas[20]) > 1 else sma20
        sma50_prev = smas[50].iloc[-2] if len(smas[50]) > 1 else sma50
        if price > sma20 and price > sma50:
            score += 1; signals.append("✅ Price above 20 and 50 SMA")
        elif price < sma20 and price < sma50:
            score -= 1; signals.append("❌ Price below 20 and 50 SMA")
        if sma20 > sma50 and sma20_prev <= sma50_prev:
            score += 2; signals.append("✅ Golden Cross (20 SMA crossed above 50 SMA)")
        elif sma20 < sma50 and sma20_prev >= sma50_prev:
            score -= 2; signals.append("❌ Death Cross (20 SMA crossed below 50 SMA)")

    if 200 in smas:
        sma200 = smas[200].iloc[-1]
        if pd.notna(sma200):
            if price > sma200:
                score += 1; signals.append("✅ Price above 200 SMA (long-term uptrend)")
            else:
                score -= 1; signals.append("❌ Price below 200 SMA (long-term downtrend)")

    # RSI signals
    rsi_now = rsi.iloc[-1]
    if pd.notna(rsi_now):
        if rsi_now < 30:
            score += 2; signals.append(f"✅ RSI oversold ({rsi_now:.1f}) — potential reversal up")
        elif rsi_now > 70:
            score -= 2; signals.append(f"❌ RSI overbought ({rsi_now:.1f}) — potential reversal down")
        elif 40 < rsi_now < 60:
            signals.append(f"⚪ RSI neutral ({rsi_now:.1f})")

    # Volume signals
    if volume is not None and len(volume) > 20:
        avg_vol = volume.iloc[-20:].mean()
        cur_vol = volume.iloc[-1]
        price_chg = close.iloc[-1] - close.iloc[-2]
        if cur_vol > avg_vol * 1.5:
            if price_chg > 0:
                score += 1; signals.append("✅ High volume on up day — bullish confirmation")
            else:
                score -= 1; signals.append("❌ High volume on down day — bearish confirmation")

    return score, signals

def _score_to_label(score):
    if score >= 5:   return "🟢 Strong Buy",   "#2ecc71"
    elif score >= 2: return "🟡 Buy",           "#f0b429"
    elif score >= -1: return "⚪ Neutral",      "#3498db"
    elif score >= -4: return "🟡 Sell",         "#e67e22"
    else:            return "🔴 Strong Sell",   "#e74c3c"

# =========================
# Plotting
# =========================

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def _plot_price_ma(hist, smas, support, resistance, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0e1117")
    _dark_ax(ax1); _dark_ax(ax2)

    close = hist["Close"]
    ax1.plot(close.index, close.values, color="#1f77b4", linewidth=1.5, label="Price", zorder=5)

    colors_ma = {20: "#f0b429", 50: "#2ecc71", 200: "#e74c3c"}
    labels_ma = {20: "SMA 20", 50: "SMA 50", 200: "SMA 200"}
    for w, sma in smas.items():
        if pd.notna(sma.iloc[-1]):
            ax1.plot(sma.index, sma.values, color=colors_ma[w], linewidth=1,
                     linestyle="--", label=labels_ma[w], alpha=0.8)

    # Support/Resistance
    for s in support[:3]:
        ax1.axhline(s[1], color="#2ecc71", linewidth=0.8, linestyle=":", alpha=0.6)
        ax1.text(close.index[-1], s[1], f"  S: ${s[1]:.2f}", color="#2ecc71",
                 fontsize=7, va="center")
    for r in resistance[:3]:
        ax1.axhline(r[1], color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.6)
        ax1.text(close.index[-1], r[1], f"  R: ${r[1]:.2f}", color="#e74c3c",
                 fontsize=7, va="center")

    ax1.set_title(f"{ticker} — Price, Moving Averages & Support/Resistance", color="white", fontsize=11)
    ax1.set_ylabel("Price ($)", color="white", fontsize=9)
    ax1.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white", loc="upper left")

    # Volume
    if "Volume" in hist.columns:
        vol = hist["Volume"]
        price_chg = close.diff()
        vol_colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in price_chg]
        ax2.bar(vol.index, vol.values, color=vol_colors, alpha=0.7, width=1.0)
        avg_vol = vol.rolling(20).mean()
        ax2.plot(avg_vol.index, avg_vol.values, color="#f0b429", linewidth=1, label="20D Avg Vol")
        ax2.set_ylabel("Volume", color="white", fontsize=9)
        ax2.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white")

    plt.tight_layout(); return fig

def _plot_macd(close, macd, signal_line, histogram, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0e1117")
    _dark_ax(ax1); _dark_ax(ax2)

    ax1.plot(close.index, close.values, color="#1f77b4", linewidth=1.5, label="Price")
    ax1.set_title(f"{ticker} — MACD Analysis", color="white", fontsize=11)
    ax1.set_ylabel("Price ($)", color="white", fontsize=9)
    ax1.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white")

    ax2.plot(macd.index, macd.values, color="#1f77b4", linewidth=1.2, label="MACD")
    ax2.plot(signal_line.index, signal_line.values, color="#e74c3c", linewidth=1.2, label="Signal")
    hist_colors = ["#2ecc71" if h >= 0 else "#e74c3c" for h in histogram]
    ax2.bar(histogram.index, histogram.values, color=hist_colors, alpha=0.7, width=1.0, label="Histogram")
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)

    # Mark crossovers
    for i in range(1, len(macd)):
        if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
            ax2.axvline(macd.index[i], color="#2ecc71", linewidth=1, alpha=0.5)
        elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
            ax2.axvline(macd.index[i], color="#e74c3c", linewidth=1, alpha=0.5)

    ax2.set_ylabel("MACD", color="white", fontsize=9)
    ax2.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white")
    plt.tight_layout(); return fig

def _plot_rsi_bb(close, rsi, bb_mid, bb_upper, bb_lower, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0e1117")
    _dark_ax(ax1); _dark_ax(ax2)

    ax1.plot(close.index, close.values, color="#1f77b4", linewidth=1.5, label="Price")
    ax1.plot(bb_mid.index, bb_mid.values, color="#f0b429", linewidth=1, linestyle="--", label="BB Mid", alpha=0.8)
    ax1.plot(bb_upper.index, bb_upper.values, color="#e74c3c", linewidth=0.8, linestyle=":", label="BB Upper", alpha=0.7)
    ax1.plot(bb_lower.index, bb_lower.values, color="#2ecc71", linewidth=0.8, linestyle=":", label="BB Lower", alpha=0.7)
    ax1.fill_between(bb_upper.index, bb_upper.values, bb_lower.values, alpha=0.05, color="#f0b429")
    ax1.set_title(f"{ticker} — Bollinger Bands & RSI", color="white", fontsize=11)
    ax1.set_ylabel("Price ($)", color="white", fontsize=9)
    ax1.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white")

    ax2.plot(rsi.index, rsi.values, color="#9b59b6", linewidth=1.2, label="RSI (14)")
    ax2.axhline(70, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(30, color="#2ecc71", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(50, color="white", linewidth=0.5, alpha=0.2)
    ax2.fill_between(rsi.index, rsi.values, 70, where=(rsi > 70), alpha=0.2, color="#e74c3c")
    ax2.fill_between(rsi.index, rsi.values, 30, where=(rsi < 30), alpha=0.2, color="#2ecc71")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="white", fontsize=9)
    ax2.legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white")
    plt.tight_layout(); return fig

# =========================
# Main Page
# =========================

def run_technical_analysis():
    st.subheader("📐 MACD & Technical Signals")
    st.markdown(
        "Full technical analysis dashboard combining **MACD**, **Moving Averages**, "
        "**Support & Resistance**, **Bollinger Bands**, **RSI**, and **Volume** into a single "
        "combined signal score. Use before every trade on Investopedia."
    )

    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
**Indicators Included:**

| Indicator | Signal | Bullish | Bearish |
|-----------|--------|---------|---------|
| **MACD (12/26/9)** | Trend momentum | MACD > Signal line | MACD < Signal line |
| **SMA 20/50/200** | Trend direction | Price above SMAs | Price below SMAs |
| **Golden/Death Cross** | Major trend shift | SMA20 crosses above SMA50 | SMA20 crosses below SMA50 |
| **RSI (14)** | Momentum | < 30 oversold | > 70 overbought |
| **Bollinger Bands** | Volatility | Price below lower band | Price above upper band |
| **Support/Resistance** | Key price levels | Price at support | Price at resistance |
| **Volume** | Confirmation | High volume on up day | High volume on down day |

**Combined Signal Score:**
- +5 or above = Strong Buy
- +2 to +4 = Buy
- -1 to +1 = Neutral
- -4 to -2 = Sell
- -5 or below = Strong Sell

**How to use with your trading workflow:**
1. Check **Market Regime** — is the overall environment Risk-On or Risk-Off?
2. Check **Mean Reversion Scanner** — is the asset oversold or overbought?
3. Run **MACD & Technical Signals** — confirm with MACD crossover and moving averages
4. Check **Economic Calendar** — no major events coming?
5. If all align → place trade on Investopedia
""")

    # ---- Settings ----
    st.markdown("### ⚙️ Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="SPY", key="ta_ticker").upper().strip()
    with col2:
        period = st.selectbox("Lookback Period", ["3mo", "6mo", "1y", "2y"], index=2, key="ta_period")
    with col3:
        macd_fast  = st.number_input("MACD Fast", value=12, min_value=5,  max_value=50,  key="ta_fast")
        macd_slow  = st.number_input("MACD Slow", value=26, min_value=10, max_value=100, key="ta_slow")
        macd_sig   = st.number_input("MACD Signal", value=9, min_value=3, max_value=30,  key="ta_sig")

    run = st.button("▶ Run Technical Analysis", key="ta_run")
    if not run:
        st.info("Enter a ticker and click Run Technical Analysis.")
        return

    # ---- Fetch ----
    with st.spinner(f"Fetching {ticker} data..."):
        hist = _fetch_data(ticker, period=period)

    if hist is None or hist.empty:
        st.error(f"Could not fetch data for {ticker}. Please check the ticker.")
        return

    if len(hist) < 60:
        st.error("Not enough data. Try a longer period.")
        return

    close  = hist["Close"]
    volume = hist["Volume"] if "Volume" in hist.columns else None
    high   = hist["High"]   if "High"   in hist.columns else close
    low    = hist["Low"]    if "Low"    in hist.columns else close

    st.success(f"Loaded {len(hist):,} days for {ticker} from {hist.index.min().date()} to {hist.index.max().date()}")

    # ---- Compute Indicators ----
    macd, signal_line, histogram = _compute_macd(close, macd_fast, macd_slow, macd_sig)
    smas     = _compute_sma(close)
    rsi      = _compute_rsi(close)
    bb_mid, bb_upper, bb_lower = _compute_bollinger(close)
    atr      = _compute_atr(high, low, close)
    support, resistance = _find_support_resistance(close)
    score, signals = _compute_signal_score(close, macd, signal_line, histogram, smas, rsi, volume)
    label, color = _score_to_label(score)

    # ---- Overall Signal ----
    st.markdown("### 🎯 Overall Technical Signal")
    st.markdown(
        f"""
        <div style="
            background-color: {color}22;
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 2rem; font-weight: 700; color: {color};">{label}</div>
            <div style="font-size: 0.9rem; color: #ccc; margin-top: 4px;">
                Combined signal score: {score:+d} | {ticker} as of {hist.index[-1].strftime('%b %d, %Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Key Metrics ----
    st.markdown("### 📊 Current Readings")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",      f"${close.iloc[-1]:.2f}")
    c2.metric("SMA 20",     f"${smas[20].iloc[-1]:.2f}" if pd.notna(smas[20].iloc[-1]) else "N/A",
              delta="Above" if close.iloc[-1] > smas[20].iloc[-1] else "Below")
    c3.metric("SMA 50",     f"${smas[50].iloc[-1]:.2f}" if pd.notna(smas[50].iloc[-1]) else "N/A",
              delta="Above" if close.iloc[-1] > smas[50].iloc[-1] else "Below")
    c4.metric("SMA 200",    f"${smas[200].iloc[-1]:.2f}" if pd.notna(smas[200].iloc[-1]) else "N/A",
              delta="Above" if pd.notna(smas[200].iloc[-1]) and close.iloc[-1] > smas[200].iloc[-1] else "Below")
    c5.metric("RSI (14)",   f"{rsi.iloc[-1]:.1f}" if pd.notna(rsi.iloc[-1]) else "N/A")
    c6.metric("ATR (14)",   f"${atr.iloc[-1]:.2f}" if pd.notna(atr.iloc[-1]) else "N/A")

    c7, c8, c9, c10 = st.columns(4)
    c7.metric("MACD",       f"{macd.iloc[-1]:.4f}")
    c8.metric("Signal Line",f"{signal_line.iloc[-1]:.4f}")
    c9.metric("Histogram",  f"{histogram.iloc[-1]:.4f}",
              delta="Bullish" if histogram.iloc[-1] > 0 else "Bearish")
    c10.metric("Signal Score", f"{score:+d}")

    # ---- Signal Breakdown ----
    st.markdown("### 📋 Signal Breakdown")
    for sig in signals:
        if "✅" in sig:
            st.success(sig)
        elif "❌" in sig:
            st.error(sig)
        else:
            st.info(sig)

    # ---- Support & Resistance ----
    st.markdown("### 🔑 Key Price Levels")
    col_s, col_r = st.columns(2)
    with col_s:
        st.markdown("**🟢 Support Levels**")
        if support:
            for s in support[:5]:
                dist = (close.iloc[-1] - s[1]) / close.iloc[-1] * 100
                st.markdown(f"- **${s[1]:.2f}** — {dist:.1f}% below current price")
        else:
            st.info("No support levels identified.")
    with col_r:
        st.markdown("**🔴 Resistance Levels**")
        if resistance:
            for r in resistance[:5]:
                dist = (r[1] - close.iloc[-1]) / close.iloc[-1] * 100
                st.markdown(f"- **${r[1]:.2f}** — {dist:.1f}% above current price")
        else:
            st.info("No resistance levels identified.")

    # ---- Charts ----
    st.markdown("### 📈 Price, Moving Averages & Volume")
    st.pyplot(_plot_price_ma(hist, smas, support, resistance, ticker))

    st.markdown("### 📊 MACD")
    st.pyplot(_plot_macd(close, macd, signal_line, histogram, ticker))

    st.markdown("### 📉 Bollinger Bands & RSI")
    st.pyplot(_plot_rsi_bb(close, rsi, bb_mid, bb_upper, bb_lower, ticker))

    # ---- Trade Guidance ----
    st.markdown("### 💡 Trade Guidance")
    price_now = close.iloc[-1]
    atr_now   = atr.iloc[-1] if pd.notna(atr.iloc[-1]) else price_now * 0.02

    if score >= 2:
        nearest_support = support[0][1] if support else price_now * 0.97
        st.success(f"""
**{label} on {ticker}**
- Consider a **long position** on Investopedia
- Suggested stop loss: **${nearest_support:.2f}** (nearest support) or **${price_now - 2*atr_now:.2f}** (2× ATR below)
- Risk per trade: **${price_now - (price_now - 2*atr_now):.2f}** per share
- Confirm with: Market Regime, Mean Reversion Scanner, Economic Calendar
""")
    elif score <= -2:
        nearest_resistance = resistance[0][1] if resistance else price_now * 1.03
        st.error(f"""
**{label} on {ticker}**
- Consider a **short position or put option** on Investopedia
- Suggested stop loss: **${nearest_resistance:.2f}** (nearest resistance) or **${price_now + 2*atr_now:.2f}** (2× ATR above)
- Risk per trade: **${(price_now + 2*atr_now) - price_now:.2f}** per share
- Confirm with: Market Regime, Mean Reversion Scanner, Economic Calendar
""")
    else:
        st.info(f"""
**{label} on {ticker}**
- No clear directional signal — wait for a stronger setup
- Watch for: MACD crossover, RSI moving out of neutral zone, price breaking support/resistance
- Current ATR: ${atr_now:.2f} — use this for position sizing
""")

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance | Refreshes every 5 minutes"
    )
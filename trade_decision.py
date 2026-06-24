# trade_decision.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# =========================
# Data Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price_data(ticker: str, period: str = "1y"):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist.dropna()
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_spy():
    try:
        tk = yf.Ticker("IVV")
        hist = tk.history(period="1y", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist["Close"].dropna()
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_vix():
    try:
        tk = yf.Ticker("^VIX")
        hist = tk.history(period="5d", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None

# =========================
# Signal Calculations
# =========================

def _compute_macd_signal(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    if macd.iloc[-1] > signal.iloc[-1] and hist.iloc[-1] > 0:
        return "✅ Bullish", 1
    elif macd.iloc[-1] < signal.iloc[-1] and hist.iloc[-1] < 0:
        return "❌ Bearish", -1
    else:
        return "⚪ Neutral", 0

def _compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _compute_mean_reversion_signal(close):
    rsi = _compute_rsi(close)
    rsi_now = rsi.iloc[-1]
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    zscore = ((close - sma20) / std20).iloc[-1]
    bb_upper = (sma20 + 2 * std20).iloc[-1]
    bb_lower = (sma20 - 2 * std20).iloc[-1]
    price = close.iloc[-1]

    if rsi_now < 30 and zscore < -2:
        return "✅ Oversold — Strong Buy Signal", 2, rsi_now, zscore
    elif rsi_now < 40 and zscore < -1:
        return "🟡 Mildly Oversold", 1, rsi_now, zscore
    elif rsi_now > 70 and zscore > 2:
        return "❌ Overbought — Avoid", -2, rsi_now, zscore
    elif rsi_now > 60 and zscore > 1:
        return "🟡 Mildly Overbought", -1, rsi_now, zscore
    else:
        return "⚪ Neutral", 0, rsi_now, zscore

def _compute_trend_signal(close):
    sma20  = close.rolling(20).mean().iloc[-1]
    sma50  = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    price  = close.iloc[-1]
    score  = 0
    details = []
    if pd.notna(sma20):
        if price > sma20: score += 1; details.append("Above SMA20")
        else: score -= 1; details.append("Below SMA20")
    if pd.notna(sma50):
        if price > sma50: score += 1; details.append("Above SMA50")
        else: score -= 1; details.append("Below SMA50")
    if sma200 and pd.notna(sma200):
        if price > sma200: score += 1; details.append("Above SMA200 ✅")
        else: score -= 1; details.append("Below SMA200 ❌")
    return score, details

def _compute_relative_strength(close, spy_close):
    if spy_close is None or len(close) < 22 or len(spy_close) < 22:
        return "⚪ N/A", 0
    ret_1m  = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0
    spy_1m  = (spy_close.iloc[-1] / spy_close.iloc[-22] - 1) * 100 if len(spy_close) >= 22 else 0
    ret_3m  = (close.iloc[-1] / close.iloc[-66] - 1) * 100 if len(close) >= 66 else ret_1m
    spy_3m  = (spy_close.iloc[-1] / spy_close.iloc[-66] - 1) * 100 if len(spy_close) >= 66 else spy_1m
    rs_1m   = ret_1m - spy_1m
    rs_3m   = ret_3m - spy_3m
    if rs_1m > 5 and rs_3m > 5:
        return f"✅ Strong Leader (+{rs_1m:.1f}% vs SPY 1M)", 2
    elif rs_1m > 0:
        return f"🟡 Leader (+{rs_1m:.1f}% vs SPY 1M)", 1
    elif rs_1m > -5:
        return f"⚪ Neutral ({rs_1m:.1f}% vs SPY 1M)", 0
    elif rs_1m > -10:
        return f"🟠 Laggard ({rs_1m:.1f}% vs SPY 1M)", -1
    else:
        return f"❌ Strong Laggard ({rs_1m:.1f}% vs SPY 1M)", -2

def _compute_atr(hist, period=14):
    high  = hist["High"]  if "High"  in hist.columns else hist["Close"]
    low   = hist["Low"]   if "Low"   in hist.columns else hist["Close"]
    close = hist["Close"]
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def _var_estimate(close, confidence=0.95):
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return None
    return float(-np.percentile(returns.tail(252), (1-confidence)*100) * 100)

# =========================
# Verdict
# =========================

def _compute_verdict(total_score, max_score):
    pct = total_score / max_score if max_score > 0 else 0
    if pct >= 0.6:
        return "🟢 PAPER BUY", "#2ecc71", "Strong alignment across models. Consider entering a long position on Investopedia."
    elif pct >= 0.3:
        return "🟡 WATCH", "#f0b429", "Mixed signals. Wait for more confirmation before entering."
    elif pct >= -0.1:
        return "⚪ NEUTRAL", "#3498db", "No clear edge. Sit on the sidelines."
    elif pct >= -0.4:
        return "🟠 CAUTION", "#e67e22", "More signals bearish than bullish. Avoid new longs."
    else:
        return "🔴 AVOID", "#e74c3c", "Strong bearish alignment. Do not enter. Wait for reversal confirmation."

# =========================
# Main Page
# =========================

def run_trade_decision():
    st.subheader("🎯 Trade Decision Dashboard")
    st.markdown(
        "Enter a ticker and this page automatically runs **all your Apollo Quant models** "
        "and combines them into a single **Buy / Watch / Avoid** verdict. "
        "Use this before every Investopedia paper trade."
    )

    st.markdown("### ⚙️ Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker to Analyze", value="SPY", key="td_ticker").upper().strip()
    with col2:
        account_size = st.number_input("Investopedia Account Size ($)", value=100000,
                                        min_value=1000, step=1000, key="td_account")
        max_risk_pct = st.number_input("Max Risk Per Trade (%)", value=1.0,
                                        min_value=0.1, max_value=5.0, step=0.1, key="td_risk")
    with col3:
        regime_override = st.selectbox("Current Market Regime",
            ["Auto-detect from site", "Risk-On 🟢", "Risk-Off 🔴", "Inflationary 🟠", "Recessionary ⚫", "Neutral 🔵"],
            key="td_regime")
        has_event = st.checkbox("⚠️ Major event in next 48hrs (FOMC/CPI/NFP)?", key="td_event")

    run = st.button("▶ Run Trade Decision", key="td_run", type="primary")
    if not run:
        st.info("Enter a ticker and click Run Trade Decision.")
        return

    with st.spinner(f"Running all models on {ticker}..."):
        hist = _fetch_price_data(ticker)
        spy  = _fetch_spy()
        vix  = _fetch_vix()

    if hist is None or hist.empty or len(hist) < 60:
        st.error(f"Could not fetch sufficient data for {ticker}.")
        return

    close  = hist["Close"]
    price  = float(close.iloc[-1])
    date   = hist.index[-1].strftime("%b %d, %Y")

    # ---- Run all models ----
    macd_label, macd_score = _compute_macd_signal(close)
    mr_label, mr_score, rsi_val, zscore_val = _compute_mean_reversion_signal(close)
    trend_score, trend_details = _compute_trend_signal(close)
    rs_label, rs_score = _compute_relative_strength(close, spy)
    var_1d = _var_estimate(close)
    atr_val = _compute_atr(hist)
    stop_loss = round(price - 2 * atr_val, 2)
    risk_per_share = round(price - stop_loss, 2)
    shares = int((account_size * max_risk_pct / 100) / risk_per_share) if risk_per_share > 0 else 0
    position_size = round(shares * price, 2)

    # Regime scoring
    regime = regime_override if regime_override != "Auto-detect from site" else "Neutral 🔵"
    regime_score = 0
    if "Risk-On" in regime:   regime_score = 2
    elif "Risk-Off" in regime: regime_score = -2
    elif "Inflationary" in regime: regime_score = 0
    elif "Recessionary" in regime: regime_score = -2
    else: regime_score = 0

    # Event penalty
    event_score = -3 if has_event else 0

    # VaR check
    var_ok = var_1d is not None and var_1d < 3.0
    var_score = 1 if var_ok else -1
    var_label = f"✅ Acceptable ({var_1d:.2f}%/day)" if var_ok else f"❌ Elevated ({var_1d:.2f}%/day — consider smaller size)" if var_1d else "⚪ N/A"

    # Trend label
    if trend_score >= 2:   trend_label = "✅ Strong Uptrend"
    elif trend_score == 1: trend_label = "🟡 Mild Uptrend"
    elif trend_score == 0: trend_label = "⚪ Neutral"
    elif trend_score == -1: trend_label = "🟠 Mild Downtrend"
    else:                   trend_label = "❌ Downtrend"

    # VIX label
    if vix is not None:
        if vix < 15:   vix_label = f"✅ Low ({vix:.1f}) — calm market"
        elif vix < 20: vix_label = f"🟡 Moderate ({vix:.1f})"
        elif vix < 25: vix_label = f"🟠 Elevated ({vix:.1f}) — caution"
        else:          vix_label = f"❌ High ({vix:.1f}) — risk-off environment"
        vix_score = 1 if vix < 20 else (-1 if vix > 25 else 0)
    else:
        vix_label = "⚪ N/A"; vix_score = 0

    event_label = "❌ YES — Reduce size or wait" if has_event else "✅ No major events in next 48hrs"

    # Total score
    total_score = (regime_score + macd_score + mr_score + rs_score +
                   trend_score + var_score + vix_score + event_score)
    max_score = 12  # rough maximum if everything bullish
    verdict_label, verdict_color, verdict_detail = _compute_verdict(total_score, max_score)

    # =============================
    # DISPLAY
    # =============================

    st.markdown(f"### 📊 Analysis for **{ticker}** — ${price:.2f} as of {date}")

    # Verdict banner
    st.markdown(
        f"""<div style="background:{verdict_color}22;border-left:6px solid {verdict_color};
        border-radius:8px;padding:20px 24px;margin-bottom:16px;">
        <div style="font-size:2rem;font-weight:800;color:{verdict_color};">{verdict_label}</div>
        <div style="color:#ccc;font-size:0.95rem;margin-top:6px;">{verdict_detail}</div>
        <div style="color:#888;font-size:0.82rem;margin-top:4px;">
        Total score: {total_score:+d} | {ticker} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
        </div></div>""",
        unsafe_allow_html=True
    )

    # Model checklist
    st.markdown("### ✅ Model Checklist")
    checks = [
        ("Market Regime",        regime,       regime_score),
        ("VIX / Volatility",     vix_label,    vix_score),
        ("Relative Strength",    rs_label,     rs_score),
        ("MACD Signal",          macd_label,   macd_score),
        ("Mean Reversion",       mr_label,     mr_score),
        ("Trend (SMAs)",         trend_label,  trend_score),
        ("VaR Check",            var_label,    var_score),
        ("Economic Calendar",    event_label,  event_score),
    ]

    for name, label, sc in checks:
        bg = "#2ecc7115" if sc > 0 else ("#e74c3c15" if sc < 0 else "#3498db10")
        border = "#2ecc71" if sc > 0 else ("#e74c3c" if sc < 0 else "#3498db")
        icon = "✅" if sc > 0 else ("❌" if sc < 0 else "⚪")
        st.markdown(
            f"""<div style="background:{bg};border-left:3px solid {border};
            padding:8px 14px;margin-bottom:6px;border-radius:0 6px 6px 0;">
            <span style="color:{border};font-weight:600;">{icon} {name}</span>
            <span style="color:#ccc;font-size:0.88rem;margin-left:12px;">{label}</span>
            <span style="color:{border};font-size:0.8rem;float:right;">{sc:+d}</span>
            </div>""",
            unsafe_allow_html=True
        )

    # Key readings
    st.markdown("### 📋 Key Readings")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price",         f"${price:.2f}")
    c2.metric("RSI (14)",      f"{rsi_val:.1f}")
    c3.metric("Z-Score",       f"{zscore_val:.2f}")
    c4.metric("ATR (14)",      f"${atr_val:.2f}")
    c5.metric("1D VaR (95%)",  f"{var_1d:.2f}%" if var_1d else "N/A")

    # Position sizing
    st.markdown("### 💰 Position Sizing")
    st.markdown(
        f"""<div style="border:1px solid #f0b42966;border-radius:8px;padding:16px 20px;background:#f0b42908;">
        <div style="font-weight:700;color:#f0b429;margin-bottom:10px;">📐 Recommended Position Size</div>
        <table style="width:100%;color:#ccc;font-size:0.88rem;border-collapse:collapse;">
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;width:200px;"><b>Entry Price</b></td><td>${price:.2f}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Stop Loss (2x ATR)</b></td><td>${stop_loss:.2f}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Risk Per Share</b></td><td>${risk_per_share:.2f}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Max Risk ({max_risk_pct}% of ${account_size:,})</b></td><td>${account_size * max_risk_pct / 100:,.0f}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Suggested Shares</b></td><td><b>{shares} shares</b></td></tr>
        <tr><td style="padding:5px 0;"><b>Total Position Value</b></td><td><b>${position_size:,.0f} ({position_size/account_size*100:.1f}% of portfolio)</b></td></tr>
        </table></div>""",
        unsafe_allow_html=True
    )

    if position_size / account_size > 0.25:
        st.warning("⚠️ Position exceeds 25% of portfolio — consider reducing size per risk rules.")

    # Trade entry summary
    st.markdown("### 📝 Trade Entry Summary")
    st.markdown(
        f"""<div style="border:1px solid #ffffff22;border-radius:8px;padding:16px 20px;background:#0d1420;">
        <table style="width:100%;color:#ccc;font-size:0.88rem;border-collapse:collapse;">
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;width:200px;"><b>Ticker</b></td><td>{ticker}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Action</b></td>
        <td style="color:{'#2ecc71' if total_score > 0 else '#e74c3c'};">{'BUY' if total_score > 2 else ('WATCH' if total_score >= 0 else 'AVOID')}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Entry</b></td><td>${price:.2f}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Stop Loss</b></td><td>${stop_loss:.2f} (2x ATR below entry)</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Shares</b></td><td>{shares}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Position Size</b></td><td>${position_size:,.0f}</td></tr>
        <tr style="border-bottom:1px solid #333;"><td style="padding:5px 0;"><b>Regime</b></td><td>{regime}</td></tr>
        <tr><td style="padding:5px 0;"><b>Date</b></td><td>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</td></tr>
        </table>
        <div style="margin-top:12px;color:#888;font-size:0.8rem;">
        Log this trade in the Performance Dashboard with your thesis before placing it on Investopedia.
        </div></div>""",
        unsafe_allow_html=True
    )

    trend_detail_str = " | ".join(trend_details) if trend_details else "N/A"
    st.caption(
        f"Trend details: {trend_detail_str} | "
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance"
    )
